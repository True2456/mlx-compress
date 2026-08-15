# DeepSeek-V4-Flash REAP: Measured Findings

**Model:** DeepSeek-V4-Flash — 284B (preview) / 304B (`-0731` release) MoE, 43
decoder layers, **256 routed experts/layer**, 1 shared expert, HISA attention,
Hyper-Connections (`hc_mult=4`), first `num_hash_layers=3` layers route via a
**fixed token→expert hash table**, not learned top-k. Native tool-calling format
is **DSML** (`｜DSML｜` markup, distinct from generic `<tool_call>` conventions).

**Hardware:** M5 Max, unified memory, served via oMLX (not LM Studio — see below).

**Scope:** Everything below is measured on this machine this session
(2026-08-01/02), not asserted. Two multi-hour debugging threads are included in
full because the root causes were non-obvious and expensive to find.

---

## TL;DR

| # | Finding | Magnitude | Status |
|---|---|---|---|
| 1 | Hash-routed layers need REAM (merge), not REAP (delete) | structural requirement, not a quality choice | ✅ done |
| 2 | Checkpoint ships **VLM-shaped** even though the model is text-only | `language_model.` prefix on every tensor | ✅ fixed (in-place rewrite) |
| 3 | Garbled tool calls were a missing `tools` block in our chat template | not a DSML parsing problem | ✅ fixed |
| 4 | oMLX has native DSML support but only reaches it via the **text** load path | `vision_config` presence routes to `mlx_vlm` instead | ✅ fixed |
| 5 | LM Studio's classifier blanks all metadata on standard HF tokenizer output | `AddedToken` dict form for bos/eos/pad | found; **do not fix** (breaks generation) |
| 6 | Full-suite benchmarks (n=200 MMLU/GSM8K, n=164 HumanEval) across three builds | see §9 — **p15 wins on every metric**, by a wide margin on two of three | ✅ measured |
| 7 | "Prune only, preserve native precision" hypothesis (§7 below) — **tested head-to-head and did not hold up** | p37-native (0% extra quant) scored *worse* than p15 (2-bit) on all three benchmarks | ⚠️ **hypothesis rejected by data**, see §9 |
| 8 | HF upload hung indefinitely on Hub API calls specifically (data PUTs to S3 worked) | root cause found in library source (`timeout=None`), fix didn't fully resolve it | ⚠️ **unresolved**, likely network-path issue |

---

## 1. Hash-routed layers: REAM is structural, not a quality preference

`n_routed_experts` is one global config field with no per-layer override, so
every layer must end at the same width after pruning. Plain REAP deletion on a
hash-routed layer leaves `tid2eid` (the fixed token→expert lookup) pointing at
removed experts with no principled fallback — confirmed by running the naive
version and hitting a shape-mismatch load error.

REAM's `assign_merges` (router-weight cosine similarity) gives every deleted
expert a destination and remaps `tid2eid` in the same step. This is **not** a
re-endorsement of REAM as a general compression strategy — it was built,
measured, and rejected for Step-3.7 (`REAM-RESULT.md`: PPL "gain" was
smoothing, real accuracy flat-to-worse). It's used here only because plain
deletion is structurally broken for these 3 layers specifically; the other 40
layers use plain REAP deletion.

## 2. The checkpoint is VLM-shaped for a text-only model

Both `models/DeepSeek-V4-Flash-fp8` and `-0731` are converted through
`mlx_vlm`'s wrapper, so every saved tensor carries a `language_model.` prefix
(`language_model.lm_head.weight`, etc.) even though the model has **zero**
vision tensors — confirmed via header inspection (1902/1822 tensors, 100%
prefixed, no `vision_*` weights). This is an artifact of the conversion
pipeline, not a property of the model.

**Fix — in-place safetensors header rewrite, no data touched:**

```python
# safetensors layout: [8-byte header length][JSON header][raw data]
# Stripping "language_model." only shortens key names, so the new header
# is guaranteed <= the old one. Pad with spaces to the identical byte
# length and every data offset stays valid -- rewrite ~30KB, not 96GB.
with open(path, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
new_hdr = {("format" if k == "__metadata__" else k.removeprefix("language_model.")): v
           for k, v in hdr.items()}
s = json.dumps(new_hdr, separators=(",", ":")).encode()
s += b" " * (n - len(s))  # pad to identical length
with open(path, "r+b") as f:
    f.seek(0); f.write(struct.pack("<Q", n)); f.write(s)
```

Verified via `mx.array_equal` spot checks across shards after sharding — always
verify tensor values are bit-identical after a header rewrite, not just that
the file loads.

**Why bother:** the raw checkpoint's weight names determine which load path an
engine picks (see §4). VLM-shaped tensors force the VLM path even for a
text-only model; renaming unlocks the text path and everything gated behind it.

Also required alongside the rename: drop `vision_config` from `config.json`,
and re-prefix any per-path `quantization`/`quantization_config` override keys
that still say `language_model.xxx` (config is not automatically kept in sync
with a manual header rewrite).

## 3. Garbled tool calls were a missing template block, not a DSML problem

`chat_template.jinja` had no `tools` handling at all — tool definitions were
silently discarded, `tool`-role messages vanished. The model, shown no tools,
improvised from training priors: Anthropic-style `<function_calls><invoke>`
(likely distillation contamination) blended with `string="true"` from its own
DSML training. Reproduced identically under LM Studio and oMLX because both
render the same file from the model directory — that cross-stack repro is what
pointed at the template rather than either serving engine.

Given the instructions in a plain system prompt (bypassing any `tools` API
field), the model emits **flawless** off-distribution qwen3-coder-style XML in
32 tokens — off-distribution formatting was never the real risk; an empty
template was.

## 4. oMLX has native DSML support, gated behind the text load path

`omlx/patches/deepseek_v4/` ships a full DSML chat template (port of
DeepSeek's own `encoding_dsv4.py`) and a DSML tool parser, injected by
patching `mlx_lm.tokenizer_utils.load`. Two things silently disable it:

- **A `chat_template.jinja` in the model directory.** The patch checks
  `if wrapper._chat_template is None` before installing its own — any file
  there wins, even a broken one. It still installs the DSML **parser**
  unconditionally, so a custom template + native parser is a real,
  confusing half-configured state (template says one format, parser expects
  another).
- **`vision_config` presence** (even `{}`) routes the model through `mlx_vlm`
  instead of `mlx_lm`, and the DSML patch only wraps the `mlx_lm` loader —
  never fires at all on the VLM path.

**Net effect:** for a model oMLX supports natively, the model directory should
have **no** `chat_template.jinja` and **no** `vision_config` key. If tool calls
appear as raw text in the client, check for those two first, in that order.

This directly conflicts with what LM Studio needs (§5 — `mlx_lm` there has no
`deepseek_v4` at all, so `vision_config: {}` is the only way to load it). The
two stacks want incompatible model-directory shapes for this architecture;
pick one deployment target per build.

## 5. LM Studio: standard HF tokenizer output blanks all model metadata

`AddedToken` dict form for `bos_token`/`eos_token`/`pad_token` in
`tokenizer_config.json` (`{"__type": "AddedToken", "content": ..., ...}`) — the
standard `tokenizers`-library output — causes LM Studio's classifier to
produce an **empty** `configJson`, cascading into blank arch/quantization
display and a hard 4096-token context cap in the UI. Reproduced with a
~600-byte throwaway model directory, independent of architecture, config size,
`vision_config`, or file layout. Converting the three fields to plain strings
fixes it immediately.

**Do not apply this fix.** With correct metadata, LM Studio stops delegating
prompt templating to `mlx_vlm` and crashes every request
(`temp_createMlxPredictionArgs` → `TypeError: undefined (reading 'template')`).
It also silently persists a broken `llm.prediction.promptTemplate` entry into
`~/.lmstudio/.internal/user-concrete-model-default-config/<owner>/<model>.json`
— `type: "jinja"` with no matching `jinjaPromptTemplate` object — which
survives unload/reload/server-restart and keeps the model broken until that
one field is deleted by hand. If a model that used to work suddenly can't
generate after any metadata-adjacent change, check that file before suspecting
the checkpoint.

## 6. Prune-ratio quality: reasoning survives, broad recall doesn't (early signal, n=100)

50% REAP+REAM prune (old checkpoint, `p50`), first tested against 100-sample
suites, before the full-suite run in §9:

| Benchmark | Result | Note |
|---|---|---|
| GSM8K (n=100) | 95% | margin of error ≈ ±10% at this n |
| MMLU (n=100) | 69% | same margin; real gap from GSM8K regardless |

The GSM8K-holds/MMLU-drops shape was directionally right (§9 confirms it at
n=200/164 on p50, and again on the other two builds) but the specific numbers
here were superseded — see §9 for the trustworthy version and for how p50
compares against p15 and against a completely different checkpoint at a
similar prune ratio.

## 7. Hypothesis: prune instead of quantizing further when the source is already low-bit — REJECTED, see §9

**This section originally argued for skipping additional quantization when a
checkpoint already ships at native low bit-width, on the theory that quant
damage generally dominates prune damage. §9's head-to-head benchmark
contradicts that: the build following this exact reasoning (p37-native, 0%
extra quant, more experts kept) scored *worse* on every metric than a build
using the older checkpoint with heavier quantization (p15, 2-bit). Read the
byte-math below as a correct method for hitting a size target, not as a
quality recommendation — the quality conclusion was wrong, or at minimum
doesn't transfer across checkpoint versions.**

The `-0731` checkpoint's experts are natively 4-bit (`mxfp4`-equivalent, real
`.scale` tensors per group, not `mx.quantize`'s affine format). Measured real
tensor bytes at native precision, 0% pruned:

| Category | Size | Scales with prune ratio? |
|---|---|---|
| `other_fixed` (attn/norms/embed/head/mtp) | 8.40 GiB | no |
| `down_proj` (all experts) | 45.69 GiB | yes, protected precision |
| hash-layer gate/up | 6.38 GiB | yes, protected precision |
| non-hash gate/up | 85.00 GiB | yes, protected precision |
| **Total** | **145.46 GiB** | |

The byte-budget equation itself is still correct and useful for *sizing* a
build (`target = other_fixed + kept_fraction × (down_proj + hash_gateup +
nonhash_gateup)`, solved for `ratio ≈ 0.367` → 95.67 GiB projected, 96 GiB
actual, within 0.4%) — just don't assume the resulting build is higher
quality than a smaller, more-quantized one built from a different checkpoint.
See §9 for what actually happened when tested.

## 9. Full-suite benchmark comparison (n=200 MMLU/GSM8K, n=164 HumanEval — 2026-08-02)

| Model | Base checkpoint | Prune ratio | Extra quant | Size | MMLU | GSM8K | HumanEval |
|---|---|---|---|---|---|---|---|
| **p15** | preview (284B) | 15% | 2-bit (non-hash gate/up) | 95GB | **59.0%** | **97.5%** | **89.0%** |
| p50 | preview (284B) | 50% | 4-bit (non-hash gate/up) | 79GB | 48.0% | 96.0% | 86.0% |
| 0731-p37-native | `-0731` (304B) | 37% | none (native 4-bit) | 96GB | 50.5% | 81.0% | 75.0% |
| Step-3.7-p15 *(different model family, reference only)* | — | 15% | shared8/head8 | — | 31.5% | 76.0% | 77.4% |

**p15 wins on every metric**, by a wide margin on GSM8K and HumanEval
(16.5pp and 14pp over p37-native — both far outside the ~±7%/±8% margin of
error at these sample sizes, i.e. real, not noise). This directly overturns
§7's hypothesis: p15 actually keeps *fewer* experts than p37-native (85% kept
at 15% prune vs p37-native's 63.7% kept at 37% prune) **and** carries extra
2-bit quantization damage p37-native doesn't have, yet still wins clean. The
variable that actually seems to matter is **which base checkpoint**, not the
prune/quant recipe applied to it — something about how REAP/REAM interacts
with the `-0731` checkpoint specifically (different native quantization
format for experts — real block-scaled fp4 vs the older checkpoint's
`mx.quantize` affine format; possibly the saliency calibration set
transferring worse to `-0731`'s actual expert-utilization patterns) costs
more than either the extra pruning or the extra quantization did on their
own. **Not yet isolated — this is the next thing worth a controlled test if
it matters:** run the *same* prune ratio and *same* quant scheme on both
checkpoints and compare, rather than varying three things (checkpoint,
ratio, quant) at once the way this comparison did.

One more thing worth checking before fully trusting the p37-native numbers:
its MMLU run finished in 298s vs 784s (p15) / 735s (p50) for the same 200
questions — 2.6x faster despite scoring worst. Plausibly benign (more
decisive answers), but worth spot-checking a few transcripts for truncated or
rushed reasoning before ruling that out as a contributor.

Benchmarks are also not the full picture — real usage impressions diverged
from this ranking (per the user, this data "doesn't necessarily represent
what I felt was better to use"), consistent with `PPL-DECOMPOSITION.md`'s own
caveat that perplexity/benchmark proxies aren't a substitute for real task
performance.

**Always compute the byte model from the actual loaded tensor shapes/dtypes**
(`mlx_vlm.load(..., lazy=True)`, walk `switch_mlp.{gate,up,down}_proj`), not
from raw on-disk safetensors bytes — the raw upstream checkpoint stores experts
as individually-keyed tensors in DeepSeek's own native format
(`layers.N.ffn.experts.E.{w1,w2,w3}.{weight,scale}`); the loader remaps and
requantizes into MLX's batched `switch_mlp` form, which can be a different
byte layout than the source file.

## 8. Upload hang: real root cause found, fix incomplete

`hf upload-large-folder` hung indefinitely, multiple ways, over ~2 hours of
diagnosis:

1. Default (Xet-enabled): `CLOSE_WAIT` socket, 0% CPU, zero progress — Xet
   negotiation stuck.
2. Xet disabled, plain LFS, single ~103GB file: `422 Unprocessable Entity` —
   **this account's LFS tier caps individual files at 50GB.** A real error,
   not a hang — the first useful signal all session.
3. Sharded to 25×~4GB files (verified byte-identical to source via
   `mx.array_equal` spot checks across shards): real multi-part upload
   progress (confirmed via `nettop` on the process — coarse per-file
   progress counters don't reflect in-flight bytes, watch actual socket
   throughput, not the CLI's own status line).
4. Progress repeatedly stalls at an identical byte count (4549 or 5757 bytes)
   on a **specific class of request** — not the large S3 data PUTs (which
   completed real parts, 14-29MB each, multiple times) but small follow-up
   calls to `huggingface.co` itself (next-part URL, commit/verify).
5. Found the actual library bug via source read:
   `huggingface_hub.utils._http.default_client_factory()` constructs its
   `httpx.Client` with **`timeout=None`** — disabled entirely. A truly dead
   connection never raises, so the library's own `http_backoff` retry logic
   (which does correctly retry on `httpx.TimeoutException`) never gets a
   chance to run.
6. Patched via the public `set_client_factory()` API with a finite timeout
   (connect=30s, read/write=120s). **Did not fully fix it** — the exact same
   stall recurred past the configured timeout on a later run, meaning
   whatever's blocking may sit below the `httpx` layer entirely (DNS
   resolution via `socket.getaddrinfo` is a plausible remaining suspect;
   `httpx`'s timeout doesn't reliably cover it).

**Status: unresolved.** Working hypothesis is a network-path issue specific to
reaching Hugging Face's own API endpoints (not S3) from this network — data
transfers work, a particular class of control-plane call doesn't, consistently,
across every combination of Xet/no-Xet/IPv4-forced/timeout-patched. Untested:
same upload from a different network. The timeout patch (script below) is
still worth using regardless — it turns silent infinite hangs into failures
that at least show *something* moved, which none of the un-patched attempts did.

```python
# upload_with_timeout.py -- run with the hf tool's OWN python, not a project venv
# (they can be different huggingface_hub versions/install locations)
import sys, httpx
from huggingface_hub.utils._http import hf_request_event_hook, set_client_factory
def _client_factory():
    return httpx.Client(
        event_hooks={"request": [hf_request_event_hook]},
        follow_redirects=True,
        timeout=httpx.Timeout(connect=30.0, read=120.0, write=120.0, pool=30.0),
    )
set_client_factory(_client_factory)
from huggingface_hub.cli.hf import main
sys.argv = ["hf"] + sys.argv[1:]
sys.exit(main())
```

---

## Related docs

- `PPL-DECOMPOSITION.md` — the quant-vs-prune methodology this session's §6/§7
  reasoning is built on (measured on Step-3.7, not yet replicated here).
- `REAM-RESULT.md` — why REAM is not a general strategy, only used here for
  hash-routed layers specifically (§1).
- `TOKENIZER-INVESTIGATION.md` — the Step-3.7 tokenizer bug with the same
  "config field silently changes which code path loads" shape as §5.
