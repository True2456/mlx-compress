# q38 engine, the MTP norm defect, and the speculative-decoding landscape

Session of 2026-08-16. Covers the standalone Qwen3.8 inference engine, a real
defect found in our published checkpoints, and a survey of every speculative
option that applies to this model — with measurements.

Engine code: `~/AppleLLM/q38_native_engine/` (see its own README for operating
detail). Model tooling: `reap_stream/fix_mtp_norms_qwen35.py`.

---

## 1. Headline results

| | decode |
|---|---|
| q38, MTP off | 24.6 tok/s |
| q38, MTP with the head as originally shipped | 22.4 tok/s |
| **q38, MTP with repaired head** | **43.7–47.3 tok/s** |
| oMLX app, same model | 42.8 tok/s |

MTP was a **net loss** until the checkpoint's MTP head was repaired. That single
fix is the difference between speculation being worth turning on and being worth
turning off, and it is the most important thing in this document.

Quality of the 5.0bpw build, measured with MTP off (MTP is rejection-verified
and cannot change outputs):

| benchmark | 5.0bpw gs64 | 4.85bpw gs128 |
|---|---|---|
| HumanEval | 91.5% (150/164) | 93.3% (153/164) |
| GSM8K | 92.5% (185/200) | 92.0% (184/200) |
| MMLU | 84.0% (168/200) | 83.0% (166/200) |
| **total** | **503/564** | **503/564** |

An exact tie. The extra 0.53 GB buys prefill speed without an oMLX patch and a
calibrated vision tower, not accuracy.

---

## 2. The MTP norm defect (fixed, published)

### What was wrong

Qwen3-Next stores every RMSNorm gamma zero-centred; MLX wants them centred on
one, and conversion adds 1.0. Our AWQ builds carried the MTP head across from
the *raw HF* source (`carry_mtp_weights.py`) **after** the backbone had already
been converted, so both published checkpoints shipped mixed conventions:

| tensor | as shipped | should be |
|---|---|---|
| backbone `layers.0.input_layernorm` | 0.9666 | correct |
| `mtp.layers.0.input_layernorm` | **0.0361** | 1.0361 |
| `mtp.layers.0.post_attention_layernorm` | 0.2063 | 1.2063 |
| `mtp.layers.0.self_attn.k_norm` | 0.7795 | 1.7795 |
| `mtp.layers.0.self_attn.q_norm` | 0.7906 | 1.7906 |
| `mtp.norm` | 1.2520 | 2.2520 |
| `mtp.pre_fc_norm_embedding` | **−0.4606** | 0.5394 |
| `mtp.pre_fc_norm_hidden` | −0.1572 | 0.8428 |

The head's first layernorm multiplied by 0.036 instead of ~1.036. Drafts stopped
matching the target, acceptance collapsed, and MTP became pure overhead. Nothing
crashed; output stayed correct, because drafts are rejection-verified.

### Decide the convention per HEAD, never per tensor

This is the generalisable lesson. Raw `q_norm`/`k_norm` average ~0.78 and raw
`mtp.norm` ~1.25, so **any** per-key `mean < 0.5` threshold fixes the low norms
and silently leaves those three raw.

- oMLX's `norm_repair` has exactly this bug — it recovers 3 of 7. Its own source
  notes the miss costs ~14pp of draft acceptance.
- The first version of our fix reproduced the bug independently, refusing on
  `k_norm` at 0.7795.

Use the **lowest** norm in the head as the signal: zero-centred gammas go
negative, one-centred gammas cannot. Then shift uniformly.

### Effect, by loader

| loader | norms correct before fix | gain from fix |
|---|---|---|
| q38 / stock mlx_vlm | 0 of 7 | 22.4 → 43.7 tok/s (**+95%**) |
| oMLX (has `norm_repair`) | 3 of 7 | **+9%** (median 53.3 vs 48.8) |

The oMLX figure is from a controlled test — `pp 4096 / tg 128`, three runs each,
alternating — after two full sweeps disagreed in both directions. Fixed runs
were 56.0 / 49.6 / 53.3, unfixed 48.8 / 49.4 / 46.6: **no overlap between the
groups**. A single sweep could not resolve a 9% effect because its warmup
artifact is worth ~30%.

### Not a general rule

DeepSeek-V4 looks similar (`mtp.0.attn_norm` +0.048) but its *backbone* matches
(+0.068), so that is the family's native distribution. Always compare a head
against its own backbone before shifting anything.

### Shipped

`reap_stream/fix_mtp_norms_qwen35.py`. Writes a new checkpoint (APFS clone,
source untouched), verifies from disk, and checks the rewrite did not zero the
untouched tensors in the same shard. Only
`model-00005-of-00005.safetensors` (0.28 GB) differs.

Both HF repos updated 2026-08-16 with a dated re-download note; both local
models promoted in place, raw shards preserved at
`~/.lmstudio/models/truemod/_mtp_norm_backups/`.

---

## 3. The speculative landscape for this model

oMLX exposes **four separate** options. They are not one feature, and MTP is not
"speculative decoding with a draft model".

| option | what drafts | draft checkpoint | phase |
|---|---|---|---|
| `mtp_enabled` ("Lightning MTP") | the checkpoint's own MTP heads | none — embedded | decode |
| `dflash_enabled` | DFlash block-diffusion draft | DFlash-format | decode |
| `vlm_mtp_enabled` | external assistant drafter | `gemma4_assistant` or `qwen3_5_mtp` | decode |
| `specprefill_enabled` | small LM scores token importance | any small LM | **prefill** |

**They are not all mutually exclusive**, and the exception matters. From
`model_settings.__post_init__`:

- `mtp_enabled` + `dflash_enabled` → rejected (both claim the decode slot)
- `vlm_mtp_enabled` + any other speculative path or TurboQuant → rejected
- **`mtp_enabled` + `specprefill_enabled` → allowed**, and is the production
  config: MTP drives tgTPS, the small draft model drives the ppTPS step above
  8192 tokens.
- `dflash_enabled` + `specprefill_enabled` → **passes validation and then
  silently does nothing.** See below. Config-level permission is not
  implementation.

### DFlash forfeits SpecPrefill — which decides the production config

`omlx/engine/dflash.py` contains **zero references to specprefill**. The feature
is wired only into `engine/batched.py` and `engine/vlm.py`, both of which call
`scheduler.set_specprefill_draft_model(...)`. DFlash runs through its own engine
class, so enabling both leaves prefill dense — visible as flat ppTPS (634–684
across every context) against 2496 on the MTP+SpecPrefill sweeps.

At 16K this dominates everything else:

| | TTFT | tgTPS | **E2E** | throughput |
|---|---|---|---|---|
| MTP + SpecPrefill | 6.6 s | 49.6 | **9.2 s** | 1802.7 |
| DFlash-q (SpecPrefill inert) | 24.4 s | 74.0 | **26.2 s** | 631.0 |

DFlash wins decode by 49% and loses end-to-end by **2.8x**. For 128 output
tokens the decode advantage is worth ~0.9 s while the prefill penalty costs
~17.8 s. Prefill dominates at these lengths, so the decode win is the smaller
term.

**MTP + SpecPrefill is therefore the production config**, not because its decode
is better — it is not, above 8K — but because DFlash currently forces you to
give up the larger win.

This reframes the DFlash result as an **upstream feature gap**: wiring
SpecPrefill scoring into `DFlashEngine` the way `batched.py` and `vlm.py` do it
would plausibly give 74 tgTPS *and* 6.6 s TTFT. That is a well-defined oMLX
contribution.

A methodological note, since this was got wrong twice in one session: reading a
config validator and inferring behaviour is not verification. `__post_init__`
not rejecting a combination says nothing about whether any engine implements it,
exactly as a matching tensor-name set said nothing about whether DSpark's
backbone would draft usefully. Check the execution path.

### Measured: MTP vs DFlash

`modal-labs/Qwen3.5-27B-DFlash` loads in q38 with no new code — mlx_vlm ships
`qwen3_dflash`, and the config matches ours (`hidden_size` 5120, `vocab_size`
248320, `target_layer_ids` max 61 for a 64-layer target). It is distilled for
Qwen3.**5**-27B, not our 3.8.

| config | mean tok/s | drafter size |
|---|---|---|
| MTP (repaired) | **39.4** | 0.28 GB, embedded |
| DFlash bf16, block 16 | 27.7 | 4.26 GB |
| DFlash quantized (8b attn / 4b MLP, gs64) | 38.7 | 1.29 GB |

**Quantizing the drafter is worth +40%** and closes the entire deficit — a
second independent confirmation, after the MTP head, that a bf16 drafter inside
a quantized model pays badly here. `q38_quantize_drafter.py` does it.

At a 66-token prompt DFlash reaches parity, not victory. **That result does not
generalise, and the short prompt was the whole problem.**

### DFlash beats MTP from ~8K up (measured in oMLX)

Full sweeps, `Qwen3.5-27B-DFlash-q` (the quantized 0.78 GB build):

| ctx | TPOT DFlash | TPOT MTP | tgTPS DFlash | tgTPS MTP |
|---|---|---|---|---|
| 1k | 23.0 ms | 23.5 ms | 43.7 | 42.8 |
| 4k | 19.2 | 17.8 | 52.4 | **56.5** |
| 8k | 16.5 | 18.6 | **61.0** | 54.3 |
| 16k | **14.4** | 20.3 | **70.1** | 49.6 |
| 32k | 20.4 | 21.2 | 49.4 | 47.5 |

DFlash gets *faster* with context to 16K while MTP degrades: +41% tgTPS at 16K.

**Caveat — single-sweep rows are unreliable here.** A second sweep of the same
config gave 48.8 / 36.0 / 51.9 / 74.0 / 81.5 / 38.0 (1k…64k). Against run 1 that
is −31% at 4K and **+65% at 32K**: the 32K row that looked like a trend break in
run 1 became the fastest row in run 2. Only the 16K result (70.1, 74.0) and the
general middle-context shape survive both runs. Fixed-context repeats, three per
config, are the protocol that resolved this elsewhere in this document; the
edges (4K, 32K, 64K) remain unresolved without them.

**Mechanism.** As context grows the 27B target's forward dominates, so what
matters is tokens accepted per target forward. DFlash drafts a block of **16**;
MTP drafts **3**. DFlash's six extra layers are cheap next to a 27B forward over
16K tokens, so the amortisation wins. Below ~4K the target forward is cheap and
that overhead is not yet repaid — which is exactly the 66-token result (38.7 vs
45.2) rather than a contradiction of it.

**The comparison is confounded in DFlash's favour to a degree, and against it in
another.** The MTP sweep ran with SpecPrefill on (ppTPS 2496 at 16K), so MTP was
decoding against a cache holding ~20% of tokens — roughly 3.3K at the 16K row —
while DFlash decoded against the full 16.4K and still won. Against that, sparse
context may itself depress MTP's acceptance. The clean test is DFlash+SpecPrefill
against MTP+SpecPrefill, which nothing prevents (see below).

**A prediction recorded here earlier — that MTP's lead would widen with context,
since it is one layer over the KV against DFlash's six — was wrong.** It is kept
in this note rather than deleted: the reasoning was plausible and still wrong,
because it counted drafter cost and ignored draft-block size.

**Open:** the 32K row breaks the trend (70.1 → 49.4 at 27.8 GB peak). Either
acceptance falls past 16K or something else changes. One point at 64K would say
whether DFlash's advantage is a 8–16K window or a general property.

### DSpark: matched, but no MLX runtime

`RadixArk/Qwen3.8-27B-DSpark` targets our exact model, is smaller (1.36B), and
claims 3.39 mean acceptance length against our MTP's 3.05. It is a **DFlash
backbone plus two heads**, both optional at inference:

- `VanillaMarkov` — `markov_w1`/`markov_w2`, an EAGLE-style logit bias from the
  previous token. Drop it and drafts are still valid, acceptance slightly lower.
- `AcceptRatePredictor` — sizes the block dynamically. Drop it, use a fixed size.

Nothing in MLX runs it today: mlx_vlm has zero `confidence_head`/`markov`
support, and oMLX's `DSparkMarkovHead`/`DSparkConfidenceHead` are registered on
`mlx_lm.deepseek_v4` only.

**Distillation is not needed.** RadixArk already distilled against Qwen3.8; the
work is a port, not a training run.

### Measured: the Markov head is load-bearing, not auxiliary

DSpark's tensors are a strict superset of a working DFlash checkpoint — 14
shared families, nothing missing, exactly four extra
(`confidence_head.proj.{weight,bias}`, `markov_head.markov_w{1,2}.weight`). So
`q38_load_dspark.py` filters those four and the backbone loads into
`DFlashDraftModel` with no new modelling code. It loads, it runs, and output is
correct.

It is also useless:

| config | mean tok/s | acceptance |
|---|---|---|
| MTP (repaired) | **45.2** | 3.05 tok/forward |
| DFlash (Qwen3.5-targeted), quantized | 38.7 | — |
| DSpark backbone minus Markov head | 23.4 | **1.18 tok/round** |

1.18 is barely above 1.0, i.e. essentially no speculation, against DSpark's
claimed 3.39. **The Markov head supplies most of the acceptance**, and calling it
optional because it is "just a logit bias" was wrong.

The general lesson: a tensor-name superset proves the weights will *load*. It
says nothing about whether the model will *work*. Check `_run_speculative`'s
`[DFLASH] accept=` line before trusting any drafter.

Two things ruled out along the way, so they are not re-investigated:
`target_layer_ids [4,16,28,40,52]` are DSpark's own and are picked up correctly;
and `projector_type: "dspark"` is handled identically to `None` in the reference
`dflash.py` (only `"domino"` differs), so mlx_vlm ignoring that field is
harmless.

### Markov head ported — helps, does not close the gap

`q38_dspark.py` adds the head to `DFlashDraftModel` (config carry-through,
`markov_w1`/`markov_w2`, bias added between logits and sampler, aux tensors
filtered). The whole checkpoint then loads unmodified.

| config | mean tok/s | acceptance |
|---|---|---|
| MTP (repaired) | **45.2** | 3.05 tok/forward |
| DFlash (Qwen3.5-targeted), quantized | 38.7 | — |
| DSpark backbone only | 23.4 | 1.18 |
| DSpark + Markov head | 23.7 | **1.33** |

Acceptance rose 1.18 → 1.33, so the head is bound and the bias alignment is at
least directionally right (a misaligned bias would have lowered it). But the
head's cost cancels the gain, and 1.33 is still nowhere near the claimed 3.39.

**The remaining gap looks architectural.** mlx_vlm's `draft_block` is
**single-shot**: one forward from `[last_token, MASK, MASK, ...]`, logits for
all positions in parallel. A *bigram* bias conditioned on a MASK token is nearly
meaningless for every position after the first — which is the shape of the
result. Block diffusion as DSpark uses it iterates: draft, refill the block with
the drafted tokens, re-run, so the bias conditions on real tokens. Supporting
that is a much deeper change than adding two tensors, and it would also be where
the confidence head (which sizes blocks per step) starts to matter.

**Recommendation: stop here.** MTP at 45.2 tok/s is well ahead of every drafter
tried, is embedded in the checkpoint, and costs 0.28 GB. The DSpark path needs
iterative block refinement in mlx_vlm before its 3.39 is reachable, and that is
a substantially larger project than the remaining items below.

(Distillation *would* work here, unlike DWQ: the objective is draft-target
agreement, which is literally what acceptance measures, not a proxy for it.)

---

## 4. SpecPrefill: vendored, tuned, defaults corrected

`q38_specprefill.py` is oMLX's implementation copied verbatim except for one
import (its `make_sampler`, which oMLX documents as matching mlx_lm's). Tuned
with `q38_tune_specprefill.py`.

19,475-token prompt of **non-repeating** natural prose, five facts planted at
depths 0.1–0.9, all of which must be retrieved:

| drafter | tokenizer match | keep | TTFT | facts |
|---|---|---|---|---|
| dense | — | 1.00 | 34.2 s | **5/5** |
| Qwen3.5-0.8B | 100% | 0.20 | 11.9 s | 1/5 |
| Qwen3.5-0.8B | 100% | 0.35 | 18.2 s | 4/5 (4,4,4) |
| **Qwen3.5-0.8B** | 100% | **0.50** | **24.6 s** | **4.7/5** (5,4,5) |
| Qwen2.5-0.5B-4bit | 0.19% | 0.20 | 10.7 s | 1/5 |
| Qwen2.5-0.5B-4bit | 0.19% | 0.35 | 16.3 s | 2.0/5 (1,2,3) |
| Qwen2.5-0.5B-4bit | 0.19% | 0.50 | 23.2 s | 3.3/5 (3,3,4) |

**`keep_pct = 0.20` — oMLX's default — loses 4 of 5 facts.** With both drafters.

**The drafter must share the tokenizer.** The matched one improves monotonically
and repeats consistently; the mismatched one returns 1,2,3 on identical settings
and never reached 5/5 at any keep rate including 0.70. The variance is the
diagnostic: SpecPrefill's lookahead is stochastic, so a scorer working on
meaningless embeddings reranks differently every run.

**The honest speedup is ~1.4x**, not 3x. At keep 0.50 it is 34.2 → 24.6 s and it
still drops a fact sometimes; at 0.70 the drafter cost eats the whole gain. The
ppTPS column flatters itself because it counts only the tokens actually
prefilled.

### Drafter choice: Qwen3.5-2B at keep 0.40

Both Qwen3.5-family drafters share the target's vocabulary exactly, so both are
viable. Measured at 16K, five facts, three repeats, `--max-tokens 256`:

| drafter | keep | TTFT | facts (3 runs) | missed |
|---|---|---|---|---|
| 0.8B | 0.30 | 13.2 s | 4, 4, 4 | Osprey |
| 0.8B | 0.40 | 16.5 s | 4, 4, 4 | Osprey |
| 0.8B | 0.50 | 20.4 s | 4, **5**, **5** | Osprey once |
| 2B | 0.30 | 13.5 s | 4, 4, 4 | Kestrel |
| **2B** | **0.40** | **17.7 s** | **5, 5, 5** | — |
| 2B | 0.50 | 22.3 s | 5, 5, 5 | — |

**The 2B is the only clean floor in the table.** The 0.8B never reaches a
reliable 5/5 — even at keep 0.50 it drops a fact one run in three. So the 2B is
both faster (17.7 s vs 20.4 s) and more reliable than the 0.8B's best setting.

**Scoring cost is far smaller than assumed.** Measured over a 16K pass, bf16,
warm, min of 3: 0.8B **0.97 s**, 2B **1.53 s**. The 2B therefore costs +0.56 s
and buys 10 points of `keep_pct`. An earlier estimate that it "needed 15 points
to break even" came from guessing the scoring pass at 3 s and 8 s; the real bar
is ~1.6 points.

**Do not quantize the scorer.** 8-bit bought nothing on the 0.8B (0.98 s vs
0.97 s) and cost 8% on the 2B (1.66 s vs 1.53 s). Prefill is compute-bound, so
there is no weight bandwidth to reclaim and dequant is added work. bf16 also
keeps quantization error away from a component that is **not**
rejection-verified: unlike an MTP or DFlash draft, a bad token selection is
never checked by the target, so there is no acceptance rate to absorb it — the
damage is silent context loss.

Each drafter has a *characteristic* blind spot — the 0.8B loses Osprey
(depth 0.3) every run, the 2B loses Kestrel (depth 0.1) every run. Same fact,
every repeat: these are deterministic rankings with systematic weaknesses, not
scatter, which is itself evidence the scores are meaningful.

**Harness trap: the sampling budget must fit a preamble.** At `--max-tokens 64`
the 2B scored 5/5, 0/5, 0/5 at keep 0.30, which looked like catastrophic
instability. It was truncation: with `enable_thinking=False` the model still
sometimes narrates ("Let me search through the conversation history...") before
answering, and 64 tokens expired mid-narration on a selection that had kept the
facts. Default is now 256. This is the second time in this document that a
generation-budget artifact was nearly read as a model result.

### Tokenizer compatibility

| drafter | vocab | id-match vs Qwen3.8 |
|---|---|---|
| Qwen2.5-0.5B-Instruct | 151,936 | 0.19% |
| Qwen3-0.6B | 151,936 | 0.19% |
| **Qwen3.5-0.8B** | 248,320 | **100.00%** |

Qwen3 kept Qwen2.5's tokenizer; the expansion to 248,320 landed at Qwen3.5. Only
Qwen3.5-family drafters are id-compatible. MLX returns **zero vectors** for
out-of-range ids rather than raising, so a mismatched drafter fails silently.

### How this was got wrong first

The initial test used **one** fact in filler that cycled eight sentences. Every
configuration passed, including keep=0.10 with a 0.19%-match drafter, and the
recommendation was keep=0.2. A lone novel span in repetitive text is findable by
novelty alone, with no understanding of the tokens — the test measured the
filler. Non-repeating prose and five facts reversed the conclusion entirely.

**Not yet wired into the server.** mlx_vlm's `Qwen3_5Attention` takes
precomputed `position_ids`/`position_embeddings` rather than owning a rope
module, so the vendored `sparse_prefill`'s `_PositionMappedRoPE` wrappers have
nothing to wrap. The adaptation is *simpler* than what it replaces: pass the
kept tokens with their original `position_ids` and RoPE is correct by
construction. Note that 3 of 4 Qwen3.5 layers are GatedDeltaNet — a recurrence
with no positional argument — so SpecPrefill is approximate on this architecture
in a second way beyond token selection.

---

## 5. Traps that cost real time

**Benchmarking with another engine resident.** Three times this session. oMLX at
89% CPU made MTP read 12.0 tok/s instead of 39.4, and produced a confident,
entirely false "DFlash is 3–5x slower and smaller blocks are worse" conclusion.
Resident-but-idle (~1% CPU) was fine; actively working was not. **Check `ps` for
`omlx-server` before believing any number.**

**`mlx_vlm.utils.load_model` calls `sanitize` only when the checkpoint is not
already MLX format.** Ours carries `metadata={"format":"mlx"}`, so every
sanitize-based patch is dead code on it — the `mtp.*` → `language_model.mtp.*`
rename had to move into a `load_weights` wrapper.

**Stock mlx_lm's qwen3_5 `sanitize` does not load our checkpoint correctly.** It
produces fluent garbage (top-5 first tokens `['권','价','Cour','_employee','แก้ว']`)
rather than failing. oMLX's patch is required on that path too.

**Absence of `[MTP]` acceptance logs is not evidence MTP is idle.** That logging
lives in `_run_speculative()`, which only handles `dflash`/`eagle3`;
`draft_kind="mtp"` stays in `_run()` and passes the drafter into
`BatchGenerator`.

**`mlx_vlm.generate.ar.BatchGenerator` is a different class from
`mlx_lm.generate.BatchGenerator`.** oMLX's native MTP patch targets the mlx_lm
one, so it cannot reach the VLM path however the head is stored.

**A model directory with `index.json` but no shards is an unfinished
download**, not a corrupt model. Cost a confusing `FileNotFoundError`.

---

## 6. Open work, by value

1. **Wire SpecPrefill into the server** via the `position_ids` approach, default
   keep 0.5 with a tokenizer-matched drafter. This is the top item because it
   **stacks with MTP** rather than replacing it — it is the difference between
   q38 matching the production oMLX setup and having only half of it.
2. **`mtp_num_draft_tokens > 3`.** Never tested; acceptance at depth 3 was 96%,
   so there may be free speed. Config change, no rebuild, and it improves the
   path that already wins.
3. **MTP vs DFlash across context lengths.** Every drafter comparison here used
   a 66-token prompt. The MTP head is one layer over the full KV; DFlash is 6
   layers each holding their own cache, so its per-round cost should grow faster
   — but that is a prediction, not a measurement.
4. **Harder SpecPrefill eval** at other context lengths and on non-retrieval
   work (summarisation may tolerate dropped spans far better than fact lookup).
5. **DSpark iterative block refinement**, if ever. Needs multi-step diffusion in
   mlx_vlm's `draft_block`; the Markov head alone got 1.18 → 1.33 against a
   claimed 3.39. Large project, and it only ever replaces MTP.
