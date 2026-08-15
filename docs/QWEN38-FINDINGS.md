# Qwen3.8-27B — quantization findings

Everything measured while building `Qwen3.8-27B-AWQ-4.85bpw`. Companion to
`DWQ-DISTRIBUTED-FINDINGS.md` (DeepSeek-V4); read §15–17 there for the AWQ
method results that carried over.

Published: https://huggingface.co/True2456/Qwen3.8-27B-AWQ-4.85bpw
Local: `~/.lmstudio/models/truemod/Qwen3.8-27B-AWQ`
Build: `reap_stream/awq_quantize_qwen35.py`, calib `reap_stream/build_qwen_calib.py`

---

## 1. The model

`Qwen/Qwen3.8-27B`, 27.781B params bf16 (55.6 GB), `model_type: qwen3_5`,
architecture `Qwen3_5ForConditionalGeneration`.

- **Dense**, not MoE. 64 layers, hidden 5120, intermediate 17408.
- **Hybrid attention**: `full_attention_interval: 4` → 16 full-attention layers,
  48 **GatedDeltaNet** (linear attention) layers. GDN is 20% of weights, over 3×
  the full-attention weights.
- 1-layer **MTP** head (`mtp_num_hidden_layers: 1`,
  `mtp_use_dedicated_embeddings: False` — shares embed/lm_head).
- 27-layer vision tower, 256K context, mrope, untied embed/lm_head over a
  248,320 vocab.

| family | params | share |
|---|---|---|
| MLP gate/up | 11.409B | 41.1% |
| MLP down_proj | 5.704B | 20.5% |
| GatedDeltaNet in_proj | 4.050B | 14.6% |
| GDN out_proj | 1.510B | 5.4% |
| embed | 1.271B | 4.6% |
| lm_head | 1.271B | 4.6% |
| attention q/k/v | 1.174B | 4.2% |
| attention o_proj | 0.503B | 1.8% |
| vision tower | 0.461B | 1.7% |
| MTP head | 0.425B | 1.5% |

---

## 2. Kernel constraint: there is no 3-bit

Verified at runtime against oMLX's own extension:

```
q2: True   q4: True   q5: True   q6: True   q8: True
q3: False
group_size 32/64/128: all True
```

oMLX's `qwen35_prefill` kernels expose q2/q4/q5/q6/q8 but **not q3**. Any 3-bit
tensor in the MLP silently drops prefill to the slow path. This is the single
biggest constraint on recipe design, and it is why the compression ladder (§7)
has no step between 4-bit and 2-bit.

Note the earlier reading of "kernels not built" was a Python version artifact:
the extension is `cpython-311`, the project venv is 3.14. Test with oMLX's own
interpreter at `/Applications/oMLX.app/Contents/Resources/Python/cpython-3.11`.

---

## 3. The imatrix inverts DeepSeek's recipe

Collected with `reap_stream/collect_imatrix_qwen35.py` (non-streaming: the model
is 55.6 GB and fits; streaming was only ever needed because DeepSeek was 155 GB).
128 samples × 512 tokens, 94 seconds, 497 entries, validated through oQ's own
loader (all finite, non-negative, counts uniform at 45,586).

Participation ratio is the comparable metric — roughly *what fraction of input
channels actually carry the energy*. Raw energy is **not** comparable across
projections, because `gate_proj`/`up_proj` read the hidden state while
`down_proj` reads the post-SwiGLU intermediate: different vector spaces.

| family | top-1% energy | participation ratio |
|---|---|---|
| attention q/k/v | 65.0% | **0.0022** |
| GDN in_proj | 59.4% | 0.0040 |
| MLP gate/up | 25.0% | 0.0396 |
| lm_head | 23.5% | 0.0920 |
| GDN out_proj | 22.6% | 0.1188 |
| attention o_proj | 17.0% | 0.2053 |
| **MLP down_proj** | 13.8% | **0.2986** |

**This is the opposite of DeepSeek-V4**, where `down_proj` is the concentrated
tensor (PR 0.27) and gate/up are flat (PR 0.84). Here `down_proj` is the
flattest tensor in the model and attention q/k/v is **135× more concentrated**.

Transferring DeepSeek's "extra bits on down_proj" would have spent them on the
family needing them least while leaving the sensitive one under-protected. The
94 seconds of measurement was the difference between a designed recipe and a
guessed one.

**Coverage gap:** calibration was text-only, so the vision tower (167 modules)
and MTP (15 tensors, stripped by mlx_vlm at load) have **no imatrix data**.
Their bit assignments are conservative guesses, not measurements.

---

## 4. Final recipe — 16.83 GB, BPW 4.85

| component | bits | basis |
|---|---|---|
| MLP gate/up | 4-bit gs128 | AWQ-calibrated |
| MLP down_proj | 4-bit gs128 | AWQ-calibrated; flattest, no premium |
| GDN in_proj | 5-bit gs64 | 2nd most concentrated |
| GDN out_proj | 4-bit gs64 | middling |
| attention q/k/v | 8-bit gs64 | most concentrated by 135×, only 4% of weights |
| attention o_proj | 4-bit gs64 | flat |
| embed | 4-bit gs128 | lookup, not a matmul |
| lm_head | 6-bit gs128 | Step-3.7 head8 precedent |
| vision tower | 8-bit gs128 | **not measured** |
| MTP head | 8/6/4-bit | see §6 |
| conv1d, norms, A_log | bf16 | negligible size, high sensitivity |

27 vision `linear_fc2` modules stay bf16: in_dim 4304 is not divisible by 128,
64 or 32.

**AWQ is sequential**: layer *i* is calibrated on activations from the
already-quantized layers above it. The attention/GDN half of each block is
computed once and reused after quantization (quantizing the MLP cannot change
it), so the pass is ~64 layer-forwards rather than the ~2080 a naive
re-forward-per-layer would cost. 84 minutes for the full build.

Calibration: 192 prompts × 1024 tokens from `calib/cloud_reap_8k.jsonl`
(coding/tool-use/agentic), re-rendered through Qwen's own chat template with a
50/50 think/nothink split, 21.0% padding. `reap_stream/build_qwen_calib.py`
converts the plain `SYSTEM:/USER:/ASSISTANT:` markers into messages and applies
the template — all 7200 rows converted, zero failures.

---

## 5. Results

### Quality — essentially free

| benchmark | AWQ 4.85bpw | bf16 | delta |
|---|---|---|---|
| HumanEval | 93.3% (153/164) | 93.9% (154/164) | −1 |
| GSM8K | 92.0% (184/200) | 92.5% (185/200) | −1 |
| MMLU | 83.0% (166/200) | 84.0% (168/200) | −2 |

**4 questions out of 564.** Contrast DeepSeek-V4 at 2.91 BPW: −17 MMLU points,
p=3.06e-06. Two differences explain it — Qwen starts from **bf16** rather than
an already-mxfp4 checkpoint, and 4.85 BPW is a far gentler target than 2.91.

### Speed (M5 Max, MTP on, prefill kernel bypassed — see §7)

| ctx | TTFT | TPOT | ppTPS | tgTPS | peak mem |
|---|---|---|---|---|---|
| 1k | 1163 ms | 15.4 ms | 880 | 65.5 | 17.0 GB |
| 8k | 10001 ms | 16.9 ms | 819 | 59.7 | 19.4 GB |
| 16k | 21411 ms | 17.1 ms | 765 | 59.0 | 21.3 GB |
| 64k | 119769 ms | 24.2 ms | 547 | 41.7 | 32.6 GB |

vs bf16: **3.9× generation** (55–65 vs 14 tgTPS), prefill at parity, a third of
the memory (32.6 vs 68.5 GB at 64k).

---

## 6. MTP: the head must be quantized

Native `mtp_enabled` supports `qwen3_5` — no external drafter and no missing
model class. (The `vlm_mtp_enabled` path *does* need a `qwen3_5_mtp` model class
that does not exist in mlx_vlm; that is a different, unused path.)

**Carried out of the bf16 original the head is 8 bare `Linear` inside a
496-`QuantizedLinear` model, and MTP then *halves* throughput.** After
quantizing (8-bit attn / 6-bit `fc` / 4-bit MLP):

```
tokens=128 cycles=42 tok/cycle=3.05 accept=86/97 (88.7%)
depth[d1=35/39, d2=28/34, d3=23/24]
timing[backbone=1813.6ms mtp=19.4ms]     <- drafter is 1.1% of backbone
```

**1.8–2.1× generation.** Safe to quantize: drafts are rejection-verified, so
drafter error costs accept-rate, never correctness. d3 acceptance was 96%, so
`mtp_num_draft_tokens` > 3 is worth trying.

Three traps, all of which produce plausible-looking wrong results:

1. **mlx_vlm strips `mtp.*` at load.** A build saved from a loaded model loses
   the head entirely (15 tensors, 0.849 GB). `reap_stream/carry_mtp_weights.py`.
2. **config.json quant keys must be `language_model.mtp.*`, not `mtp.*`.** oMLX
   remaps the weights and its class_predicate looks up the remapped name; a miss
   inits at the default 4-bit and shape-errors on the 6-bit `fc`.
3. **`mx.load` returns arrays still backed by the file.** `mx.save_safetensors`
   to that same path zeroed every untouched tensor — all 7 MTP norms → 0.0000 —
   while the model still loaded and answered correctly, because norms only
   affect the draft head. `mx.eval` first, then write to a temp file and rename.

**Norm conventions are mixed inside the head itself.** In Qwen's own checkpoint
`mtp.norm` is already MLX-shifted (mean +1.2520) while the per-layer norms are
raw-HF (+0.0361, +0.7906). oMLX decides per key by magnitude. Carry the values
verbatim; do not apply a blanket shift.

---

## 7. oMLX prefill kernel: found, reported, fixed upstream

`omlx/patches/qwen35_q4_mlp.py` routes 4-bit MLP matmuls to a native kernel
above `OMLX_QWEN35_Q4_MLP_MIN_TOKENS` (default 2048). That kernel's speed comes
from the **NAX tensor-unit path, which the extension gates on group_size == 64**
(`csrc/qwen35_prefill.cpp`). At gs128 it demotes to plain Metal, which is
**3.6–3.7× slower than MLX's own qmm**.

Controlled, same binary and model, env var the only difference:

| routing | TTFT | ppTPS |
|---|---|---|
| default (kernel on) | 7982 ms | 513.2 |
| kernel disabled | 4429 ms | 924.8 |

Output is numerically correct either way (rel err 5e-5); purely throughput.
All nine variants at gs128 land between 0.21× and 0.26×, so it is not a variant
choice. Re-encoding gs128 metadata to gs64 (`mx.repeat(scales, 2, axis=1)`)
recovers full speed with **bit-identical** output, proving the diagnosis — but
only reaches parity, at the cost of doubled scale storage, so the fix skips
routing instead.

**Reported as [jundot/omlx#2657](https://github.com/jundot/omlx/pull/2657).**
Until it merges, launch via `~/bin/omlx-fast` or set
`OMLX_QWEN35_Q4_MLP_MIN_TOKENS=999999999`. Repro:
`artifacts/omlx_patches/qwen35_qmm_repro.py`.

Two false starts worth not repeating: 5-bit is **not** pathologically slow
(1.28× faster than bf16), and an early "the update already fixed it" reading was
wrong because the app had not actually been relaunched from the Dock — verify
the process parent (`ps -o ppid=`) before trusting an env-var A/B.

---

## 8. The checkpoint is oMLX-only

Keeping `mtp.*` in the checkpoint means stock `mlx_vlm` sees those keys, flips
`should_shift_norm_weights`, and applies `+1.0` to layernorms that already have
it. Verified: layer-0 norm goes 0.9648 → 1.9609 and output becomes fluent
garbage (`'ombたくiboldimmereltsivec...'`). It **loads and generates**, which is
the dangerous part.

oMLX patches `sanitize` to gate the shift on `conv1d` layout instead, which is
correct for a converted checkpoint. Do not load this build outside oMLX. To make
it portable, move the 15 `mtp.*` tensors to a subdirectory sidecar and drop them
from `model.safetensors.index.json` (loader is index-driven; its glob fallback
is non-recursive) — you lose MTP, you gain portability.

---

## 9. How much smaller can it go — untested ladder

Effective bits include group overhead (+0.25 at gs128, +0.50 at gs64).
**There is no 3-bit**, so every cut jumps to 2-bit.

| variant | change | size | BPW |
|---|---|---|---|
| current | — | 16.74 GB | 4.82 |
| A | `down_proj` → 2-bit | 15.31 GB | 4.41 |
| B | A + `gate/up` → 2-bit | 12.46 GB | 3.59 |
| C | B + GDN in_proj 5→4 | 11.95 GB | 3.44 |
| D | C + embed→2, lm_head→4 | 11.32 GB | 3.26 |
| E | D + attn qkv→6, o_proj→2 | 10.90 GB | 3.14 |
| floor | everything 2-bit gs128 | 7.81 GB | 2.25 |

**A is the principled first cut and a real test of the imatrix**: `down_proj` is
the flattest tensor, so the measurement predicts it tolerates 2-bit best. If it
craters, concentration does not predict bit tolerance and that is worth knowing.

**B is where the compression is** (MLP = 61.6% of weights) and where trouble is
expected: this is a **dense** model, with none of the expert redundancy that let
DeepSeek survive 2.91 BPW. Cutting the other way, Qwen starts from bf16. Those
push opposite directions; genuinely unknown.

Two cautions: at 2-bit, AWQ calibration becomes essential rather than helpful
(uncalibrated 2-bit measured 28.5% MMLU — chance is 25% — with gibberish
output), and 2-bit gs128 measured **slower than bf16** (0.65×) in the kernel
benchmark, so the smallest build may not be the fastest.

Cheapest next step is the ablation harness (`test_awq_stage_ablation.py`
pattern) measuring real-token reconstruction error at 2-bit vs 4-bit per family
**at depth** — minutes rather than ~85 min per build. It will not predict
benchmark scores (proxies over-credited repeatedly, see DWQ findings §17) but it
will show which families fall off a cliff.
