---
license: other
license_name: deepseek
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
tags:
  - mlx
  - omlx
  - awq
  - moe
  - quantized
  - deepseek
pipeline_tag: text-generation
---

# DeepSeek-V4-Flash-0731 — AWQ 2/3-bit (MLX / oMLX)

AWQ-calibrated mixed-precision quantization of `DeepSeek-V4-Flash-0731`
(304B MoE, 43 layers, 256 routed experts/layer, Hyper-Connections, HISA
attention) for Apple Silicon via **oMLX**.

Routed experts carry the aggressive quantization; everything else stays at
8-bit.

| component | precision |
|---|---|
| `switch_mlp.gate_proj` / `up_proj` | 2-bit, group size 128 |
| `switch_mlp.down_proj` | 3-bit, group size 64 |
| attention, shared experts, embeddings, head | 8-bit, group size 64 |
| MTP / DSpark drafter (`-mtpq`) | 2-bit gs128 / 3-bit gs64 — same recipe |

**Backbone: 278.6B quantized params · 101.4 GB · BPW 2.91**
**oMLX build with quantized MTP drafter: 108.2 GB total** (drafter 7.04 GB,
down from 10.86 GB unquantized)

The MTP/DSpark speculative drafter is **quantized to the same 2/3-bit recipe**,
not left at native precision. BPW above is quoted for the backbone weights;
the drafter is additional and is not part of the 278.6B count.

---

## Recommended sampling

```
temperature = 1.0
top_p       = 0.95
min_p       = 0.05
```

For long-context work, enable **KV cache quantization at 4-bit or 6-bit** —
this model's context cost is dominated by KV, and 6-bit is essentially free
quality-wise while 4-bit buys substantially more usable context.

`reasoning_effort` supports `low`, `high`, `max` (not `medium`/`xhigh` — the
chat template rejects those). Pass via `chat_template_kwargs`.

---

## Performance (M5 Max, 128 GB, oMLX, MTP enabled)

### Single request

| test | TTFT (ms) | TPOT (ms) | prefill tok/s | gen tok/s | E2E (s) | peak mem |
|---|---|---|---|---|---|---|
| pp 1024 / tg 128 | 1,650.8 | 19.9 | 620.3 | 50.8 | 4.2 | 102.4 GB |
| pp 4096 / tg 128 | 5,927.1 | 20.3 | 691.1 | 49.7 | 8.5 | 103.4 GB |
| pp 8192 / tg 128 | 12,823.2 | 21.7 | 638.8 | 46.4 | 15.6 | 104.5 GB |
| pp 16384 / tg 128 | 30,535.1 | 21.4 | 536.6 | 47.0 | 33.3 | 106.7 GB |

**Generation speed is essentially flat with context** — TPOT stays at ~20–22 ms
from 1k to 16k, i.e. ~47–51 tok/s throughout. The MTP/DSpark drafter is doing
its job.

### Batched

| batch | gen tok/s | prefill tok/s | avg TTFT (ms) | E2E (s) | speedup |
|---|---|---|---|---|---|
| 1× | 50.8 | 620.3 | 1,650.8 | 4.2 | 1.00× |
| 2× | 36.9 | 466.5 | 4,389.7 | 11.3 | 0.73× |
| 4× | 54.6 | 467.4 | 8,620.2 | 18.1 | 1.07× |
| 8× | 75.2 | 470.5 | 16,935.1 | 31.0 | 1.48× |

### Two things worth knowing

**Build the custom kernels for long context.** If oMLX reports
`dsa_indexer_scores/dsa_topk_indices unavailable (glm_moe_dsa extension not
built)`, the sparse-attention indexer falls back to a slow MLX path and
long-context *prefill* is several times slower than it needs to be. Rebuild
with `OMLX_WITH_CUSTOM_KERNEL=1`. Generation speed is unaffected — this is
purely a TTFT issue, and it is the largest single speedup available.

**Memory and context.** KV growth is modest (~4 GB from 1k to 16k), but the
weights alone occupy ~102 GB, so headroom runs out well before very long
contexts on a 128 GB machine. Use **4-bit or 6-bit KV cache quantization** for
long-context work — 6-bit is effectively free, 4-bit buys substantially more
usable context.

---

## Measured results

Benchmarks via oMLX's own `omlx.eval` harness (n=200 MMLU/GSM8K, n=164
HumanEval), MTP enabled:

| Benchmark | v1 (single-turn calib) | **this build (v2)** |
|---|---|---|
| MMLU | 34.5% | **61.5%** |
| GSM8K | 77.0% | **93.5%** |
| HumanEval | 72.6% | **87.2%** |

The v2 recalibration used a broader, multi-turn agentic calibration set plus
`down_proj` at group size 64. Both changes were necessary — the v1 calibration
was single-turn only and left the model badly mismatched on real conversation
structure.

### How much does quantization cost?

Measured against the **unquantized teacher** on identical questions
(likelihood-scored MMLU, n=200, paired):

| | MMLU |
|---|---|
| teacher (native) | **81.5%** |
| this build | 64.5% |
| gap | **−17.0pp** (95% CI [10.1, 23.9]) |

Paired: 42 questions the teacher answers and this build misses, versus 8 the
other way. McNemar **p = 3.06e-06**.

This is an honest statement of what 2/3-bit costs on knowledge recall. It is
*not* a reason to avoid the model — it is a 304B model running on a single
Apple Silicon machine at 2.91 BPW, and GSM8K (93.5%) and HumanEval (87.2%)
remain close to ceiling. Knowledge-recall breadth is where the compression is
paid for.

---

## How this recipe compares (all measured, same 200 questions)

Every number below is likelihood-scored MMLU on the *identical* stratified
sample, scored by the same code, paired question-by-question. Absolute values
are not interchangeable with the generation-scored table above; the
*comparisons* are what matter.

| build | MMLU | size | vs this build (paired) |
|---|---|---|---|
| teacher (native mxfp4/mxfp8) | **81.5%** | 155.0 GB | +42/−8, p=3.06e-06 |
| **this build (AWQ 2/3-bit)** | **64.5%** | **108.2 GB** | — |
| + DWQ distillation | 66.5% | 108.2 GB | +24/−20, **p=0.65** |
| same recipe, **no calibration** | 28.5% | 92 GB | emits gibberish |

### 1. Quantization costs ~17 MMLU points, and that gap is real

Against the native teacher on identical questions: 42 questions the teacher
answers that this build misses, versus 8 the other way. McNemar **p=3.06e-06**.
This is an honest statement of what 2/3-bit costs on knowledge recall — and it
is unclaimed headroom, not something any method below recovered.

### 2. AWQ calibration is load-bearing, not a refinement

A plain round-to-nearest build at the *same* bits and group sizes scores 28.5%
— barely above MMLU's 25% chance floor — and generates literal gibberish
(`'}<?iger}<?codeline...'`). Weight reconstruction error was verified at 44.5%
RMS, exactly what honest 2-bit RTN produces, so this is not a broken build: at
2-bit, without activation-aware scaling, the model simply collapses.
**Calibration is the difference between a working model and noise.**

### 3. DWQ distillation added nothing

Distilling the quantized student toward the native teacher (100 steps,
expert scales only, agentic-weighted calibration) cut held-out KL by **~89–91%**
and moved MMLU not at all (p=0.65; near-symmetric 24 gained / 20 lost — churn,
not learning). Two loss variants came out statistically identical (p=0.75).

A cautionary result worth repeating: **KL-to-teacher over-credits badly.** A 91%
improvement in the distillation objective bought zero measurable capability.
Judge distillation on exact-answer or behavioural evals, never on the loss.


### Method note

Teacher and low-bit builds were measured with a streaming likelihood scorer
(`reap_stream/mmlu_streamed.py`) that reads one block at a time — ~8 GB
resident — so a 155 GB model can be benchmarked on a 128 GB machine. Prompting
is oMLX's own 5-shot MMLU format and stratified sample.

---

## Notes

- **oMLX-specific.** Uses oMLX's custom `deepseek_v4` loader; not portable to
  generic MLX/GGUF runtimes as-is.
- **MTP/DSpark drafter is quantized** to the same 2-bit gs128 / 3-bit gs64
  recipe (7.04 GB, saving 3.82 GB over native). Safe to quantize aggressively:
  DSpark uses exact rejection sampling, so every drafted token is verified by
  the main model — drafter error costs accept-rate (speed) only, never output
  correctness.
- **Native tool calling** is DSML (`｜DSML｜` markup), not the generic
  `<tool_call>` convention.

## Sibling model

- [`DeepSeek-V4-Flash-0731-DWQ`](https://huggingface.co/True2456/DeepSeek-V4-Flash-0731-DWQ)
  — distillation-trained variant of this build. See its card before using it:
  it shows no measured capability gain over this model and has a suspected
  early-turn-termination issue.
