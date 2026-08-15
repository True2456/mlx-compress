---
name: mlx-compress
description: >-
  Shrinking large language models on Apple Silicon end to end: analyse (importance
  matrices, saliency), prune (REAP), merge (REAM), quantize (AWQ/DWQ, bit
  allocation per tensor family), and evaluate — including streaming everything for
  models larger than unified memory. Use when quantizing or pruning an MLX model,
  choosing bits or group sizes, designing a calibration set, judging whether a
  compressed build is actually good, or when the user mentions AWQ, DWQ, imatrix,
  oQ/oQe, REAP, REAM, expert pruning, group_size, BPW, MTP drafters, or a model
  that does not fit in RAM.
---

# Compressing models on Apple Silicon

Hard-won from shipping 2.9–4.9 BPW builds of DeepSeek-V4-Flash (304B MoE),
Qwen3.8-27B (dense hybrid), Step-3.7-Flash, Ling-3.0 and Laguna on 64–128 GB
Macs. Every number here is measured, not inferred.

Tooling: [mlx-compress](https://github.com/True2456/mlx-compress),
[mlx-reap-streaming](https://github.com/True2456/mlx-reap-streaming).

---

## The five rules that matter most

**1. Benchmark, or you know nothing.** Distributional and reconstruction proxies
over-credit, consistently and badly. Distilling a quantized student toward its
teacher cut held-out KL by 89–91% and moved MMLU **not at all** (p=0.65, 24
questions gained / 20 lost). Judge on exact-answer or behavioural evals. This is
the single most expensive lesson here.

**2. Bit allocation does not transfer between architectures.** On DeepSeek-V4,
`down_proj` is the outlier-concentrated tensor and earns extra bits. On
Qwen3.8, `down_proj` is the **flattest** tensor in the model and attention
q/k/v is 135× more concentrated. Copying a recipe across spends bits on the
family that needs them least. Measure with an imatrix first: 94 seconds.

**3. Calibration data quality dominates method choice.** Switching from
single-turn text to multi-turn agentic conversations, rendered through the
model's own chat template, moved MMLU 34.5% → 61.5%. No amount of method
tuning came close to that.

**4. Failures here are silent, not loud.** Wrong models load and emit fluent
text. Assert bit-exactness wherever you can, and be suspicious of any change
that "worked first time". See the trap list below.

**5. Confirm per-layer findings at depth.** An apparent defect at layer 1
reversed completely at layer 40, where absolute errors are ~1000× larger.
Weight by absolute error, never percentage.

---

## What to reach for

| Goal | Do this | Not this |
|---|---|---|
| Shrink a model | AWQ with real calibration | RTN (2-bit RTN = 28.5% MMLU, chance is 25%, gibberish output) |
| Choose bits per tensor | imatrix → participation ratio | copying another model's recipe |
| Recover quantization damage | nothing that is known to work | DWQ — 89% KL cut, zero benchmark gain, plus an EOS regression |
| Improve AWQ with an imatrix | skip it | AWQ already computes the same statistic and then measures **real output error**, which strictly beats a proxy weight |
| Shrink a MoE further | REAP (prune experts) | REAM (merge) by default — it helps some checkpoints and hurts others, and perplexity cannot tell you which |
| Model bigger than RAM | stream block-by-block | trying to fit it; there is no host/device split on a Mac |

---

## Choosing bits: use participation ratio

An imatrix is a running sum of squared input activations per channel. It has no
global state, so it streams (155 GB model calibrated in 6.3 GB).

**Raw activation energy is not comparable across projections** — `gate_proj`
reads the hidden state, `down_proj` reads the post-SwiGLU intermediate.
Different vector spaces. Use **participation ratio**, which is scale-free and
answers "what fraction of input channels actually carry the energy":

```
PR = (Σx²)² / (n · Σ(x²)²)      # low PR = concentrated = wants more bits
```

Measured on Qwen3.8-27B:

| family | PR | assigned |
|---|---|---|
| attention q/k/v | 0.0022 | 8-bit (only 4% of weights, cheap insurance) |
| GDN `in_proj` | 0.0040 | 5-bit |
| MLP gate/up | 0.0396 | 4-bit |
| MLP `down_proj` | 0.2986 | 4-bit, no premium |

A useful sanity check: AWQ's own scale spread should track PR. It did — 1.88×
for gate/up, 6.66× for `down_proj` on DeepSeek, matching their concentration.

---

## Calibration sets

Render through the model's **own chat template**, not raw text. Structural
tokens (`<|im_start|>`, pre-opened `<think>`, tool blocks) are what the model
sees at inference; calibrating without them is off-distribution.

- Match the deployment mix. For a VLM, include images — text-only calibration
  never runs the vision tower, leaving its bits a guess.
- Watch the **ratio in the prefix actually consumed**, not the file. A
  round-robin interleave over sources once produced 80% images in the first 352
  rows when the target was 30%.
- Split think/nothink 50/50 if the model has both modes.
- Padding is fine (~20–30%) and masking it out measured **worse**. But track
  real tokens: shorter multimodal rows once halved the real-token count at a
  fixed grid.
- Never calibrate on your eval set. Check for `prefix`/`answer` fields.

---

## Pruning and merging MoE experts

REAP scores each expert by router-weighted activation norm and drops the lowest.
Same streaming loop as everything else here: run one block, score it, free it.

- **Cache eviction is the load-bearing part**, not the scoring. `mx.eval()` and
  `mx.clear_cache()` inside the per-prompt loop are what keep peak memory flat.
- Adapter surface is five hooks: resolve the text model, list MoE layer indices,
  get expert count, embed tokens, install a probe, run one layer, free one layer.
- Checkpoint per layer. These runs are long and Thunderbolt links drop.

**REAM merges instead of deleting** — folding each pruned expert into its most
similar kept one. Do not reach for it by default. Measured on Step-3.7: PPL
improved by 0.194 while accuracy stayed flat (24/24 prune vs 23/24 REAM). The
PPL gain was smoothing, not capability. This is the cleanest demonstration in
this whole body of work that **perplexity over-credits** — if you evaluate
REAM on PPL you will ship it, and you will be wrong.

Prune first, then quantize. Pruning changes the activation distribution, so an
imatrix or AWQ calibration collected before pruning describes a model that no
longer exists.

## Traps that produce plausible, wrong models

**`mx.load` returns file-backed arrays.** `mx.save_safetensors` to that same
path truncates the file before the lazy reads happen; untouched tensors come
back as zeros. Measured: all 7 MTP norms → 0.0000, while the model still loaded
and answered correctly, because those norms only affect a draft head.
`mx.eval` first, then write to a temp file and rename.

**`vars()` finds nothing on an MLX Module** (it is a dict subclass). A
hand-rolled traversal reported "0 quantized, 0 skipped" instead of failing. Use
`nn.quantize(model, class_predicate=...)` or `module.children()`.

**Recipe keys must match module paths, not checkpoint tensor names.** Keying on
`visual` when the module is `vision_tower` silently left 111 modules at bf16.

**Quantization config keys must match what the loader looks up.** For MTP heads
under a VLM wrapper that is `language_model.mtp.*`, not `mtp.*`; a miss inits at
the default bit width and shape-errors on anything else.

**Norm-convention double-shift.** Some loaders add `+1.0` to layernorms when
they detect a raw-HF checkpoint. Re-saving an already-converted model can
trigger it twice: norms go 0.96 → 1.96 and output becomes fluent garbage. Check
a layer-0 norm mean after any round-trip.

**MTP/drafter heads get stripped at load** by some loaders, so a build saved
from a loaded model loses them silently. And a bf16 drafter inside a quantized
model can *halve* throughput by falling outside the quantized verify path —
quantize the drafter to match. Safe to do: drafts are rejection-verified, so
drafter error costs accept-rate, never correctness.

**Hand-written forward passes.** If you split a block to insert calibration,
assert bit-exactness against the model's own block before quantizing anything.
This caught a case where `cache=None` was not a valid calling convention at all.

---

## Evaluating a build

Minimum bar before claiming anything:

1. **Load and generate.** Catches gibberish, not much else.
2. **MMLU / GSM8K / HumanEval** against the *unquantized parent*, same
   questions. Paired McNemar is far more sensitive than independent proportions.
3. **Behavioural probes** if the model is agentic — repetition/looping and
   premature end-of-turn. A build can score fine on benchmarks and still stop
   mid-task; measure P(EOS) at turn starts.
4. **Speed and memory** at realistic context lengths, not just 1k.

Expect roughly: bf16 → ~4.8 BPW costs almost nothing (4 questions of 564 on a
dense 27B). Requantizing an already-4-bit checkpoint to ~2.9 BPW costs ~17 MMLU
points. The starting precision matters as much as the target.

---

## Models larger than unified memory

There is no host/device split on a Mac, so "keep it in RAM and stream to the
GPU" does not exist. Everything must work a block at a time:

```
embed all prompts → for each layer: run it, free it, clear cache → repeat
```

`mx.eval()` inside the loop is load-bearing; without it the graph accumulates
lazily and peak memory grows anyway. Also purge any probe registry that holds
references to freed layers, or the stream silently stops streaming (measured
3.4 GB → 98.3 GB over 43 layers).

This works for saliency collection, imatrix collection, and likelihood-scored
evaluation. A 155 GB model can be benchmarked on a 128 GB machine this way.

---

## Relationship to the other MLX skills

`mlx` and `swift-mlx-lm` are vendored: they cover running, serving and
fine-tuning, and their advice is API-level. This skill covers what happens when
you actually compress a frontier model and measure it. Where they disagree,
this one is measured on 27B–304B models.

One disagreement matters. The `mlx` skill's `reference/quantization.md`
describes DWQ as using "knowledge distillation to preserve quality", with no
caveat. Measured on DeepSeek-V4: held-out KL fell 89–91% and MMLU did not move
(p=0.65), while P(EOS) at agentic turn starts rose 0.0004 → 0.0795. The CLI
works exactly as documented; what is undocumented is that the objective it
optimises does not track capability. Treat `mlx_lm.dwq` as unproven, not as a
quality-preserving default.

Use `mlx` for conversion, serving and LoRA. Use this for deciding what to
compress, by how much, and whether the result is any good.

See `reference.md` for architecture-specific recipes, kernel constraints,
distributed setup, the REAP adapter surface, and the full negative-result list.
