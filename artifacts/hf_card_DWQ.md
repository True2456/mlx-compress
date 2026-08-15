---
license: other
license_name: deepseek
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
tags:
  - mlx
  - omlx
  - dwq
  - awq
  - moe
  - quantized
  - distillation
  - deepseek
pipeline_tag: text-generation
---

# DeepSeek-V4-Flash-0731 — DWQ (distillation-trained 2/3-bit, MLX / oMLX)

DWQ (distillation-aware weight quantization) applied on top of the
[AWQ 2/3-bit build](https://huggingface.co/True2456/DeepSeek-V4-Flash-0731-AWQ):
the quantized student's expert **scale factors** were trained to match the
native teacher's output distribution, distilled from `DeepSeek-V4-Flash-0731`
across two Apple Silicon machines over Thunderbolt.

Quantization recipe is unchanged from the AWQ build — DWQ only adjusts scale
values (129 tensors, all `switch_mlp`), so size and BPW are identical.

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

## ⚠️ Read this before choosing this model

**This build may end turns early.** At agentic turn-start positions its
probability of emitting `<｜end▁of▁sentence｜>` is measurably elevated versus
the AWQ build:

| | P(EOS) at turn-start |
|---|---|
| teacher (native) | ~0.0006 |
| AWQ build | 0.0004 |
| **this build** | **0.0795** |

Under greedy decoding this caused 1 of 8 held-out agentic probes to generate
nothing, and shortened outputs on the rest. Under the recommended sampling
settings it is far less severe, and in practice it may not surface at all —
but the elevation is real and measured.

**If you see turns ending prematurely — an empty assistant response, or a stop
right after announcing an intended action — use the
[AWQ build](https://huggingface.co/True2456/DeepSeek-V4-Flash-0731-AWQ)
instead.** It has no such elevation and no measured capability disadvantage.

---

## Recommended sampling

```
temperature = 1.0
top_p       = 0.95
min_p       = 0.05
```

`min_p = 0.05` matters more here than on the AWQ build — it prunes the tail
and keeps generation stable.

For long-context work, enable **KV cache quantization at 4-bit or 6-bit**.

---

## Measured results — honest summary

**DWQ produced no measurable capability gain over the AWQ build.**

MMLU, likelihood-scored, n=200, paired on identical questions:

| build | MMLU | size | vs AWQ (paired) |
|---|---|---|---|
| teacher (native mxfp4) | **81.5%** | 155.0 GB | +42/−8, p=3.06e-06 |
| AWQ build | 64.5% | 108.2 GB | — |
| **this build (DWQ)** | 66.5% | 108.2 GB | +24/−20, **p=0.65** |
| AWQ recipe, no calibration | 28.5% | 92 GB | gibberish output |

For context on what *does* matter: stripping AWQ calibration from the same
recipe drops the model to 28.5% — barely above MMLU's 25% chance floor, with
literally incoherent output. Calibration is load-bearing; distillation on top
of it is not.

The +2.0pp is not statistically distinguishable from zero — the near-symmetric
gained/lost split shows the model shuffling answers rather than improving. The
teacher remains ~30 questions ahead.

### What DWQ did and didn't do

- **Training objective improved enormously**: held-out KL fell **~89–91%**
  across every configuration tried.
- **Capability did not follow**: MMLU p = 0.65.
- Two independent loss variants (EOS-penalty weights 1.0 and 5.0) came out
  statistically identical (p = 0.75).

This is a clean demonstration that **KL-to-teacher over-credits**: a 91%
reduction in the distillation objective bought nothing measurable on
exact-answer accuracy. Judge distillation runs on behavioural or exact-answer
evaluations, never on the training loss.

### Why the early-termination happens

Diagnosed as **decision-critical support omission** (cf.
[arXiv 2607.07050](https://arxiv.org/html/2607.07050)): top-k distillation
renormalizes over only the teacher's top-k tokens, so every other token in the
~129k vocabulary receives *zero gradient* and can drift freely. `EOS` is rare
in teacher targets (top-1 in 0.188% of positions) and fell outside the top-k at
~30% of positions when k=128.

Raising k to 1024 and forcing `EOS` into the retained support at every position
reduced silent probes from 5/8 to 1/8 — confirming the mechanism — but did not
fully close the gap. It is **not** a data problem: the teacher never wants EOS
at these positions (top-1 in 0/62 sampled turn-starts).

---

## Performance

Identical to the [AWQ build](https://huggingface.co/True2456/DeepSeek-V4-Flash-0731-AWQ)
— DWQ changes scale *values* only, not the recipe, tensor shapes, or size. See
that card for measured throughput, latency and memory, and for the
`OMLX_WITH_CUSTOM_KERNEL=1` note on long-context prefill.

---

## Notes

- **oMLX-specific.** Uses oMLX's custom `deepseek_v4` loader.
- **MTP/DSpark drafter is quantized** to the same 2-bit gs128 / 3-bit gs64
  recipe (7.04 GB, −3.82 GB vs native). DSpark uses exact rejection sampling,
  so drafter quantization costs accept-rate only, never correctness.
- Trained 100 steps, scales-only, on 2458 agentic-weighted calibration
  trajectories with teacher targets at k=1024, sequence length 1024.

## Recommendation

For general use, prefer the
[AWQ build](https://huggingface.co/True2456/DeepSeek-V4-Flash-0731-AWQ).
This model is published for reproducibility and for anyone wanting to build on
the DWQ work — not because it is measurably better.
