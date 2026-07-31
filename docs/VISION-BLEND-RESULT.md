# Vision-Blended Reap Plan: Built and Validated

**Date:** 2026-07-23. Follow-up to `FINDINGS.md` §2 (vision blindness) and
`PPL-DECOMPOSITION.md`. Closes the vision question with the first
image-conditioned measurement.

## The plan

`FINDINGS.md` §2 proposed two fixes. Both were tested against the saved
saliency artifacts before building anything:

1. **Union top-N vision experts into the keep set — measured nearly useless.**
   Protecting top-32/layer moves discarded vision mass only 9.42% → 8.84%.
   The lost mass is spread across mid-ranked experts (the same flat-saliency
   property that makes this model resist pruning), not concentrated at the top.
2. **Explicit score combine — works.** Mass-normalized blend per layer,
   `S = (1−β)·t̂ + β·v̂`, top-245. Frontier (β = 0.1–0.5) has its knee at
   **β = 0.3**: vision mass discarded **9.42% → 6.51%** (worst layer
   13.40% → 8.59%) for text discarded 5.04% → 5.46% (≈ +0.001 NLL by the
   measured scaling — noise). The z-scored Contrastive Skill-Shield at α = 0.5
   lands on the same frontier point; the mass blend is simpler.

Plan: `artifacts/plans/plan_p15_blend03.json` (42 layers, uniform 245,
95.3% keep-overlap with text-only p15).
Model: `models/Step-3.7-p15-4bit-vblend` — 92 GB, 4.632 bpw, same profile as
the original student. `models/Step-3.7-full-4bit` was deleted (with approval)
to make disk room; reproducible from `plan_p00_full.json`.

## Text regression check — passes

Same 500 held-out prompts / evaluator / settings as `ppl-p15-4bit-500.json`:

| category | BF16 | p15-old | p15-vblend | ΔNLL vs old |
|---|---|---|---|---|
| agentic | 6.80 | 7.37 | 7.08 | −0.040 |
| coding | 8.02 | 7.54 | 7.43 | −0.015 |
| general_instruction | 6.18 | 6.00 | 6.25 | +0.040 |
| reasoning_math | 2.47 | 2.51 | 2.52 | +0.003 |
| tool_use | 43.86 | 36.66 | 36.62 | −0.001 |
| **OVERALL** | 6.65 | 6.61 | **6.55** | **−0.008** |

Category shifts are within plan-change noise (agentic and general_instruction
offset); overall a wash. The blended plan costs nothing on text.

## Multimodal answer-NLL — the decisive measurement

New evaluator `reap_stream/eval_multimodal_nll.py`: real images through the
BF16 vision tower → merged embeds → full forward → NLL on **answer tokens
only**. Held-out set built by `scripts/build_multimodal_eval_set.py`
(`calib/multimodal_eval/`): 150 ChartQA **test** rows (every calib pass used
train) + 100 VQAv2 validation rows offset past the 300 used by vision
saliency; image + query + answer read from the same dataset row in one pass.
Identical 250 rows for both students; only the reap plan differs.

| category | p15-old NLL | p15-vblend NLL | Δ | answer tokens |
|---|---|---|---|---|
| chartqa | 9.0017 | 8.9659 | **−0.036** | 438 |
| vqa_natural | 6.8574 | 6.7274 | **−0.130** | 141 |
| **OVERALL** | 8.4795 | 8.4208 | **−0.059** | 579 |

Both categories improve, in the direction the proxy predicted, with the larger
gain on natural photos — consistent with vision/text rank correlation being
lowest exactly where text-only saliency covers least. Caveats: 579 answer
tokens is a small sample and per-row variance was not saved, so treat the
magnitude as indicative rather than precise; the direction, its consistency
across categories, and its agreement with the 31%-less-discarded-mass proxy
are the evidence. Absolute NLL is high because terse one/two-word VQA answers
are intrinsically hard to commit to; no BF16 reference was run (streamed
vision eval not built), so absolute vision damage vs BF16 remains unmeasured.

## Bottom line

**`Step-3.7-p15-4bit-vblend` strictly dominates the original student**: text
unchanged (−0.008 overall NLL), vision measurably better (−0.059 answer NLL),
identical size and speed profile. It should replace `Step-3.7-p15-4bit` as the
deploy model. REAM/DPP/DWQ remain closed per `PPL-DECOMPOSITION.md` — reap
damage is too small for any recovery scheme to have headroom.
