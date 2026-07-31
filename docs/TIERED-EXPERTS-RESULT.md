# Saliency-Tiered Expert Quantization: Built, Measured, Rejected

**Date:** 2026-07-23. Tests whether per-expert mixed-precision quantization —
allocating bits by REAP saliency rank instead of uniformly — beats uniform
4-bit at (approximately) equal size. **It does not. Uniform wins, clearly.**

## Setup

- Model: `Step-3.7-p15-tiered` — same vision-blended p15 plan as the deploy
  model, but each layer's 245 kept experts split into three physical banks by
  blended-saliency rank: **61 hot @ 6-bit / 123 base @ 4-bit / 61 cold @
  3-bit**, gs64 affine, everything else identical to vblend.
- Built by `scripts/build_student_tiered.py`; bank dispatch
  (`reap_stream/tiered.TieredSwitchGLU`) unit-tested **bit-exact** against
  stock SwitchGLU, so routing/selection is provably identical — only storage
  precision differs. Smoke generation coherent.
- Size honesty: came out **97 GB / 4.868 bpw** vs vblend's 92 GB / 4.632 —
  the 6-bit bank's scale overhead exceeds the 3-bit bank's savings, so tiered
  was actually ~5% *larger*, making its loss even less ambiguous.
- Note: the tiered model is a measurement instrument only. Stock LM Studio
  resolves step3p7 from its bundled mlx_vlm and cannot load bank classes;
  evals auto-patch via `reap_stream.tiered.maybe_patch_tiered`. The dispatch
  also runs ~3× expert FLOPs (each bank computes all top-8 slots, masked) —
  fine for evals, deliberately not production.

## Results (same instruments as all prior comparisons)

Text, 500 held-out prompts:

| category | BF16 | vblend (uniform 4-bit) | tiered | ΔNLL tiered−vblend |
|---|---|---|---|---|
| agentic | 6.80 | 7.08 | 7.76 | +0.091 |
| coding | 8.02 | 7.43 | 8.78 | +0.167 |
| general_instruction | 6.18 | 6.25 | 7.63 | +0.200 |
| reasoning_math | 2.47 | 2.52 | 2.53 | +0.006 |
| tool_use | 43.86 | 36.62 | 44.82 | +0.202 |
| **OVERALL** | 6.65 | **6.55** | 7.35 | **+0.114** |

Multimodal answer-NLL, 250 held-out images: overall −0.043 (chartqa −0.084,
vqa_natural +0.086) — a wash within noise, and irrelevant next to the text
regression.

## Interpretation

1. **The 3-bit cold bank did the damage.** +0.114 overall NLL is ~13× the
   entire reap cost of removing 43 experts outright. The bottom quartile of
   experts by saliency still carries real, irreplaceable signal.
2. **This is the flat-saliency property again.** The same measured fact that
   makes this model prune-resistant (no dead experts, near-uniform saliency,
   capacity is real) makes it tier-resistant: there is no negligible tail to
   raid for bits. Saliency rank is not quantization tolerance.
3. **The category fingerprint matches routing breadth.** reasoning_math
   (narrow, hot-expert routing) was untouched (+0.006); broad-distribution
   categories that spread routing across many experts (general_instruction,
   tool_use, coding) hit the 3-bit bank constantly and paid for it.
4. **Down-tiering is dead; up-tiering is still open.** Nothing here rules out
   *adding* bits above a uniform 4-bit floor (deployable via stock per-module
   quant config, zero runtime cost). That is exactly what the tomography
   sweep (`scripts/tomography_sweep.py`) measures.

## Consequence for the 64 GB-device question

This result makes sub-4-bit compression harder than hoped: if 3-bit on just
25% of experts costs +0.11 NLL, pushing most experts to 2–3 bits (required
for a ~56 GB usable model even with p20–25 reap) will not come free from
clever allocation alone. A 64 GB build will likely need actual recovery
training (DWQ/distillation — the regime where it earns its cost in the
literature), not just smarter static bit assignment.

## Artifacts

`artifacts/ppl-p15-tiered-500.json`, `artifacts/mmeval-p15-tiered.json`,
`artifacts/build-p15-tiered.log`. Model deleted after measurement
(rebuildable in ~2 min from the plan + builder).
