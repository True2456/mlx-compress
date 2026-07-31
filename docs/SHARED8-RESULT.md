# vblend + shared8: The New Deploy Model

**Date:** 2026-07-23. Combines the two validated fixes from this session's
work: the vision-blended REAP plan (`VISION-BLEND-RESULT.md`) and the
shared-expert/dense/router 8-bit precision policy located by the tomography
sweep (`TOMOGRAPHY-RESULT.md`). Built and validated with the same instruments
used throughout: 500-prompt held-out text PPL, 250-image held-out multimodal
answer-NLL.

Built by `scripts/build_student_shared8.py`: REAP p15 vision-blend plan
(245/288 experts) + share_expert/dense-layer-MLPs(0-2)/router-gates at 8-bit,
everything else 4-bit gs64 affine, vision tower BF16. Stock per-path
quantization config -- no custom classes, loads in LM Studio like any
quantized checkpoint. **93 GB, 4.658 bpw** (vblend was 92 GB / 4.632).

## Results vs vblend (previous deploy model)

Text, 500 held-out prompts:

| category | BF16 | vblend | **shared8** | ΔNLL |
|---|---|---|---|---|
| agentic | 6.80 | 7.08 | **6.92** | −0.022 |
| coding | 8.02 | 7.43 | **6.49** | −0.135 |
| general_instruction | 6.18 | 6.25 | **5.12** | −0.200 |
| reasoning_math | 2.47 | 2.52 | **2.49** | −0.014 |
| tool_use | 43.86 | 36.62 | **27.18** | −0.298 |
| **OVERALL** | 6.65 | 6.55 | **5.93** | **−0.100** |

Multimodal, 250 held-out images (chartqa test + vqa_natural, never used in
calibration):

| category | vblend | **shared8** | Δ |
|---|---|---|---|
| chartqa | 8.966 | **8.949** | −0.017 |
| vqa_natural | 6.727 | **6.653** | −0.075 |
| **OVERALL** | 8.421 | **8.390** | **−0.031** |

**shared8 improves every category on every instrument measured this
session**, for +1 GB. This is now the largest single quality gain of any
change tested against the original text-only p15 4-bit student -- larger than
the vision blend, and in the opposite direction of the (rejected) tiered
experiment.

## Why this worked when tiering routed experts didn't

The tiered-experts result showed routed MoE experts have no exploitable slack
-- flat saliency, no negligible tail. This result shows the opposite is true
for a different weight class entirely: the shared expert, dense layers, and
router gates fire on every token (no top-8 sparsity), so there is no
implicit smoothing/regularization from partial activation the way routed
experts get. REAP saliency scoring never touches this weight class at all --
it is invisible to every pruning/tiering scheme tried this session, and it
was where the real headroom was.

## Deploy status

Moved into `~/.lmstudio/models/truemod/Step-3.7-p15-4bit-vblend-shared8`,
replacing vblend as the symlinked deploy model at
`models/Step-3.7-p15-4bit-vblend-shared8`. `Step-3.7-p15-4bit-vblend` (92 GB)
retained pending confidence in the new model in actual use; safe to delete
once satisfied (fully reproducible from `artifacts/plans/plan_p15_blend03.json`
+ `scripts/build_student.py`).

## Open threads

- 64GB-device recipe: per `TOMOGRAPHY-RESULT.md`, spend low-bit budget on
  routed experts, keep shared/dense/router weights at 8-bit regardless --
  this result is the concrete confirmation that policy is worth keeping.
- DWQ: still closed at this bit-width (no proven headroom), but worth
  revisiting once a low-bit routed-expert build exists, where a real gap is
  likely to open up.
