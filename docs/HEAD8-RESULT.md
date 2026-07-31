# head8: the always-on weight class the sweep never tested

**Date:** 2026-07-24. `lm_head` and `embed_tokens` at 8-bit instead of 4-bit,
on top of the shared8 deploy model. Improves every measured category for
+0.53 GB. Found while investigating a numeric-corruption report
(`TOKENIZER-INVESTIGATION.md`) — the head was the prime suspect, was cleared,
and was then noticed to be untested.

## Why it was invisible

`SHARED8-RESULT.md`'s thesis: components that fire on **every token** carry the
quantization damage, because they get no implicit smoothing from top-k partial
activation. `lm_head`/`embed_tokens` are the extreme case — every token, and
every one of the 128,896 vocabulary rows participates in every argmax.

But the build predicate returns bare `True` for them, so they inherited the
4-bit default, and `tomography_sweep.py`'s variants were six layer windows plus
shared8. The head was never a variant. It is the same blind spot twice over:
REAP saliency cannot see the shared expert, and the tomography sweep could not
see the head.

## Method

Rather than rebuild 93 GB (the disk had no room, and a fresh build could differ
elsewhere), `scripts/build_head8_inplace.py` takes the deployed shared8
checkpoint, **hardlinks the 18 untouched shards**, and rewrites only the two
holding the head tensors — 5.7 GB of new bytes, ~2 minutes. Shared inodes prove
nothing outside those two tensors changed.

The 8-bit weights are quantized from the **BF16 base**, not from the student's
existing 4-bit head. This matters — see the failure below.

## Result (ΔNLL vs shared8, 500 held-out prompts, negative = better)

| category | BF16 | shared8 | **+head8** | ΔNLL |
|---|---|---|---|---|
| agentic | 6.796 | 6.921 | **6.896** | −0.0037 |
| coding | 8.016 | 6.488 | **6.363** | −0.0194 |
| general_instruction | 6.184 | 5.115 | **5.082** | −0.0064 |
| reasoning_math | 2.472 | 2.485 | **2.482** | −0.0012 |
| tool_use | 43.855 | 27.177 | **26.544** | −0.0236 |
| **OVERALL** | 6.650 | 5.930 | **5.880** | **−0.0086** |

Every category improves, no regressions — the same clean sweep shared8 produced,
at half the cost. Magnitude is ~9% of shared8's −0.100, proportionate given
shared8 touched 177 modules and this touches one tensor pair. Largest gains are
`tool_use` and `coding`, the two most token-precision-sensitive categories,
mirroring shared8's pattern.

**Cost:** +0.53 GB (1.06B params, 4.5 → 8.5 bpw). 93 → 93.5 GB, ~4.68 bpw.

## The failed first attempt (kept — it measures something useful)

The first build dequantized the student's existing **4-bit** head and
requantized it to 8-bit. Same tensors, same bit-width, same +0.53 GB:

| | OVERALL | vs shared8 |
|---|---|---|
| shared8 (4-bit head) | 5.930 | — |
| head8 **from BF16** | **5.880** | **−0.0086** |
| head8 **from the 4-bit head** | 6.010 | **+0.013** |

The two runs bracket the baseline in opposite directions, differing *only* in
source precision. Quantization is one-way: the wider container stored
already-degraded values and added a second rounding pass. Written up as
`FINDINGS.md` §8a, because it generalizes — converting a released quantized
checkpoint to another format is a quantization of an already-damaged model, not
a requantization of the original.

It also yields a free invariant: **more bits cannot beat the source**, so a
precision *increase* that measures worse means the source is wrong, not the
target. That is what caught this one.

## How low can the head go?

`reap_stream/diag_head_digits.py` measures quantization perturbation on the
digit rows against the tightest inter-digit argmax margin (0.8463 in BF16):

| bits | gs | ratio | cosine | size |
|---|---|---|---|---|
| 8 | 64 | 0.007 | 0.999986 | 1.12 GB |
| 6 | 64 | 0.029 | 0.999764 | 0.86 GB |
| 4 | 64 | 0.121 | 0.995886 | 0.59 GB |
| 3 | 64 | 0.254 | — | 0.51 GB |
| 2 | 64 | 0.525 | — | 0.36 GB |

The ratio roughly doubles per bit removed; at 2-bit the noise exceeds half the
distance between adjacent digit rows. BF16 is not worth it — 8-bit is already
at 0.7% of the margin with cosine 0.999986, and BF16 costs +0.99 GB beyond
8-bit to remove an error nothing can resolve.

**Policy for the 64GB recipe:** head at 8-bit; never below 4-bit. Now stated
explicitly in the predicate rather than inherited from the default, so lowering
the global bit-width cannot drag it down silently.

## Status

- `scripts/build_student_shared8.py` predicate updated — head at 8-bit.
- **Promoted**: deployed at
  `~/.lmstudio/models/truemod/Step-3.7-p15-vblend-shared8-head8` (symlinked into
  `models/`), loaded at 262144 context / parallel 4. Note LM Studio does not
  inherit per-model context or sampler settings across a checkpoint swap.
- **Not uploaded to HF** — blocked by the private-repo storage limit. The Hub
  still carries the 4-bit-head weights.
- Multimodal NLL (250 held-out images) **not yet re-run** — shared8 was
  validated on both instruments and this has only text PPL so far.

## Artifacts

`artifacts/ppl-shared8-head8-500.json`,
`artifacts/ppl-head8-requant-from-4bit-500.json`,
`artifacts/ppl-p15-vblend-shared8-500.json` (baseline),
`scripts/build_head8_inplace.py`, `reap_stream/diag_head_digits.py`.
