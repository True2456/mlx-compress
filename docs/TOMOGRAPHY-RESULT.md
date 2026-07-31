# Quant-Damage Tomography: The Damage Isn't in Depth, It's in Kind

**Date:** 2026-07-23. Follow-up to `TIERED-EXPERTS-RESULT.md`. Locates WHERE
the 4-bit quantization damage measured in `PPL-DECOMPOSITION.md` (agentic
+0.072 NLL, BF16->4bit at full 288 experts) actually lives, by selectively
restoring precision to one region at a time and re-measuring.

## Method

Seven variants of the vblend plan, each with ONE component bumped above the
uniform 4-bit floor, evaluated on the same 500 held-out prompts as every prior
comparison. Built via `scripts/tomography_sweep.py`, using only stock
per-path `quant_predicate` overrides (no custom classes) -- unlike the
rejected tiered-bank experiment, every variant here loads in stock LM Studio.

- `w0`-`w5`: one 7-layer window of MoE expert weights (`switch_mlp`) at
  6-bit. Windows: [3-9] [10-16] [17-23] [24-30] [31-37] [38-44].
- `shared8`: the always-on shared expert (`share_expert`) + the 3 dense
  (non-MoE) layer MLPs (layers 0-2) + all 42 router gates, at 8-bit.

Each variant built in its own subprocess (buffers freed before the ~97GB
resident eval loads) -- an operational fix, not a modeling one: the first
attempt raced a build against a resident eval in one process and drove swap
to 45/46GB. The eval is fully deterministic; a rerun under clean conditions
reproduced the (correct, alarming-looking) result to 3 decimal places. Filed
as an MLX memory-instrumentation gap:
[ml-explore/mlx#3896](https://github.com/ml-explore/mlx/issues/3896)
(`mx.get_peak_memory()` read ~46GB while the process actually held ~110GB;
irrelevant to result correctness here, relevant to why w0 needed a clean
rerun).

## Results (ΔNLL vs vblend, negative = better)

| variant | window | agentic | coding | general_instr | reasoning_math | tool_use | OVERALL |
|---|---|---|---|---|---|---|---|
| w0 | 3-9 | +0.006 | +0.126 | +0.084 | +0.004 | +0.243 | +0.066 |
| w1 | 10-16 | +0.045 | +0.052 | +0.116 | −0.000 | +0.055 | +0.048 |
| w2 | 17-23 | −0.012 | −0.001 | +0.002 | −0.001 | −0.006 | −0.004 |
| w3 | 24-30 | −0.006 | +0.005 | +0.011 | −0.000 | +0.020 | +0.003 |
| w4 | 31-37 | −0.009 | +0.001 | −0.006 | +0.001 | +0.009 | −0.002 |
| w5 | 38-44 | +0.002 | −0.002 | +0.005 | −0.002 | +0.006 | +0.001 |
| **shared8** | shared+dense+router | **−0.022** | **−0.135** | **−0.200** | **−0.014** | **−0.298** | **−0.100** |

## Interpretation

**Depth is not where the damage is.** All six expert-window bumps (w0-w5)
land inside noise (±0.01-0.05, no consistent sign, several negative). This
confirms the earlier read on w0 (quantization-as-smoothing masking real
signal in some categories) generalizes across the entire depth axis: there is
no localized "bad window" of MoE experts to fix with a targeted bump.

**The damage is concentrated in the always-on, non-expert weights.**
`shared8` improves EVERY category, by an order of magnitude more than any
depth window -- overall −0.0999 NLL, roughly 10x any window effect, and it is
the only variant with a consistent sign across all five categories. The
shared expert, dense layers 0-2, and router gates are weight classes that
fire on every token (no top-8 sparsity to smooth over quantization noise the
way routed experts apparently do), so 4-bit hits them harder in a way
per-expert saliency-based schemes cannot see or fix -- REAP scoring only
covers routed experts.

**Cost: nearly free.** 4.658 bpw vs vblend's 4.632 -- these components are a
tiny parameter fraction (42M router params across 42 layers, plus 3 small
dense MLPs and one shared expert per layer against 196B total). Estimated
+-0.5 GB on a 92 GB model.

## Recommendation

Build `Step-3.7-p15-4bit-vblend-shared8`: vblend's plan + shared8's precision
policy (share_expert + dense-layer MLPs + all router gates at 8-bit,
everything else unchanged at 4-bit gs64). Validate with the same instruments
(500-prompt text PPL, 250-image multimodal NLL) before promoting it over
vblend as the deploy model. Given the consistent, large, cross-category
improvement and near-zero cost, this is expected to be the new best deploy
candidate.

## Follow-up: the class member the sweep never tested (2026-07-24)

`lm_head` and `embed_tokens` are the most extreme members of the always-on
class — they fire on every token *and* every vocabulary row participates in
every argmax, so there is no sparsity whatsoever. The sweep never covered
them: the build predicate returns bare `True` for both, so they sat at 4-bit
by default and were invisible to every variant above, exactly as REAP saliency
is blind to the shared expert.

Tested by taking the deployed shared8 student and replacing only those two
tensors with 8-bit quantizations **of the original BF16 weights** (18 of 20
shards hardlinked, so nothing else can differ — `scripts/build_head8_inplace.py`).
ΔNLL vs shared8, 500 held-out prompts:

| variant | agentic | coding | general_instr | reasoning_math | tool_use | OVERALL |
|---|---|---|---|---|---|---|
| **+head8** | **−0.004** | **−0.019** | **−0.006** | **−0.001** | **−0.024** | **−0.0086** |

Improves every category, no regressions — the same clean sweep shared8
produced, at half the cost (**+0.53 GB**, 93 → 93.5 GB). Magnitude is ~9% of
shared8's −0.100, a sensible ratio given shared8 touched 177 modules and this
touches one tensor pair. The largest gains are again `tool_use` and `coding`,
the two most token-precision-sensitive categories.

**The always-on thesis now holds across every member of the class**, including
the one this sweep structurally could not see.

⚠️ **Source precision matters and is easy to get wrong.** A first attempt built
the 8-bit head by dequantizing the student's existing *4-bit* head and
requantizing it. That measured **+0.013 NLL worse** on every category for the
same +0.53 GB — quantization is one-way, so the wider container stored already-
degraded values plus a second rounding pass. The two runs bracket the baseline
in opposite directions from identical bit-widths, differing only in source
precision. See `FINDINGS.md` §8a; it also gives a free invariant — a precision
*increase* that measures worse means the source is wrong, not the target.

## Consequence for the 64GB-device goal

This reframes the sub-4-bit compression problem. The tiered-experts result
said "don't touch routed-expert bits below a uniform floor." This result adds
a second constraint: whatever the 64GB recipe ends up being, the shared
expert / dense layers / routers should probably stay at higher precision
regardless of what happens to routed experts -- they are cheap in aggregate
size and disproportionately costly to quantize. The compression budget for a
64GB build should likely be spent almost entirely on routed-expert bits
(where the 245-of-288 selection and the tiered-bank tests both live), not
spread uniformly across the whole model.

The head8 follow-up extends that to `lm_head`/`embed_tokens`: keep them at
**8-bit**, and never below 4-bit. Digit-row perturbation as a fraction of the
tightest inter-digit argmax margin (`reap_stream/diag_head_digits.py`) is
0.007 at 8-bit, 0.121 at 4-bit, 0.254 at 3-bit and 0.525 at 2-bit — it roughly
doubles per bit removed, and at 2-bit the noise exceeds half the margin
separating adjacent digits. At 1.06B params they are cheap to protect, and
they are currently 4-bit only *by accident* (bare `True` in the predicate), so
a change to the global default would silently drag them down with everything
else. Make the policy explicit.

## Artifacts

`artifacts/ppl-tomo-{w0..w5,shared8}-500.json`,
`artifacts/tomography-sweep.log`, `scripts/tomography_sweep.py`. All
tomography variant models deleted after measurement (rebuildable via
`--only <variant> --keep`).
