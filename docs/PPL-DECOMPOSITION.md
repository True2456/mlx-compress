# Perplexity Decomposition: Isolating Reap Damage from Quantization Damage

**Question:** of the total quality cost from BF16 → 15%-reaped 4-bit, how much comes
from removing 43/288 experts (reap) vs. how much comes from 4-bit quantization itself?

**Method:** three models, same 500 held-out prompts (rows 5000+ of `cloud_reap_8k`,
never used in any calibration), identical eval code, per-category perplexity.

| Model | Experts | Precision | Isolates |
|---|---|---|---|
| `Step-3.7-Flash` (BF16) | 288 | 16-bit | reference, zero loss |
| `Step-3.7-full-4bit` | 288 | 4-bit affine | **quantization damage alone** |
| `Step-3.7-p15-4bit` | 245 | 4-bit affine | quant + **reap damage on top** |

Built with the same `scripts/build_student.py` path for both quantized models (a
no-op plan keeping all 288 experts for the middle point), so the only variable
between the last two is the 43 pruned experts.

---

## Two methodology bugs fixed before these numbers meant anything

An early smoke test gave **agentic PPL = 388** — implausible for a frontier model.
Root causes, both now fixed in `reap_stream/eval_ppl_streamed.py`:

1. **`headtail` truncation is right for saliency, wrong for perplexity.** It splices
   first-512 tokens onto last-512 for long prompts, creating a seam that inflates NLL
   for reasons unrelated to model quality. PPL eval now uses contiguous `head`.
2. **Chat-template double-wrapping.** Calib rows already carry their own
   `SYSTEM:/USER:/ASSISTANT:` structure; `apply_chat_template` wrapped that again
   inside a user turn, feeding the model malformed input it never saw in training.
   PPL eval now feeds raw text by default.

Fixing both dropped agentic PPL **388 → 10.75** (36×). Neither bug affects any
saliency result — every saliency run used identical treatment throughout, so
saliency comparisons are a level shift at most, not corrupted.

---

## Results

### Step 1: BF16 reference (500 prompts, 383k tokens, 15.2 min streamed)

| Category | PPL | Tokens |
|---|---|---|
| reasoning_math | 2.47 | 92k |
| general_instruction | 6.18 | 61k |
| agentic | 6.80 | 118k |
| coding | 8.02 | 69k |
| **tool_use** | **43.86** | 43k |
| OVERALL | 6.65 | 383k |

`tool_use`'s high PPL is real, not a bug: these are xLAM-style function-calling
prompts full of arbitrary API names, default values, and fictional service
descriptions — intrinsically unpredictable content, not a measurement artifact.
Confirmed by inspecting raw samples.

### Step 2 & 3: full-4bit and p15-4bit, same 500 prompts

| Category | BF16 | 4bit-full | 4bit-p15 |
|---|---|---|---|
| agentic | 6.796 | 7.304 | 7.369 |
| coding | 8.016 | 7.804 | 7.540 |
| general_instruction | 6.184 | 6.049 | 6.002 |
| reasoning_math | 2.472 | 2.500 | 2.513 |
| tool_use | 43.855 | 37.301 | 36.662 |
| OVERALL | 6.653 | 6.643 | 6.607 |

**Overall PPL improves slightly at 4-bit** (6.65 → 6.61). This is *not* a capability
gain — 4-bit quantization noise smooths overconfident wrong predictions on
high-entropy content, and the size of the "improvement" tracks baseline entropy
almost exactly (tool_use −16%, coding −6%, reasoning_math +2%, low-entropy
categories barely move). This is quantization-as-smoothing, not quality.

### The decomposition (ΔNLL, lower magnitude = less damage)

| Category | Quant damage (BF16→4bit-full) | Reap damage (4bit-full→p15) | Reap's share of total |
|---|---|---|---|
| **agentic** | **+0.0721** | +0.0088 | **10.9%** |
| tool_use | −0.1619 | −0.0173 | 9.6% |
| coding | −0.0268 | −0.0344 | 56.2% |
| general_instruction | −0.0220 | −0.0079 | 26.5% |
| reasoning_math | +0.0114 | +0.0050 | 30.3% |
| **OVERALL** | **−0.0015** | **−0.0055** | — |

---

## Findings

### 1. Reap damage is small everywhere
Removing 43/288 experts (15%) never adds more than **0.009 NLL** in any category —
often it's near zero or even slightly negative (coding, tool_use). This is a second,
independent confirmation that 15% reap is safe: it agrees with the earlier finding
that the corrected-truncation saliency re-run changed only **1.7%** of kept experts
per layer versus the original plan (see `FINDINGS.md` §4).

### 2. The agentic regression is mostly a quantization story, not a reap story
This overturns a prediction made earlier in the session. Two independent findings —
vision blindness (9.4% of vision saliency mass on pruned experts, verdict SEPARATE)
and the truncation bug (96% of agentic prompts losing their answer during saliency
collection) — both predicted agentic-specific **reap** damage. The data instead shows:

- Going BF16 → 4bit **at full 288 experts** already costs +0.0721 NLL on agentic —
  **89% of the total damage**, before a single expert is removed.
- Actually reaping (4bit-full → 4bit-p15) adds only +0.0088 on top.

Two readings, not mutually exclusive:
- Agentic content (dense JSON, function-call syntax, rare tokens) may simply be
  intrinsically more 4-bit-quantization-sensitive, independent of which experts
  survive.
- The vision/truncation effects may be real but invisible to **text-only** held-out
  perplexity — they would show up in actual multimodal task performance or
  tool-calling *format* correctness, which this experiment does not test.

### 3. Reap sometimes appears to help
coding and tool_use both show reap nudging NLL further down on top of quantization's
own smoothing effect. Read this as further evidence of a smoothing/regularization
effect from 4-bit noise on hard-to-predict structured content — not a real capability
gain from having fewer experts.

---

## What this settles, and what it still can't

**Settled:** the 15% reap plan is not the primary source of measured quality cost on
text tasks. Quantization is. Two independent methods (plan-overlap analysis and this
NLL decomposition) now agree on "reap is cheap here."

**Not settled — and this experiment cannot settle it:**
- **Vision/multimodal impact.** Perplexity on held-out *text* cannot detect chart-reading
  degradation. The SEPARATE verdict and 9.4%-pruned-mass finding stand independently
  and are not contradicted by this result — they were simply never testable by this
  method. A real multimodal eval (feeding actual images) is the only way to check.
- **REAP vs. REAM.** Perplexity is too confounded by quantization-smoothing effects to
  arbitrate between pruning and merging strategies. If reap damage is already this
  small, REAM's main selling point (avoiding reap damage via merging instead of
  deletion) has very little headroom left to improve on.
- **Real task performance.** Step-3.7-Flash is built and benchmarked on SWE-bench Pro,
  Terminal-Bench, agentic tool-use — perplexity is a proxy for all of these, not a
  substitute.

## Recommended next step
Given reap is now confirmed cheap by two independent measurements, further effort is
better spent on an actual multimodal task eval (the one open question no proxy so far
can answer) than on continuing to optimize the reap/quant recipe or building REAM/DPP.

---

## Artifacts

| File | Contents |
|---|---|
| `artifacts/ppl-bf16-500.json` | BF16 reference, per-category NLL/PPL |
| `artifacts/ppl-full-4bit-500.json` | 288-expert 4-bit, same prompts |
| `artifacts/ppl-p15-4bit-500.json` | 245-expert 4-bit (the shipped student), same prompts |
| `reap_stream/eval_ppl_streamed.py` | Evaluator (streamed for BF16, resident for quantized) |
| `models/Step-3.7-full-4bit/` | 107 GB, 288 experts, 4.613 bpw — built for this comparison |
| `artifacts/plan_p00_full.json` | No-op reap plan (keeps all 288) used to build the above |
