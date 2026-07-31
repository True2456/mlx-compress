# REAM: built, measured, rejected — and a clean case of PPL over-crediting

**Date:** 2026-07-25. Merging low-saliency experts (instead of pruning them)
was built end-to-end and evaluated. **Verdict: rejected as a deploy candidate.**
Its large perplexity gain turned out to be a smoothing artifact with no
capability behind it — the most useful thing the exercise produced.

## What was built

`reap_stream/ream.py` (+ `test_ream.py`, 6/6 unit tests) and
`scripts/build_student_ream.py`. Each pruned expert is merged into its
most-similar kept expert (router-row cosine), and the kept expert becomes the
saliency-weighted average of itself plus everything it absorbed. Router matrix
and bias are merged the same way. Output count is identical to pruning (245/288)
— REAM changes the kept experts' *values*, not the count.

Built clean on the first end-to-end run: 1806 experts merged (43×42), 4.683
bpw, 93 GB, coherent and correct generation (17×23=391, exact Fibonacci, exact
tool-call JSON, correct Rayleigh-scattering prose). The merge code, which had
only seen synthetic tensors, worked on the real 198B model.

## The trap: perplexity said it was a big win

500 held-out prompts, vs the shared8-head8 deploy model:

| category | shared8-head8 ppl | REAM ppl | ΔNLL |
|---|---|---|---|
| tool_use | 26.54 | **15.57** | −0.533 |
| coding | 6.36 | **4.66** | −0.312 |
| general_instruction | 5.08 | **4.10** | −0.215 |
| agentic | 6.90 | **5.95** | −0.148 |
| reasoning_math | 2.48 | 2.50 | **+0.007** |
| **OVERALL** | 5.880 | **4.843** | **−0.194** |

−0.194 NLL overall is ~2× the shared8 + head8 gains *combined*, from a weight
merge, on a model measured to have no exploitable slack. Two tells said "don't
believe it": the magnitude was implausible, and the gains scaled with category
entropy while **reasoning_math — the one category with a single right answer —
did not improve.** That is the textbook signature of smoothing: averaging expert
weights flattens the output distribution, which perplexity rewards on
hedge-friendly text without improving the model.

## The arbiter: exact-answer accuracy

Perplexity cannot distinguish a real gain from smoothing, so both models were
scored right/wrong on 24 exact-answer items (14 multi-step arithmetic + 10
factual) where smoothing cannot help — same scorer, temperature 0, parallel 1.
`scripts/accuracy_eval.py`, results in `artifacts/acc_{ream,shared8}.json`.

| model | overall | math | factual | PPL |
|---|---|---|---|---|
| **shared8-head8 (prune)** | **24/24** | 14/14 | 10/10 | 5.880 |
| REAM (merge) | 23/24 | 13/14 | 10/10 | 4.843 |

**The −0.194 PPL advantage bought zero accuracy** — REAM is one item behind
(noise). Every prediction held: PPL said math wouldn't improve, and math
accuracy is level; the PPL gains were all in high-entropy categories, exactly
where a flatter distribution lowers perplexity for free.

## Follow-up: the agentic categories, tested directly (the important one)

The accuracy test above was math + factual -- the category where PPL predicted
no gain. But REAM's *biggest* PPL gains were tool_use (-0.53) and coding, which
that test did not cover. So it was re-run on 15 tool-call items scored right/
wrong: correct function name (from a provided catalog) AND correct extracted
arguments -- exactly what smoothing cannot fake. `scripts/toolcall_eval.py`,
`artifacts/tc_{ream,shared8}.json`.

| test | shared8-head8 (prune) | REAM (merge) |
|---|---|---|
| tool-call accuracy | **14/15** | **14/15** |
| tool_use *perplexity* | 26.5 | 15.6 (**-0.53 NLL**) |

**A 41% lower perplexity on tool_use converted to zero tool-call capability --
a dead tie.** This is the strongest case against PPL in the whole project: its
largest and most agentic-relevant signal (-0.53 NLL on the exact category that
matters for agent work) was worth nothing in practice. If PPL were trustworthy
anywhere it would be where its signal is biggest; that is precisely where it
most oversold.

## Conclusions

1. **Pruning stays.** REAM is not a deploy candidate. On this prune-resistant,
   flat-saliency model, merging the least-salient experts produces a mushier
   model that scores better on perplexity and no better on task accuracy -- including tool-call/agentic tasks, tested directly.
2. **This is the session's central lesson, demonstrated.** PPL moved −0.194
   while real capability was flat-to-worse. Trusting the perplexity number would
   have shipped a worse model as an upgrade. Perplexity is a proxy; on any
   quality change, confirm with a non-PPL arbiter before promoting.
3. **The co-occurrence gate would have predicted this** (docs/FINDINGS.md §10):
   flat saliency + prune-resistance implies merge partners are near-orthogonal,
   so blending them averages away specialization. We skipped that gate to
   measure directly; the direct measurement agrees.

## Kept

Build + eval code retained — REAM is a correct, working method, just not a win
*here*. It could still help a genuinely redundant MoE (one with a real
low-saliency tail), and `assign_merges` accepts a co-occurrence matrix for a
principled partner choice when that data exists. Artifacts:
`artifacts/ppl-ream-shared8-head8-500.json`, `artifacts/acc_ream.json`,
`artifacts/acc_shared8.json`, `scripts/accuracy_eval.py`, `reap_stream/ream.py`.
The 93 GB REAM checkpoint can be deleted (rebuildable from
`scripts/build_student_ream.py`).
