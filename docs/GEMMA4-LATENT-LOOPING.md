# Gemma-4 latent-space reasoning retrofit: goals, method, progress

Related but distinct from [layer-looping-notes.md](layer-looping-notes.md),
which explores training-free layer repetition/tying on Step-3.7. This is a
different mechanism entirely: a trained, continuous-thought curriculum
retrofit on Gemma-4, aimed at the project's actual target use case (agentic
coding / tool use), not a generic reasoning benchmark.

## Goal

Retrofit continuous latent-space reasoning (COCONUT-style: the model feeds
its own hidden state back in as the next input embedding instead of
decoding to text, for some number of "thinking" steps) onto an existing
pretrained Gemma-4 model via QLoRA fine-tuning, rather than training a
recurrent architecture from scratch. If it works, this should let the model
do more "thinking" per generated token without the cost of verbose explicit
chain-of-thought.

Model size: **12B**, chosen for iteration speed while the technique is
unproven, not because of any evidence it's uniquely suited to this size (see
Layer-saliency findings below — the real evidence argues against a size
threshold existing at all in this range).

Precision: **bf16, non-QAT**, for the first validation pass specifically —
isolates whether the technique itself works, without also fighting a
quantization-noise confound on top of an already-novel retrofit. QAT is
irrelevant until/unless a 4-bit deployment target is chosen later.

## Why retrofit is plausible at all

Real, load-bearing precedent, not assumed:

- **COCONUT** (Hao et al., Meta/UCSD, arXiv:2412.06769) — curriculum SFT on
  a *pretrained* GPT-2, progressively replacing explicit CoT sentences with
  continuous thought vectors. This is the method being adapted here.
- **iCoT** (Deng et al., arXiv:2405.14838) — same theme, stepwise CoT
  internalization via fine-tuning.
- **Pause tokens** (Goyal et al., Meta, arXiv:2310.02226) — extra latent
  compute via dummy tokens on a pretrained LLaMA, a lighter-weight relative.

**What doesn't have retrofit precedent**: genuine weight-shared recurrent
architecture (a real "loop this specific middle block" design, à la Huginn,
Ouro, LOTUS — all confirmed via direct paper lookup, all trained from
scratch or near-scratch, none retrofit onto an existing non-recurrent
checkpoint). The original proposal for this project described exactly that
architecture (a "sandwich": frozen prelude → weight-shared looped middle →
frozen coda) with specific numbers (α≈0.85 residual anchoring, 20-25%/50-
60%/15-20% layer split, "5-10x latency reduction") that turned out to be
unsourced — real terminology glued to invented specifics, the same pattern
later confirmed as a live problem when Pi (running this project's own
fine-tuned model) fabricated a "GaLore + Hybrid-MoE + Symmetry Loss" training
plan misreading a data folder as trained weights. **Decision: build the
COCONUT-style whole-model loop, not the sandwich architecture** — it's the
version with actual empirical retrofit precedent.

## Layer-saliency measurement: what actually happened

Before committing to "loop the middle 50-60%" as the original proposal
assumed, we measured it directly rather than trust the guess — and the
result overturned the premise twice.

**Round 1** (`reap_stream/collect_loop_saliency.py`, Block Influence metric —
`BI_i = 1 - mean_cos_sim(h_in, h_out)` per layer, ShortGPT-style): reasoning-
heavy vs. simple prompts. Consistent, reproduced-4-times signal at the very
first/last layers (Gemma-4-12B layer 46, 5, 47, 0), not the middle. Already
contradicted the sandwich premise.

**Round 2, the real correction**: that "stable, reproduced" signal was
re-tested with reasoning and simple prompts padded to equal token length
(neutral, content-free padding). **It collapsed to noise** (+0.005 max, down
from +0.059) — the entire earlier signal was a length-confound artifact
(reasoning prompts were 58 tokens avg, simple were 22), not a real
reasoning-specific effect. Confirmed at both 12B and 31B.

**What survived length control**: a real, length-matched **tool-call-
specific** signal — tool-call prompts (with a genuine `tools` schema) vs.
both reasoning and simple, both length-matched to ~250-290 tokens. Real
differential at layers 12, 4, 32-33, 45 (12B) and 57, 52, 56, 39-42 (31B).
Confirms tool-calling drives genuinely distinctive layer processing;
"reasoning" generically does not, on this metric. **The specific layers
don't transfer between scales** (12B's strongest signal at ~25% depth vs.
31B's at ~95% depth) — a loop-boundary choice validated on 12B would need
re-measurement at 31B, not inheritance by percentage.

Full data: `artifacts/loop-saliency-12b-length-matched/`,
`artifacts/loop-saliency-31b-length-matched/`.

Net effect on the plan: no defensible loop-boundary layer selection exists
yet even for the sandwich design, on top of it lacking retrofit precedent —
two independent reasons to go with whole-model looping instead, which
doesn't require choosing a boundary at all.

## Training data

`build_lora_data.py`'s SFT data (used for the successful agentic-coding
LoRA) slices each trajectory into one row per assistant turn — correct for
ordinary SFT, but throws away the multi-step structure a continuous-thought
curriculum needs to compress.

`scripts/build_loop_curriculum_data.py` reverses that: reconstructs full,
unsliced trajectories (fullest available slice per `source_trajectory_id`),
filtered to ≥3 assistant turns. **3,367 usable trajectories** (2,863 train /
336 valid / 168 test) across Fable-5, GPT-5.6-Sol, Kimi-K3 — median 8
assistant turns per trajectory, up to 106.

Real example (Fable-5, 11-message trajectory): investigate → read 3 files →
edit → verify tests pass → explain root cause. Genuine ReAct-style
reasoning-and-acting, just spread across discrete assistant turns with tool
calls between them rather than packed into one `<think>` block.

Compression candidates: only **nonempty** assistant `content` spans (most
turns in these traces are empty -- just a tool call, nothing to compress).
The final assistant turn is never a candidate -- it's the actual target.

`data/loop_curriculum_data/{train,valid,test}.jsonl`.

## Evaluation harness

Three real, on-policy, license-clean (CC-BY-4.0) failure-mining datasets
for Gemma-4-12B, mined against `gemma-4-12b-it-qat-frontierdistill` (the
QAT+LoRA-fused variant, not the clean bf16 base used here -- caveat baked
into the eval, checked below):

- `True2456/gemma4-onpolicy-50topics-2000-corrections`
- `True2456/gemma4-onpolicy-student-corrections`
- `True2456/gemma4-onpolicy-50topics-corrections`

Exact-match scoring (`scripts/eval_gemma4_corrections.py`), precisely
because these prompts constrain output format ("return only the integer",
"return code line only without markdown formatting") -- avoids the PPL-
over-crediting trap documented in `REAM-RESULT.md`.

**Baseline established** (clean bf16 base, full ~500-row test set,
`artifacts/corrections_baseline_bf16_full.json`):

| Dataset | n | exact-match (orig) | corrected scorer |
|---|---|---|---|
| 50topics-2000-corrections | 200 | 13.0% | 13.0% |
| student-corrections | 200 | 4.0% | **22.0%** |
| 50topics-corrections | 100 | 15.0% | 15.0% |
| **Overall** | **500** | **9.8%** | **17.0%** |

**The 9.8% figure was a scorer artifact and should not be used.** Plain string
equality marked `{"x": 60, "y": 28, "z": -31}` wrong against an expected
`{"x":60,"y":28,"z":-31}` -- the same answer with conventional JSON spacing.
That alone understated `student-corrections` by 18 points. `answers_match()`
now compares JSON structurally.

Deliberately NOT fixed by stripping whitespace globally: that would pass Python
answers whose indentation is semantic, swapping an under-crediting scorer for an
over-crediting one -- the exact failure mode `REAM-RESULT.md` documents.

Model-variant mismatch checked, not assumed: 9.8% (not near-0%) confirms
these documented failures substantially transfer to the clean base too, not
just quirks of the fused variant the data was mined from -- the eval set is
legitimately applicable here.

## Mechanism validation

Before building the full curriculum pipeline: does MLX's autodiff actually
track gradients through the core COCONUT loop (run model → take last hidden
state → feed it back as the next input embedding → run again → ... →
backward)? Isolated smoke test (`reap_stream/smoke_continuous_thought.py`),
3 iterations through the full 48-layer stack, LoRA on one layer. `lora_b`
received a real, nonzero gradient (0.72 norm) after backprop through the
whole chain -- confirmed working. (`lora_a` showing exactly zero gradient on
this first step is a separate, expected artifact of zero-initialized LoRA,
not a mechanism failure -- `lora_b` is the tensor whose gradient doesn't
depend on its own current value.)

`Gemma4TextModel.__call__` natively supports `inputs_embeds`, bypassing the
token-embedding lookup -- no monkey-patching needed, mirrors HF convention.

## Training pipeline

`reap_stream/lora_loop_gemma4.py`. Curriculum stage `k` = number of leading
eligible segments (in trajectory order) to compress. Loss: cross-entropy on
real-token positions only, masked out at continuous-thought positions (no
target token exists there) -- conservative simplification currently also
excludes the position transitioning from the last continuous thought back to
real text from the loss (real available signal being left out, not a
correctness bug -- candidate for a later refinement).

QLoRA-style: all attention (q/k/v/o) + MLP (gate/up/down) projections
wrapped, matching the successful 31B agentic-coding LoRA's target-module
choice. `MAX_TRAJECTORY_TOKENS = 4096` filters extreme outliers (the longest
raw trajectory found was 133 messages / ~66 assistant turns) --
565/2,863 train trajectories survive this filter.

Status as of this writing: Stage 0 (pure SFT warmup, no compression yet --
matches COCONUT's own Stage 0) smoke run in progress. Loss trending
correctly (5.67 → 3.43 → 2.22 over first 10 iterations), first checkpoint
saved cleanly. Stages 1+ (progressive continuous-thought compression) not
yet run.

## Performance audit (post-smoke-test)

The first working smoke run cost ~3.3 min/iteration, which made a realistically
scoped curriculum a multi-day commitment. Profiling the trainer found the cause
was a bug, not an inherent cost of the technique.

**The finding: `grad_checkpoint` was being called once per layer.** It patches
`type(layer).__call__` -- a *class*-level change that already covers all 48
instances, which is why mlx_lm's own trainer calls it exactly once on
`layers[0]`. Calling it in a loop nested 48 checkpoint wrappers around the same
function, so every backward re-ran the whole nested chain.

Measured on a fixed 2048-token, 1-continuous-thought example (12B bf16, rank-8
LoRA on all attn+mlp projections), fwd+bwd only:

| Variant | fwd+bwd | MLX peak |
|---|---|---|
| as-is (`grad_checkpoint` x48) | 115.67s | 61.0 GB |
| `grad_checkpoint` once | 7.43s | 61.0 GB |
| + masks hoisted out of the layer loop | 7.27s | 61.0 GB |
| + chunked/checkpointed vocab projection | 7.37s | 59.9 GB |

**15.6x faster at identical peak memory** -- the nesting bought nothing at all.
The other two changes are within timing noise; the chunked projection is kept
because it bounds the logits tensor (vocab 262144, so a full-sequence logits
array is multi-GB and grows with the token cap), and the mask hoisting is kept
because it matches what `Gemma4TextModel._make_masks` does internally.

**Separate correctness bug found in the same pass**: Gemma-4 sets
`final_logit_softcapping = 30.0` and applies it at inference via
`LanguageModel.logits_from_hidden`, but the training loss projected hidden
states straight through `embed_tokens.as_linear` with no cap -- fitting the
adapter to a different output transform than the deploy-time one. Now applied
in the loss.

**Data-scale finding, and the correction to it**: `MAX_TRAJECTORY_TOKENS` was
4096, but the measured median trajectory (train split, >=1 eligible segment) is
**8016 tokens** -- the cap sat below the median and kept only 565/2512 (22%) of
usable trajectories. It was raised to 8192 (1287 rows, 51%) on that basis.

**That was wrong, and measuring it is what showed why.** A 10-iteration run at
the 8192 cap averaged 122.4s/iter with extreme variance: 1170 tokens took 4.2s,
3616 tokens took 23.4s, but 5063 tokens took **230.9s**. 1.4x the tokens for 10x
the time is not explicable by quadratic attention.

Sweeping sequence length in isolation found the real shape:

| seq | fwd+bwd | real phys_footprint_peak |
|---|---|---|
| 1024 | 3.38s | 45 GB |
| 2048 | 6.91s | 66 GB |
| 3072 | 10.67s | 88 GB |

Time and memory are **linear up to 3072** -- ~3.5s and ~21 GB per 1024 tokens.
Then the trend breaks:

| seq | fwd+bwd | real peak | vs linear trend |
|---|---|---|---|
| 4096 | 74.84s | 110 GB | ~14s predicted, **5.3x off** |

**The nonlinearity is the RAM ceiling, not compute.** Two cautions learned here:

1. *Measurements near the ceiling are not reproducible without controlling
   machine state.* The 4096 point first measured **145.52s**, then **74.84s**
   from a clean state after killing background processes and letting swap
   settle -- a 2x difference from contamination alone. The first number was
   nearly written into this doc as fact.
2. During the clean run swap barely moved (3930 -> 3980 MB), so this is likely
   macOS **memory compression** near the ceiling rather than classic paging.

Since only the tail is expensive (median trajectory is ~1400-1700 tokens), the
cap is chosen by arithmetic over the length distribution:

| cap | rows | median len | blended cost |
|---|---|---|---|
| 3072 | 434 | 1370 | ~7s/iter |
| 4096 | 565 | 1709 | ~23.5s/iter (23% of rows sit above 3072 at ~75s each) |

3.4x the per-iteration cost for 30% more data is a bad trade, so the cap is
**3072**. Buying back the lost coverage (this keeps only 434/2512 = 17% of the
corpus) means reducing activation memory, not raising the cap -- ~21 GB per
1024 tokens is high for a 12B with gradient checkpointing on, and that is the
open lead.

**Curriculum-depth finding, and the two bugs behind it**: eligible segments per
trajectory initially measured only ~1.5-1.9, implying the planned 8-stage
curriculum was unsupported by the corpus. Investigating *why* found that most of
the shortfall was our own bugs, not the data.

*Context on the corpus.* The 2,863 train trajectories contain 27,146 assistant
turns (the ~16k figure from earlier SFT work counted turn-level slices, a
different unit). Only 36% of those turns have nonempty `content` -- the rest are
pure tool calls with nothing to compress. That much is a genuine property of
agentic traces.

*Bug 1: `reasoning_content` was ignored.* Assistant turns also carry a
`reasoning_content` field, which the Gemma-4 template renders inside
`<|channel>thought ... <channel|>` -- verified present in the token stream, not
dropped. 1,460 turns carry it with **no** `content` at all, so keying
eligibility on `content` alone made them invisible. This is the single best
compression target in the corpus: deliberation emitted immediately before an
action, which is precisely what COCONUT replaces.

*Bug 2: spans covered whole messages, including tool calls.* The original
`_segment_token_span` diffed rendered-prefix lengths, which yields the span of
the entire message. **41% of eligible segments also carry a tool call**, so that
silently replaced the action with continuous thoughts, leaving the following
tool-response message with nothing that produced it -- corrupting exactly the
ReAct structure this corpus was chosen for. Replaced with precise character-to-
token span mapping (fast tokenizer `offset_mapping`, relying on the template's
per-message character-prefix property, verified) that covers only free text.
Validated by decoding 209 spans across 120 trajectories: **zero contain a tool
call**, and every span decodes to deliberation text.

Effect at the 3072 cap:

| | before | after |
|---|---|---|
| usable trajectories | 434 | **684** (+58%) |
| mean eligible segments | 1.5 | **2.24** |
| >=2 segments | -- | 75.6% |
| >=3 segments | -- | 29.7% |

The trajectory count rose because reasoning-only turns made whole trajectories
eligible that previously had zero compressible segments. **A ~3-stage curriculum
is now genuinely supported.** An 8-stage one still is not (11% of trajectories
reach 4 segments), so `MAX_ELIGIBLE_SEGMENTS = 8` remains generous rather than
binding -- but the earlier "stages past ~1 are no-ops" conclusion was mostly an
artifact of the eligibility bug, not the corpus.

Neither bug affected the previously shipped 12B/31B agentic LoRAs: both live in
`lora_loop_gemma4.py` (written for this work, and only ever run at stage 0,
which computes no spans), while those adapters came from `build_lora_data.py` ->
`cloud_train_gemma4.py`, which does ordinary SFT on cumulative slices, never
replaces spans, and explicitly preserves `reasoning_content`.

Also fixed: per-iteration timing/ETA is now logged (it was previously only
inferable from wall clock), and the post-step `mx.eval` now covers only the
trainable params and optimizer state rather than walking the full 12B tree.

### Resulting cost estimate

At the 3072-token cap: **~7s/iteration** blended over the length distribution
(consistent with iterations actually observed -- 1170 tok in 4.2s, 3616 tok in
23.4s). With ~3 usable curriculum stages and 1-2 epochs over the 434 surviving
trajectories, that is roughly **~2,000 iterations, or 3-6 hours** -- not the
multi-day commitment the pre-fix 3.3 min/iter implied.

Two caveats worth keeping attached to that number. It is arithmetic from
measured per-iteration cost, not a measured end-to-end run. And the iteration
*count* actually required for the technique to work is the genuinely unknown
quantity -- profiling cannot answer it, only a real run scored against the
9.8% corrections baseline can.

## First trained result: stage 0 regresses

Stage 0 (700 iters, rank 128, lr 2e-5, 684 trajectories, loss 4.37 -> 0.35)
scored against the corrections eval, corrected scorer:

| Dataset | base | stage 0 |
|---|---|---|
| 50topics-2000-corrections | 13.0% | **3.0%** |
| student-corrections | 22.0% | 23.0% |
| 50topics-corrections | 15.0% | **2.0%** |
| **Overall** | **17.0%** | **10.8%** |

**A 6.2-point regression.** Under the original exact-match scorer this looked
like 9.8% -> 9.4%, i.e. harmless; the scorer fix is what exposed it.

Per-row analysis (42 lost, 40 gained under exact-match) shows the two effects
were of very different kinds:

- **Gains were formatting conformity**, not capability -- compact JSON spacing
  and terseness picked up from agentic tool-call traces, which happen to match
  what `student-corrections` expects. Under the corrected scorer these largely
  evaporate (22.0% -> 23.0%, not 4.0% -> 19.5%).
- **Losses were genuine correctness regressions** on symbolic/numeric work:
  `CGACTT` -> `CGATTT`, `CTTATA` -> `TTAATC`, `5468.4 J` -> `5622`.

So stage-0 SFT on agentic trajectories traded real accuracy for surface style.
This is catastrophic forgetting on the eval's task distribution, and rank 128 at
lr 2e-5 for ~1 epoch is an aggressive enough config to explain it.

**Implication for the experiment**: the arm A / arm B comparison is still
internally valid (both start from the same stage-0 checkpoint), but running it
from a checkpoint that has already lost 6 points of capability measures the
mechanism on damaged ground. Re-tuning stage 0 (lower LR, fewer iterations,
possibly lower rank) should come first.

**Implication for the eval**: it is substantially sensitive to output
formatting, which is a poor instrument for detecting whether latent reasoning
helps. The corrected scorer removes the JSON-spacing component, but the
`50topics` sets remain exact-string comparisons over units and sequences. A
reasoning-sensitive eval is still missing.

## Open questions / not yet done

- Full curriculum run (stages 0 through some K) at real iteration counts,
  not just the current smoke-scale validation.
- Re-run against `data/eval_corrections/` post-training to see if the
  retrofit moves the needle on real documented failures vs. the 9.8%
  baseline -- the actual test of whether this was worth doing.
- The length-matched tool-call-specific layers (12B: 12, 4, 32-33, 45) are a
  live lead for a *targeted* variant if whole-model looping alone
  underperforms -- not pursued yet, whole-model was prioritized as the
  version with real retrofit precedent.
- 31B port: explicitly not assumed transferable from 12B (per the layer-
  saliency scale-mismatch finding) -- would need its own measurement pass
  if pursued.
- **Redundant recompute across continuous-thought steps**: each CT step calls
  `run_stack` on the whole sequence so far but consumes only the last
  position's hidden state, and the final pass then recomputes the prefix a
  third time. A differentiable KV cache would collapse stage-`k` cost from
  roughly `k+1` full-sequence passes to ~1. Bounded upside (~2x at stage 1,
  which is where most of this data sits) and MLX's `KVCache` mutates
  preallocated buffers, so whether gradients survive it needs checking before
  committing -- not attempted yet.
- **Curriculum scope**: after the eligibility/span fixes, ~3 stages are
  supported (76% of trajectories have >=2 segments, 30% have >=3). Going deeper
  than that would need a different notion of compression candidate -- e.g.
  splitting long single segments -- rather than more data.
- **Capacity**: full fine-tuning is not available at this scale (86.1 GB for
  weights + grads + Adam states before activations, vs ~64 GB of activations at
  3072 tokens on a 128 GB machine). LoRA rank is the cheap dial instead --
  rank 128 is 524.6M trainable params, 16x rank 8, for ~3 GB of optimizer state
  and no measurable time cost. `--unfreeze-first-layers N` (~1.8 GB/layer) is
  implemented but unused; it targets the early layers, where the mechanism's
  actual distribution shift lands, and is the next lever if stage 1+
  underperforms.
