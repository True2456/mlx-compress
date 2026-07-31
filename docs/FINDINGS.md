# Step-3.7-Flash REAP: Measured Findings

**Model:** Step-3.7-Flash — StepFun 198B VLM (196B language backbone + 1.8B ViT),
45 decoder layers (dense 0–2, **MoE 3–44 = 42 layers**), **288 experts/layer**,
top-k=8, 1 shared expert, sigmoid router with `e_score_correction_bias`, SwiGLU,
RMSNorm. 375 GB BF16 on disk. Experts = **97% of parameters**.

**Hardware:** M5 Max, 128 GiB unified memory (`iogpu.wired_limit_mb` = 115 GiB).

**Scope:** Everything below is *measured*, not asserted. Where a conclusion was
later overturned by better data, both the wrong and corrected versions are shown —
several "obvious" fixes measured *worse* than what they replaced.

---

## TL;DR — the three measured effects, ranked

| # | Finding | Magnitude | Status |
|---|---|---|---|
| 1 | **Vision blindness** — text-only saliency can't see vision experts | **9.42%** of vision saliency mass sits on pruned experts; verdict **SEPARATE** (ρ=0.381) | ⚠️ **unaddressed** |
| 2 | **Layer-adaptive allocation** — uniform per-layer keep counts are suboptimal | **5.3%** less discarded mass at p15, free from existing data | available, unbuilt |
| 3 | **Truncation fix** — 384-token head truncation dropped answers | **1.7%** of experts changed in the final plan | ✅ done, minor |

**The vision issue is the largest open problem by a wide margin.**

> **See also `TOKENIZER-INVESTIGATION.md`** — numeric corruption in serving
> was a **real bug, found and fixed** (2026-07-26): `tokenizer_config.json`
> declares `tokenizer_class="LlamaTokenizerFast"`, which makes
> `AutoTokenizer` (what LM Studio actually calls) discard this model's real
> pretokenizer for Llama's own SentencePiece/Metaspace scheme. Two keys
> deleted, no weight change; verified 0/12 → 5/5 on both canonical failing
> cases via live generation. **PPL/NLL numbers below are unaffected** —
> `eval_ppl_streamed.py` goes through `transformers`, a different code path
> than the one that broke serving, so every run here used correct
> tokenisation regardless.

---

## 1. The reap decision: 15% (keep 245/288)

Discarded router-saliency mass by rung (5k prompts; 1k and 5k agreed to within
0.05%, so **saliency stabilises by ~1k prompts**):

| Reap | Keep/288 | Discarded mass (mean / worst layer) | 4-bit size |
|---|---|---|---|
| 10% | 260 | 3.0% / 4.0% | ~89 GB |
| **15%** | **245** | **5.0% / 6.8%** | **~84 GB** |
| 20% | 231 | 7.1% / 9.6% | ~80 GB |
| 25% | 216 | 9.6% / 12.9% | ~75 GB |

**This model resists pruning.** Every expert fires (zero dead experts), saliency is
flat, and prune sets are layer-local (Jaccard ~0.12 across depth — only ~4 experts
prunable everywhere). Unlike models where 25–50% comes free, capacity here is real.

Rationale for 15%: it's the only rung with empirical validation (a prior 158B ≈ 15%
build worked well); cost accelerates past it; and **reaping is not the size lever** —
quantization is. 15%→20% saves ~4 GB while nearly doubling worst-layer loss.

Raw REAP scores grow ~140× with depth (mean 0.17 at layer 3 → 24.0 at layer 44), so
**scores are not comparable across layers** without normalization.

---

## 2. Vision blindness ⚠️ — the biggest open issue

### The mechanism (verified in source)
Image features are `_masked_scatter`'d into `inputs_embeds` (`step3p7.py:105-111`)
and flow through the **same 42 MoE layers** being pruned. There is no separate
"vision MoE" — image tokens are routed by the same routers to the same experts.

But the collector builds embeddings via `text.embed_tokens(tokens)` only
(`collect_step3p7.py:312`) — it never calls `vision_model` or
`get_input_embeddings`. **So experts specialising in image tokens receive zero
activation during saliency collection, score ~0, and are pruned first.**

Net effect: the pipeline carefully preserves the vision *encoder* at BF16 while
deleting the experts that *consume its output*.

### The experiment
A dedicated vision-only saliency pass (`reap_stream/collect_vision_saliency.py`)
routed **300 images through the real vision path** — processor → `pixel_values` →
`get_input_embeddings` → merged embeds → the same block-streaming loop. Calibration
was deliberately **mixed: 200 ChartQA (synthetic plots) + 100 VQAv2 (natural
photos)**, because charts alone are a narrow slice of "vision" and would risk a
falsely reassuring result. Runtime: 1302 s.

Verdict thresholds were **pre-committed** before seeing data (MIXED if ρ>0.75 and
overlap>0.6; SEPARATE if ρ<0.4 or overlap<0.35).

### Result: SEPARATE

| Metric | Value |
|---|---|
| Mean rank correlation (vision vs text) | **0.381** |
| Mean top-25% expert overlap | **0.478** |
| Vision-top experts inside p15 prune set | **3.1 per layer** (max 9) |
| **Vision saliency mass on pruned experts** | **9.42%** (max **13.40%**) |

Correlation is lowest in early layers (~0.16–0.33, layers 3–14) and rises to
~0.36–0.54 through mid and late layers. The effect is **roughly uniform across
depth** — it is not concentrated anywhere in particular.

> **Correction — an error in earlier analysis.** An initial run of this comparison
> used the *partial* checkpoint, where layers 36–44 had `reap_count = 0`
> (not yet collected). Correlating against all-zero text saliency produced
> meaningless near-zero/negative values (e.g. layer 37: −0.145, layer 40: −0.115),
> which were mistakenly reported as a real "late-layer collapse" and called the most
> interesting finding. **On complete data those layers are normal** (layer 37: 0.457,
> layer 40: 0.435, layer 44: 0.537). There is no late-layer collapse. The headline
> verdict (SEPARATE) and the 9.4% mass figure were unaffected — both were robust
> across partial and complete data.

### Options (neither built)
1. **Protect vision-heavy experts** — union the top vision-saliency experts per layer
   into the keep set before applying the ratio cut. Cheap; slightly raises expert count.
2. **Combine explicitly via Contrastive Skill-Shield** (Theorem 6 in
   `frontier-moe-pruning-theorems.md`), treating vision as the target domain:
   `S_CS(e) = S̃_text(e) + α·max(0, S̃_vision(e) − S̃_text(e))`.
   **z-score normalization is required** — vision came from 300 images vs text's 2500
   prompts, so raw magnitudes aren't comparable.

Keeping the two saliency maps **separate and combining explicitly** is preferable to
mixing vision into the calibration set: mixing makes the blend ratio an implicit
hyperparameter and destroys the ability to see which modality wanted which expert.

---

## 3. Calibration data defects (both real, both fixed)

### 3a. Truncation dropped the answer
The collector did `tokenizer.encode(text)[:384]`. Share of prompts whose
`ASSISTANT:` response starts **beyond** the token budget:

| Category | beyond 384 | beyond 1024 |
|---|---|---|
| reasoning_math | 0.4% | ~0% |
| coding | 8.8% | 2% |
| tool_use | 41% | 1% |
| general_instruction | **96.3%** | 35% |
| **agentic** | **96.4%** | **64%** |

Token distribution: **median 798, p90 5195, p99 6380, max 10,182.** Agentic prompts
(system prompt + tool schemas + file contents) are enormous — no front-truncation
budget reaches their answers.

**Fidelity measured against full-length ground truth** (not assumed):

| Window / mode | Jaccard vs full | Spearman vs full |
|---|---|---|
| 384 head (original) | 0.783 | 0.938 |
| 1024 head | 0.834 | 0.960 |
| 1024 **tail** | **0.694** ← worst | 0.917 |
| **1024 headtail** | **0.858** ← best | **0.977** |

> **`tail` — the initially proposed fix — measured WORST.** Most routing statistics
> come from processing the long context, not the short answer; tail truncation
> discards that context. Tested before shipping; the intuition was wrong.

`_truncate()` now supports `head|tail|headtail` via `--truncation`.

### 3b. Multimodal rows were misaligned
`build_calib_mix.py:186` paired local PNGs (sorted by filename) against a **freshly
re-loaded** HF ChartQA dataset **by raw index**, silently desyncing image and Q&A.
Verified on two samples (an antiretroviral-therapy chart paired with a "positive
view peak" question; a Trump-approval chart paired with an online-classes question).
All 500 multimodal rows suspect.

- Excluded from DWQ via `artifacts/dwq-targets/exclude_indices.json` (105 of 1500).
- Fixed properly by `scripts/build_multimodal_calib_fixed.py` /
  `build_multimodal_calib_mixed.py`, which read image + query + answer from the
  **same dataset row in one pass** so they cannot desync.

**Note:** the misalignment was irrelevant to *REAP saliency* (no images were ever
fed), but genuinely harmful to *DWQ*, where the text-only teacher was asked to
predict chart answers it could not see.

---

## 4. Result of the corrected saliency re-run

Re-ran with `--max-tokens 1024 --truncation headtail --max-samples 2500`
(1.40 h, 42 layers, `nested_ok: true`) → `artifacts/step37-1024-headtail/`.

**Plan diff, new p15 vs original p15:**

| Metric | Value |
|---|---|
| Experts changed per layer | **4.3 of 245 (1.7%)** |
| Range | 2–10 |
| Total slots changed | 180 across 42 layers |
| Prune-set Jaccard (old vs new) | **0.821** |

**~98% of the pruning decision was identical.** The truncation flaw was real and
correctly diagnosed, but its downstream effect on the plan is small — expert ranking
is robust to this input-distribution shift. Changes cluster in early layers
(layer 3: 9 changed) and are smallest late (layers 36–44: ~3), consistent with early
layers being more sensitive to surface/formatting features.

Use the new plan going forward (it is more faithful and already paid for), but it is
**not** a meaningful quality upgrade on its own.

---

## 5. Layer-adaptive expert allocation (measured, unbuilt)

Current plans keep a **uniform 245 experts in every layer**. Allocating the same
total budget non-uniformly — greedily, by *fractional* mass cost per layer since raw
scores aren't cross-comparable — measurably reduces total discarded saliency:

| Rung | Uniform loss | Adaptive loss | Improvement | Adaptive keep range |
|---|---|---|---|---|
| p15 | 2.1107 | 1.9996 | **5.3%** | 190–276 (mean 245) |
| p25 | 4.0473 | 3.9442 | 2.5% | 150–262 (mean 216) |

Computed from **existing saliency** — no new collection needed. Caveats: discarded
mass is a proxy, not a quality measurement; and adaptive allocation **breaks the
nesting property** (p10 ⊂ p15 ⊂ p20 ⊂ p25) that the launcher currently verifies.

---

## 6. Memory: the biggest operational lesson

### MLX under-reports GPU memory by >2×
The collector logged `peak_mb = 46506` (~46 GB) while the process actually held
**109–110 GB**:

```
footprint <pid>
  109 GB   IOAccelerator (graphics)   [all DIRTY, 0 reclaimable]
  phys_footprint: 110 GB
```

**Always use `footprint <pid>`, never `mx.get_peak_memory()`.** Trusting MLX's number
allowed two crashes to happen unseen. (Note: `pgrep -f run_mac_bf16` also matches the
zsh wrapper — filter for the real Python PID.)

### Failure modes observed
1. **OS-level hard reboot** — the whole machine went down mid-run.
2. **GPU watchdog timeout** — `[METAL] Command buffer execution failed: Caused GPU
   Timeout Error`. Memory pressure stalls GPU ops until macOS kills the buffer.

**Leading indicator for both: `vm_stat` compressor > 40 GB and swap expanding.**
Healthy = compressor ≈ 0. Do *not* watch "free RAM" — macOS always reports ~0.

### Root cause and fix
Not `layers_at_once`, not prompt count. **MLX's allocator hoards freed buffers**, and
`mx.clear_cache()` was only called once per *layer* — so 2500 per-prompt forwards
accumulated every freed intermediate.

```python
for i in range(len(hidden)):
    hidden[i] = _run_layer(layer, hidden[i], sliding_window)
    if (i + 1) % _CACHE_EVERY == 0:      # _CACHE_EVERY = 200
        mx.clear_cache()
```

| | Before | After |
|---|---|---|
| IOAccelerator | **109 GB** | **34 GB** |
| Compressor | 70 GB | **0.1 GB** |
| Swap used | 9.6 GB | 553 MB |

**~3× reduction at zero cost** to sample count, layers, or plan quality. Also fixed:
`hidden = [_run_layer(...) for h in hidden]` materialised an entire second ~21 GB list
before releasing the old one — now updates in place.

With this fix, the text (≈48 GB) and vision (≈14 GB) passes ran **concurrently** with
compressor at only 1.0 GB.

### Sizing reference
- MoE layer weights: **9.06 GB each** (288 × 3 proj × 4096×1280 × BF16)
- Hidden states: `n_prompts × max_tokens × 4096 × 2 bytes` (2500×1024 ⇒ ~21 GB)
- `iogpu.wired_limit_mb` is already 115 GiB; **raising it further starves the OS** and
  causes the hard-reboot failure mode.

---

## 7. DWQ: built, works, value unproven

Pipeline (`mlx_vlm`-native, since `mlx_lm` only knows `step3p5`, not `step3p7`):
- **Phase 1** `dwq_collect_targets.py` — streams the BF16 teacher, caches **top-128**
  logits per prompt (377 MB for 1500 prompts, ~32 min). Full-vocab would be ~495 GB
  and is unnecessary: LM distributions are peaked, top-128 captures >99.9% of mass.
- **Phase 2** `dwq_train_student.py` — 92 GB student resident, trains only affine
  quant scales, checkpoints the small (~5.3 GB) trainable tensors with auto-resume.

### Blockers hit (all real, all fixed)
1. **`GatherQMM::vjp` error** — MoE routing indices come from `argpartition`
   (non-differentiable) but the quantized gather-matmul VJP tried to differentiate
   them. Fix: `mx.stop_gradient(indices)` at the `SwitchGLU` boundary.
2. **Recompilation every step** — variable prompt lengths → new graph each step
   (45–80 s/step). Fix: pad all to `max_tokens`, mask the loss.
3. **Adam optimizer state OOM** — 5.3 B trainable scale params × 2 momentum buffers
   ≈ +59 GB. Fix: `--scales-only` + `--optimizer sgd` → stable 109 GB, ~45 s/step.
4. **Gradient accumulation made things WORSE** — every variant (fp32, +gc/clear_cache,
   fp16) *raised* memory (115–130 GB) and tripled step time. Abandoned.
5. **Divergence at lr=1e-4** — loss climbed 2–5× above the untrained baseline and did
   not recover. Diagnosed with `diag_specific_kl.py` (trained loss vs untouched-student
   baseline on the *same* prompts). Fixed with lr=1e-6, then 1e-7.

### Why its value is questionable
The **untrained** student's baseline KL vs the teacher was already **0.06–1.4** — low.
There is little gap left for DWQ to close, and it was never shown to beat noise.
Effort is better spent on calibration quality (§2, §3) than DWQ tuning.

---

## 8. Quantization format

| Format | scales | biases | DWQ-trainable? | group_size |
|---|---|---|---|---|
| **affine 4-bit** | float32 | float32 | ✅ yes | 64 |
| **nvfp4** | **uint8** (FP8 microscales) | none | ❌ no | **must be 16** |

- nvfp4 is the better raw 4-bit format but stock DWQ cannot train its uint8 scales.
- **nvfp4 requires no calibration data at all**, which given §3 is a real advantage.
- Vision tower left **unquantized (BF16)** — the deploy model keeps full vision.
- LM Studio runs `step3p7` + affine; **DWQ is invisible to the runtime**.

Built student: `models/Step-3.7-p15-4bit` — 92 GB, 245 experts, **4.632 bpw**,
via `scripts/build_student.py` (fused apply+quantize, never writes the ~316 GB
reaped-BF16 intermediate). Smoke-tested: loads, generates coherently, 58 tok/s.
**Coherent with zero recovery**, corroborating the 15% choice.

### 8a. Requantizing from an already-quantized source is strictly lossy

Measured accidentally, then kept — an 8-bit `lm_head`/`embed_tokens` built by
**dequantizing the student's existing 4-bit head and requantizing to 8-bit**,
against the same shared8 student with its head left at 4-bit. Identical weights
elsewhere (18 of 20 shards hardlinked), 500 held-out prompts:

| category | 4-bit head | 8-bit head **from 4-bit source** | ΔNLL |
|---|---|---|---|
| agentic | 6.921 | 7.075 | +0.022 |
| coding | 6.488 | 6.545 | +0.009 |
| general_instruction | 5.115 | 5.194 | +0.016 |
| reasoning_math | 2.4848 | 2.4846 | −0.000 |
| tool_use | 27.18 | 27.81 | +0.023 |
| **OVERALL** | **5.930** | **6.010** | **+0.013** |

**Doubling the bit-width made every category worse** and cost +0.53 GB.
Quantization is lossy and one-way: the 4-bit rounding is already baked in, so
the wider container stores degraded values and the second rounding pass adds
fresh error on top. `artifacts/ppl-head8-requant-from-4bit-500.json`.

**Rule:** always quantize from the highest-precision source available. This
applies directly to *converting* a released quantized checkpoint into another
format — e.g. an existing 4-bit MLX repo re-encoded to nvfp4 — which inherits
the original's loss and adds its own. A conversion is not a requantization of
the original model; it is a quantization of an already-damaged one.

The result also functions as a sanity check on any precision experiment: more
bits cannot make a model better-than-source. A precision increase measuring
*worse* means the source, not the target, is wrong.

---

## 9. Router architecture (relevant to proposed "Router Bias Surgery")

```python
corrected_scores = scores + router_bias          # SELECTION only
topk_indices = argpartition(-corrected_scores, kth=top_k-1)[..., :top_k]
topk_weights = take_along_axis(scores, topk_indices)   # weights use UNBIASED scores
topk_weights = topk_weights / sum(topk_weights)        # norm_expert_weight: True
```

Three consequences:
1. **No routing void.** `argpartition` runs over the **245 survivors**, so every token
   still gets exactly 8 experts. A rank-9 kept expert is promoted automatically.
2. **Mass is already conserved** by `norm_expert_weight: True`.
3. **Relative ordering among survivors is unchanged** — each keeps its original bias.

`router_bias` is a DeepSeek-style aux-loss-free **load-balancing** term; load imbalance
costs training throughput, not inference quality (no capacity limits at inference).
Proposals to "absorb probability mass" via bias adjustment therefore address a problem
this architecture already handles.

---

## 10. Untested proposals — status

| Idea | Assessment |
|---|---|
| **REAM** (merging instead of pruning) | Genuinely well-matched: this model *is* prune-resistant, and REAM consumes REAP scores so existing saliency is reusable. Biggest build; unvalidated here. |
| **DPP-REAP** (diversity selection) | Math is correct. But the co-occurrence kernel is **~36× under-sampled off-diagonal** (2.8% vs 0.077% of tokens), so dividing by `total_tokens` makes it diagonally dominant → **degenerates back to REAP**. Fix: normalize by co-occurrence count. Naive greedy is also ~2.6×10¹¹ flops/layer; incremental Cholesky reduces it to ~1.7×10⁷. |
| **Router Bias Surgery** | Addresses a non-problem — see §9. |
| **RMSNorm residual projection saliency** | The strongest untested idea. RMSNorm's Jacobian contains `P⊥ₓ = I − xxᵀ/‖x‖²`, so expert output parallel to the residual stream is rescaled away and should not count toward saliency. Cheap, drop-in, independent. |
| **Layer-adaptive allocation** | Already measured: **5.3%** at p15 (§5). |

### Cheapest decisive next experiment
Add co-occurrence tracking to `LayerSaliency` and measure **off-diagonal mass** of the
correlation matrix. If `C_{i,j} ≈ 0`, experts are already orthogonal and **both DPP and
REAM are ruled out** for ~40 min of machine time.

**Note:** co-occurrence is inherently *pairwise*; everything currently stored
(`reap`, `freq`, `gate_sum`, `reap_count`) is a **per-expert marginal**. It cannot be
recovered from existing artifacts — a new pass is required. But **5k is unnecessary**:
at 1000 prompts each pair averages ~694 co-occurrences (threshold is 5), ample for a
yes/no redundancy gate. ~40 min vs ~3 h.

---

## 10b. Perplexity evaluation — two methodology bugs found

Building the first *non-proxy* measurement (`reap_stream/eval_ppl_streamed.py`)
surfaced two bugs that would have made every PPL number meaningless. Both were
caught by noticing an implausible result, not by the code failing.

Initial smoke test on 6 held-out prompts gave **agentic PPL = 388** — absurd for a
frontier model, while `reasoning_math` scored a healthy 3.30. That split was the clue.

| Category | Original | + `head` trunc | + raw text (**correct**) |
|---|---|---|---|
| agentic | 388.41 | 220.56 | **10.75** |
| coding | 42.30 | 42.30 | **9.13** |
| reasoning_math | 3.30 | 3.17 | **2.55** |
| **OVERALL** | 42.94 | 33.22 | **6.13** |

**Bug 1 — `headtail` truncation is wrong for perplexity.**
It's correct for *saliency* (sampling both task setup and answer gives representative
routing statistics), but for perplexity it splices first-512 onto last-512, creating a
hard discontinuity mid-sequence. Tokens after that seam are genuinely unpredictable,
inflating NLL for reasons unrelated to model quality. Agentic prompts (5000–10000
tokens) are the most truncated, hence worst affected; `reasoning_math` mostly fits
under 1024 and was untouched. **PPL eval now defaults to `head`** (contiguous prefix).

**Bug 2 — chat-template double-wrapping** (inherited from `_tokenize_prompts`).
The calib rows already carry their own `SYSTEM:/USER:/ASSISTANT:` structure, but
`apply_chat_template` wrapped the whole thing *again* as a single user turn:
```
<|begin_of_sentence|><|im_start|>user\nSYSTEM:\nYou are a helpful assistant...
```
The model receives `SYSTEM:` as literal text inside a user message — a structure it
never saw in training. This is why agentic (elaborate embedded SYSTEM blocks + function
schemas) and coding suffered most, while plain `USER:/ASSISTANT:` reasoning_math barely
moved. Fixing it gave a **36× improvement on agentic**. PPL eval now feeds raw text;
`--chat-template` opts back in.

> **Does this invalidate the saliency work? No.** Every saliency run used identical
> treatment, so comparisons *between* them (old vs new plan, text vs vision) remain
> valid — it's a level shift, not a differential one. It would only have corrupted
> perplexity, which is exactly where it was caught.

**Eval protocol:** 500 held-out prompts from rows 5000+ (never used in calibration —
saliency used the first 2500, DWQ the first 1500), multimodal excluded (misaligned and
imageless, unanswerable as text), `head` truncation at 1024 tokens, raw text, with
**per-category breakdown** — an aggregate number can hide "coding fine, agentic
degraded 5%".

Measured BF16 streaming cost: **0.86 s fixed/prompt + 1.13 ms/token** → ~17 min for
500 prompts. Resident quantized models are far faster (~1–2 min, no per-layer disk reads).

---

## 11. Method lessons

- **`footprint <pid>`, not MLX counters.** MLX under-reported by >2×.
- **Compressor + swap are the crash predictors**, not free RAM.
- **MLX hoards freed buffers** — `mx.clear_cache()` inside hot loops, not just at phase
  boundaries. Single highest-leverage memory fix found.
- **List comprehensions over big tensors double peak memory** — update in place.
- **Never compare against a partial checkpoint.** Zero-filled layers silently produce
  meaningless correlations that look like real structure (§2 correction).
- **Pair multimodal data in one pass.** Index-based pairing across separate loads
  desyncs silently.
- **Test the fix, not just the problem.** `tail` truncation and gradient accumulation
  both sounded right and both measured *worse*. Every fix that survived here was
  validated against ground truth first.
- **An implausible number is a bug report.** Agentic PPL of 388 wasn't a finding about
  the model — it was two stacked methodology bugs (§10b). Sanity-check magnitudes
  against what the quantity *should* look like before interpreting it.
- **The right setting is task-dependent.** `headtail` is correct for saliency and wrong
  for perplexity; the same knob flips depending on whether you're sampling routing
  statistics or modelling a contiguous sequence.
- **Most numbers here are still proxies.** Perplexity (§10b) is the first non-proxy
  measurement. Real task benchmarks (SWE-bench, Terminal-Bench — what this model is
  actually built for) remain unrun.
