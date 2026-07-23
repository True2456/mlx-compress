# Step-3.7-Flash: REAP → 4-bit → DWQ — Full Session Summary

**Goal:** a smaller, high-quality local-serving Step-3.7-Flash (`step3p7`, a VLM MoE)
for LM Studio on an M5 Max, via expert pruning (REAP) + quantization, optionally
recovered with distilled quantization (DWQ).

**Hardware:** M5 Max **128 GiB** + M3 Max 64 GiB over Thunderbolt 3.

> ⚠️ **Read §11 first if you are about to run anything.** Several conclusions in
> the first half of this session were later **overturned by measurement**. The
> corrections are more valuable than the original findings.

---

## 1. The model

| Property | Value |
|---|---|
| Text hidden size | 4096 |
| Decoder layers | 45 (dense 0–2, **MoE 3–44 = 42 layers**) |
| Experts / MoE layer | **288**, top-k 8, sigmoid router |
| MoE intermediate | 1280 (+ shared expert 1280) |
| Vocab | 128,896 |
| Full BF16 on disk | **375 GB** |
| Text params | ~185.5 B; **experts = 97% of params** |
| **Weights per MoE layer** | **~9.06 GB** (288 × 3 proj × 4096×1280 × BF16) |

---

## 2. REAP: saliency and the reap decision

REAP saliency = mean of `gate_weight × ‖expert_activation‖` over routed tokens.
Collected via a layer-wise streaming collector (one window of blocks resident at
a time, hidden states carried across windows).

**Discarded router-saliency mass per rung** (5k prompts; 1k and 5k agreed to
within 0.05%, i.e. **saliency stabilises by ~1k prompts**):

| Reap | Keep/288 | Discarded mass (mean / worst layer) | 4-bit size |
|---|---|---|---|
| 10% | 260 | 3.0% / 4.0% | ~89 GB |
| **15%** | **245** | **5.0% / 6.8%** | **~84 GB** |
| 20% | 231 | 7.1% / 9.6% | ~80 GB |
| 25% | 216 | 9.6% / 12.9% | ~75 GB |

**This model resists pruning:** every expert fires (zero dead), prune sets are
layer-local (Jaccard ~0.12 across depth), no universal dead weight.

### Decision: 15% reap (keep 245/288)
- Only rung with empirical validation (a prior 158B ≈ 15% build "worked quite well").
- Cost accelerates past 15%.
- **Not reaping for size** — quantization owns that axis; every rung fits the M5.
  15%→20% saves ~4 GB but nearly doubles worst-layer loss.

---

## 3. Quant format decision

Inspected the actual quantized-layer internals:

| Format | scales | biases | DWQ-trainable? | group_size |
|---|---|---|---|---|
| **affine 4-bit** | float32 | float32 | ✅ yes | 64 |
| **nvfp4** | **uint8** (FP8 microscales) | none | ❌ no | **must be 16** |

- nvfp4 is the better *raw* 4-bit format, but stock DWQ can't train its uint8
  scales (needs a custom float-shadow + STE port).
- **Vision tower left unquantized (BF16)** in the student — the deploy model keeps
  full vision; DWQ only refines the text path.
- **LM Studio runs step3p7 + affine**, and **DWQ is invisible to the runtime**
  (it's a training method; output is a normal quantized model).
- **nvfp4 needs no calibration data at all** — which, given §6, is a real advantage.

---

## 4. Artifacts

| Artifact | Path | Notes |
|---|---|---|
| BF16 base | `models/Step-3.7-Flash` | 375 GB; the DWQ teacher (streamed) |
| **Student** | `models/Step-3.7-p15-4bit` | **92 GB**, 245 experts, 4.632 bpw, vision BF16 |
| 6-bit teacher | *(deleted)* | 153 GB; built then found unnecessary — see §5 |

`scripts/build_student.py` does a **fused apply+quantize**: slices experts (p15)
and quantizes in one pass, writing only the 92 GB output and never the ~316 GB
reaped-BF16 intermediate. Smoke-tested: loads, generates coherently, 58 tok/s.
**Already coherent with zero recovery** — corroborates the 15% decision.

---

## 5. Architecture findings

- **`mlx_lm` only knows `step3p5`, not `step3p7`** → stock `mlx_lm.quant.dwq`
  cannot load these checkpoints. All DWQ had to be ported onto `mlx_vlm`.
- **`step3p7`'s `StepTextModel` does NOT inherit `PipelineMixin`** → the 2-Mac
  resident pipeline would need model surgery (mixin + inter-rank send/recv).
- **Avoided entirely by streaming**: the existing collector already streams a full
  forward block-by-block on one Mac. So teacher logits are computed by **streaming
  the BF16 teacher** — better than the 6-bit teacher *and* no pipeline needed.
- **Consequence: the 6-bit teacher build + M3 transfer were unnecessary.** Built,
  transferred, then deleted to reclaim 153 GB when disk filled.
- Vision path (for future multimodal DWQ): `Step3VLProcessor(text=…, images=[img])`
  → `input_ids/pixel_values/patch_pixel_values/num_patches` →
  `model.get_input_embeddings(...)` → merged embeds feed the normal decoder stream.
  Text needs exactly one literal `<im_patch>` placeholder per image.

---

## 6. ⚠️ Calibration data problems (found late, both real)

### 6a. Multimodal rows were misaligned
`build_calib_mix.py:186` paired local PNGs (sorted by filename) with a **freshly
re-loaded** HF ChartQA dataset **by raw index** — silently desyncing image and Q&A.
Verified on 2 samples: a chart about antiretroviral therapy paired with a question
about "positive view" peaks; a Trump-approval chart paired with an online-classes
question. **All 500 multimodal rows are suspect.**

- Excluded from DWQ via `artifacts/dwq-targets/exclude_indices.json` (105 of 1500).
- **Fixed properly** by `scripts/build_multimodal_calib_fixed.py` — reads image +
  query + answer from the **same dataset row in one pass**, so they cannot desync.
  300 correctly-paired rows at `calib/multimodal_fixed/` (spot-verified against the
  source image).

### 6b. Truncation cut off the ASSISTANT answer — MEASURED, not theorised
The collector did `tokenizer.encode(text)[:max_tokens]` with `max_tokens=384`.
Measured share of prompts whose `ASSISTANT:` starts **beyond** the budget:

| Category | beyond 384 | beyond 1024 |
|---|---|---|
| reasoning_math | 0.4% | ~0% |
| coding | 8.8% | 2% |
| tool_use | 41% | 1% |
| general_instruction | **96.3%** | 35% |
| **agentic** | **96.4%** | **64%** |

Token distribution: **median 798, p90 5195, p99 6380, max 10,182.** Agentic prompts
(system prompt + tool schemas + file contents) are enormous — no feasible front
truncation reaches their answers.

**Impact, measured against full-length ground truth** (not asserted):

| Window / mode | Jaccard vs full | Spearman vs full |
|---|---|---|
| 384 (original) | 0.783 | 0.938 |
| 1024 head | 0.834 | 0.960 |
| 1024 **tail** | **0.694** | 0.917 |
| **1024 headtail** | **0.858** | **0.977** |

**Verdict:** 384 was ~78% faithful — imperfect but not garbage (so prior "the data
is fine" reviews weren't wrong about *content*; this is a pipeline setting, not a
dataset defect). **`headtail` wins** — and note **`tail` (my initial proposed fix)
was the WORST option**, because most routing statistics come from processing the
long context, not the short answer. Tested before shipping; the intuition was wrong.

`_truncate()` in `collect_step3p7.py` now supports `head|tail|headtail`, exposed as
`--truncation` on the launcher.

---

## 7. DWQ pipeline (Path B — `mlx_vlm`-native)

DWQ is **self-supervised**: you supply *input prompts only*; the teacher's logits
are the labels. Same prompts used in both phases.

**Phase 1 — `reap_stream/dwq_collect_targets.py`** (M5, ~32 min for 1500 prompts):
streams the BF16 teacher block-by-block, applies final norm + `lm_head`, keeps
**top-128** logits per position → one safetensors per prompt (`input_ids`,
`topk_vals`, `topk_idx`) in `artifacts/dwq-targets/` (377 MB).
Top-128 vs full-vocab: full would be ~495 GB and is pointless — LM distributions
are peaked and top-128 captures >99.9% of mass. Saving `input_ids` guarantees
phase 2 feeds identical sequences.

**Phase 2 — `reap_stream/dwq_train_student.py`** (M5): loads the 92 GB student
resident, unfreezes affine quant scales, KL over cached top-k, checkpoints the
small (~5.3 GB) trainable-scale tensors every N steps with auto-resume.

### Phase-2 blockers hit (all real, all fixed)
1. **`GatherQMM::vjp` error** — MoE routing indices come from `argpartition`
   (non-differentiable) but the quantized gather-matmul VJP tried to differentiate
   them. Fix: monkeypatch `SwitchGLU.__call__` to `mx.stop_gradient(indices)`.
2. **Recompilation per step** — variable prompt lengths → new graph each step
   (45–80s/step). Fix: pad all to `max_tokens`, mask the loss. One shape, one compile.
3. **Adam optimizer state OOM** — 5.3 B trainable scale params × 2 momentum buffers
   ≈ +59 GB, past the wired limit → swap thrash (80s/step). Fix: `--scales-only`
   (drops biases, halves to 2.6 B) + `--optimizer sgd` (no momentum) → stable 109 GB,
   ~45s/step.
4. **Gradient accumulation made things WORSE** — tried to fix optimizer noise; every
   variant (fp32 accum, +gc/clear_cache, fp16 accum) *raised* memory (115–130 GB) and
   tripled step time. Abandoned. **Measured, not assumed.**
5. **Divergence at lr=1e-4** — loss climbed 2–5× above the untrained baseline and did
   not recover. Diagnosed by comparing against baseline KL on the *same* prompts
   (`reap_stream/diag_specific_kl.py`). Fixed by `lr=1e-6`, later `1e-7`.

### The uncomfortable truth about DWQ's value here
The **untrained** student's baseline KL vs the teacher was already **0.06–1.4** —
low. That's good news for the reap+quant choices, but it means **there is little gap
left for DWQ to close**, and we never demonstrated it moving the needle beyond noise.
Combined with §6, effort is better spent on calibration quality than DWQ tuning.

---

## 8. ⚠️ Memory: the biggest operational lesson

**MLX's own counters under-report GPU memory by >2×.** The collector logged
`peak_mb=46506` (~46 GB) while the process was actually holding **109–110 GB**:

```
footprint <pid>
  109 GB   IOAccelerator (graphics)   [all DIRTY, 0 reclaimable]
  phys_footprint: 110 GB
```

**Always use `footprint <pid>`, never `mx.get_peak_memory()`, for real usage.**
Trusting MLX's number is what let two crashes blindside us.

### Failure modes observed
1. **OS-level hard reboot** — machine went down entirely mid-run.
2. **GPU watchdog timeout** — `[METAL] Command buffer execution failed: Caused GPU
   Timeout Error`. Memory pressure stalls GPU ops until macOS kills the buffer.

Leading indicator for both: **compressor >40 GB and swap expanding** (`vm_stat`,
`sysctl vm.swapusage`). Healthy = compressor near 0. Watch that, not free RAM
(macOS always reports ~0 free).

### Root cause and the fix that worked
Not `layers_at_once`, not prompt count. **MLX's allocator hoards freed buffers**
and `mx.clear_cache()` was only called once per *layer* — so 2500 per-prompt
forwards accumulated every freed intermediate.

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

**~3× reduction at zero cost** to sample count, layers, or plan quality.

Also fixed: `hidden = [_run_layer(...) for h in hidden]` materialised an entire
second ~21 GB list before releasing the old one. Now updates in place.

### Sizing reference
- `iogpu.wired_limit_mb` is already **117760 = 115 GiB**; raising it further is
  counterproductive (starves the OS → the hard-reboot condition).
- Per MoE layer weights: **9.06 GB**; `layers_at_once=2` ⇒ ~18 GB resident.
- Hidden states: `n_prompts × max_tokens × 4096 × 2 bytes` (2500×1024 ⇒ ~21 GB).

---

## 9. Current state

- ✅ Reap decided (15%); student built + smoke-tested coherent.
- ✅ Truncation fix implemented (`headtail`) and **validated against ground truth**.
- ✅ Multimodal misalignment fixed (`calib/multimodal_fixed/`, 300 clean pairs).
- ✅ Memory root-caused and fixed (109 → 34 GB).
- 🟡 **Corrected saliency run in progress**: 2500 prompts, 1024 tokens, headtail →
  `artifacts/step37-1024-headtail/`. Resumable from `checkpoints/state.json`.
- ⏸ DWQ paused — value in question (§7), and it would need rebuilding against the
  new plan anyway.

### Next steps
1. Finish the corrected saliency run → new p10/p15/p20/p25 plans.
2. **Diff new p15 vs old p15** — per-layer expert churn quantifies whether the
   truncation fix actually changed the reap decision.
3. Build the student from whichever plan wins; **nvfp4** (`--mode nvfp4
   --group-size 16`) is the recommended no-calibration-risk deploy format.
4. Only revisit DWQ / dynamic-quant if a real eval shows a gap worth closing.
5. Ground truth for *any* of this is a downstream eval on real coding/agentic/tool
   tasks — every metric above is a proxy.

---

## 10. Reusable learnings

- **`footprint <pid>`, not MLX counters.** MLX under-reported by >2×.
- **Compressor + swap are the crash predictors**, not free RAM.
- **MLX hoards freed buffers** — call `mx.clear_cache()` inside hot loops, not just
  at phase boundaries. Single highest-leverage memory fix found.
- **List comprehensions over big tensors double peak memory** — update in place.
- **Stream, don't pipeline**, when the model already has a streaming forward.
- **Fuse apply+quantize** to skip huge intermediates.
- **REAP saliency stabilises by ~1k prompts** — more samples buy ranking confidence,
  not correctness.
- **DWQ is self-supervised and runtime-invisible** — no labels, no custom engine.
- **nvfp4 scales are uint8** → not DWQ-trainable; affine is.
- **The arch, not the quant, gates LM Studio compatibility.**
- **Pair multimodal data in one pass.** Index-based pairing across separate loads
  desyncs silently.
- **Test the fix, not just the problem.** `tail` truncation and gradient accumulation
  both *sounded* right and both measured *worse*. Every fix here that survived was
  validated against ground truth first.
