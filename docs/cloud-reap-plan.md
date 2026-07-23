# Cloud REAP Plan (Step-3.7-Flash)

Budget: **~430 AUD Google Cloud credits** (~280 USD). Goal: REAP full BF16 Step with a **10% → 15% → 20% → 25%** gated ladder, HF-upload the last good checkpoint, Mac-side quant/eval. Skip kitchen-sink SFT on the first pass.

## Principles

1. **Saliency once → nested plans → progressive apply with a quality gate.**
2. **Ladder: 10% → 15% → 20% → 25%** expert prune. Continue only if the gate passes.
3. **Calib quality ≫ calib volume.** Target ~5–8k real traces, not 100k–800k SFT dumps.
4. **HF upload the last good checkpoint** — not every rung, and not four blind exports.
5. **Kill the GPU VM** once the winner is on HF (or GCS). Idle GPU is the credit killer.
6. **LoRA / imatrix are phase 2**, only if the best REAP still looks weak on Mac.

## Target model

| Item | Value |
| --- | --- |
| Base | `stepfun-ai/Step-3.7-Flash` (BF16, Hub ~**400 GB**) |
| MoE | layers ~3–44, **288** experts, top-8 (+ shared / vision / MTP dense — freeze those) |
| Prune ladder | **10% → 15% → 20% → 25%** experts removed per MoE layer (nested from one ranking) |
| Approx keeps | 10% → keep **260**; 15% → keep **245**; 20% → keep **231**; 25% → keep **216** (from 288; `int(288 * ratio)` prune) |

Do **not** brand these as “170B / 165B / 160B” until an apply reports real param count and on-disk size.

## Progressive prune protocol (locked)

```
saliency(base) once
  → plan_10, plan_15, plan_20, plan_25   # nested: prune set_10 ⊂ set_15 ⊂ set_20 ⊂ set_25

apply plan_10 from base
  → gate: held-out PPL vs base + tiny gen/tool smoke
  → FAIL: upload nothing (or upload k259 only if still useful); stop
  → PASS: continue

apply plan_15 from base
  → same gate vs base (and note Δ vs 10%)
  → FAIL: **HF-upload the 10% model**; stop
  → PASS: continue

apply plan_20 from base
  → same gate vs base
  → FAIL: **HF-upload the 15% model**; stop
  → PASS: continue

apply plan_25 from base
  → same gate
  → FAIL: **HF-upload the 20% model**
  → PASS: **HF-upload the 25% model**

delete intermediate weight trees on the VM after the winner is uploaded
```

### Gate (cheap, on-VM)

| Check | Rule of thumb |
| --- | --- |
| Held-out PPL | Stop if ΔPPL vs base is large/unstable (calibrate threshold on first 10% run; don’t worship a fixed %) |
| Smoke gen | 20–50 prompts: coding + tools + short reasoning — reject if collapse / empty / nonsense |
| Do not | Full SWE-Bench / huge harness on the GPU clock |

**Re-run saliency between rungs? No** for this ladder. Only re-profile if 25% fails badly and you want a second adaptive pass later.

## Calibration mix (~6–8k rows)

Build a **single JSONL** of concatenated chat text (user + assistant + tools). Prefer **unique trajectories**, not every cumulative SFT prefix.

| Slice | Share | Sources |
| --- | --- | --- |
| Agentic coding / debug | ~35% | Fable-5 (1 row per trajectory, full messages) |
| Tool use | ~25% | Fable-5 tool subset + xlam-style / gold tool rows |
| Reasoning / math | ~15% | Gold reasoning + small public math/logic |
| General instruction | ~15% | Gold formatting + thin UltraFeedback sample (hundreds–low thousands, not 65k) |
| Multimodal (optional) | ~10% | **Small** ChartQA / WebSight sample only if vision experts matter; else 0% |

**Caps:** max tokens per row ~1k–2k mixed lengths (short/medium/long). No 820k WebSight in calib.

**Held-out:** 200–500 rows for smoke PPL / spot gen — not a full eval suite on the GPU clock.

## GCP layout

### Machines

| Role | Spec (starting point) | Notes |
| --- | --- | --- |
| **CPU Staging VM (Phase 1)** | `e2-standard-4` ($0.13/hr) + **1.5 TB SSD Disk** | Pre-downloads 400 GB BF16 base model for 3 cents; zero GPU idle cost |
| **REAP GPU Compute VM (Phase 2-4)** | **8×H100 80GB** (or 8x A100 80GB) | Attach pre-downloaded 1.5 TB disk; runs saliency + progressive apply immediately |
| **Disk** | **1.5 TB SSD** (`step37-disk`) | Holds Base (~400GB) + winner weights (~300GB) + scratch space |

If Spot preemption is painful, use on-demand for the saliency+apply GPU window only.

### Secrets

- HF token with **write** on a private namespace (e.g. `True2456/...`)
- Do not bake tokens into the image; use env / Secret Manager

## Pipeline phases

### Phase 0 — Prep (local, cheap)

1. Finish / freeze calib JSONL (~6–8k).
2. Confirm HF login + empty private repos or a single repo with revision tags.
3. Patch Cerebras REAP for Step-3.7 (same class of work as MiniMax) **before** burning GPU time.
4. Smoke the collector on a tiny MoE or a 1–2 layer dry run in cloud.

### Phase 1 — Download & Staging (CPU VM - $0.03 total cost)

1. Spin up cheap $0.13/hr CPU VM with 1.5 TB Persistent Disk (`step37-disk`).
2. Pull `stepfun-ai/Step-3.7-Flash` (400 GB BF16) directly to `step37-disk`.
3. Verify shard count / `model.safetensors.index.json`.
4. Delete CPU VM; keep `step37-disk`.
5. **Only then** spin up GPU VM and attach `step37-disk`. GPU starts saliency collection instantly!

### Phase 2 — Saliency (the expensive part that matters)

1. Run layerwise REAP collect on MoE layers only (dense 0–2, shared, MTP, vision frozen).
2. Save:
   - `saliency.json` (or equivalent per-layer scores)
   - `trace.json` / run metadata (tokens, samples, git SHA, seed)
3. From saliency, emit **nested plans** in seconds:
   - `plan_p10.json` — prune 10% (keep ~259)
   - `plan_p15.json` — prune 15% (keep ~245)
   - `plan_p20.json` — prune 20% (keep ~230)
   - `plan_p25.json` — prune 25% (keep ~216)

Upload plans + saliency to HF immediately (tiny).

### Phase 3 — Progressive apply (10 → 15 → 20 → 25)

1. Apply `plan_p10` from base → gate (PPL + smoke).
2. If pass: apply `plan_p15` → gate.
3. If pass: apply `plan_p20` → gate.
4. If pass: apply `plan_p25` → gate.
5. Keep **only the last passing checkpoint** on disk for upload; drop losers to free space.
6. Log a small table: ratio / keep / disk GB / val PPL / smoke pass.

Prefer re-applying each plan **from base** (clean, nested by construction). Nested apply on the previous ckpt is OK if I/O-cheaper and prune sets are verified subsets.

### Phase 4 — Publish winner + stop

1. `huggingface-cli upload` the **last good** pruned weights (private repo).
2. Upload plans + saliency + gate log to the same repo or `-artifacts`.
3. **Terminate GPU VM.** Keep PD only if you will resume within days.

### Phase 5 — Mac (your 128 GB)

1. `hf download` the winning REAP repo.
2. Convert / run via **mlx_vlm** (not plain `mlx_lm` for Step).
3. Quantize locally. Skip cloud imatrix unless you explicitly want llama.cpp GGUF later.
4. Eval: coding, tools, a bit of reasoning; compare qualitatively to your existing 148B REAP checkpoint.

### Phase 6 — Optional (only if Phase 5 is weak)

| Option | When | Notes |
| --- | --- | --- |
| Stop early at 10% / 15% / 20% | Gate failed at next rung | Already the default protocol |
| Short recovery LoRA/SFT | Clear routing/gen regressions on the winner | **≤5–10k** high-quality traces; target router + light attn |
| imatrix | Only if GGUF path chosen | Local/Mac or short quant-path job |
| Re-saliency + second ladder | 25% fails oddly | New collect on base or on 15% ckpt — new GPU window |

## Budget guardrails

| Do | Don’t |
| --- | --- |
| Spot / short GPU windows | Leave 8×H100 up overnight “just in case” |
| HF upload pruned weights | Download 3× BF16 to home |
| 6–8k calib | 820k WebSight for REAP |
| Nested JSON plans + gated applies | Blind export of all three weight trees |
| Upload last good only | Upload every rung “just in case” |
| Kill VM after upload | Train LoRA “while you’re there” by default |

Rough expectation: **tens to low hundreds of USD** for a careful phase 1–4, not ~$15. Leave headroom for Spot retries.

## Success criteria

- [ ] Saliency + `plan_p10/p15/p20/p25` on HF
- [ ] Gate log for each rung attempted
- [ ] **One** winning pruned Step checkpoint on HF with known size
- [ ] Mac load + smoke gen works
- [ ] GPU VM at $0 after publish
- [ ] Credits left for a retry or small LoRA if needed

## Explicit non-goals (v1)

- Shipping all four full weight trees by default
- Mega multimodal SFT in the same job as REAP
- Cloud imatrix “for free accuracy”
- Claiming WebGPU/WGSL SOTA from a few shader strings in traces
- Running cloud REAP on synthetic `[CATEGORY TASK #NNNN]` filler calib

## Suggested HF layout

```
True2456/Step-3.7-Flash-REAP-p20          # example winner (p10 / p15 / p20 / p25)
True2456/Step-3.7-Flash-REAP-artifacts     # saliency + plan_p*.json + gate_log.json
```

## Script direction (not ready yet — stubs)

| Script | Status |
| --- | --- |
| `scripts/build_calib_mix.py` | **Broken for prod** — pads ~98% synthetic filler; must pull Fable-5 / real sources |
| `scripts/run_cloud_reap_pipeline.py` | **Stub** — simulated PPL/saliency; still labels 170B/165B/160B; no real apply/gate |
| `scripts/upload_hf.py` | Shape OK — needs real winner dir + `HF_TOKEN` |
| Cerebras Step-3.7 patches | **Missing** — only MiniMax patches in `vendor/PATCHES.md` |

Rework toward:

1. Real `build_calib_mix.py` → `calib/cloud_reap_8k.jsonl` (no synthetic pad)
2. `run_reap_saliency` → saliency + nested `plan_p10/p15/p20/p25`
3. `apply_reap_ladder` → 10 → 15 → 20 → 25 with PPL/smoke gate; keep last good
4. `upload_hf.py` → winner + artifacts
5. Manual gate before LoRA / re-saliency
