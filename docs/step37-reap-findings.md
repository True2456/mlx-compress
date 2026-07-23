# Step-3.7-Flash REAP findings (Mac BF16, 5k calib)

> ⚠️ **SUPERSEDED IN PART.** This run used `max_tokens=384` **head** truncation,
> which cuts off the `ASSISTANT:` answer for **96% of agentic** and **96% of
> general_instruction** prompts (their system prompts + tool schemas alone exceed
> the budget; median prompt is 798 tokens but p90 is 5195).
>
> Measured against full-length ground truth, this saliency is **~78% faithful**
> (Jaccard 0.783 on the bottom-25% prune set) vs **0.858 for `headtail`-1024**.
> Not garbage — but the plans below are built on a partially-truncated signal.
>
> A corrected run (`--max-tokens 1024 --truncation headtail`) is in
> `artifacts/step37-1024-headtail/`. See `docs/step37-reap-dwq-summary.md` §6b.
> **Everything else in this document still stands.**

**Status:** collect + nested plans **done**. Weight apply **not** done yet.

| Item | Value |
| --- | --- |
| Base | `models/Step-3.7-Flash` (BF16, 288 experts / MoE layer) |
| Artifacts | `artifacts/step37-bf16-layerwise-5k/` |
| Calib | `calib/cloud_reap_8k.jsonl` — **5000** prompts × **384** tokens ⚠️ *head-truncated; see banner* |
| MoE layers scored | **42** (decoder layers **3–44**; dense **0–2** not MoE) |
| Runtime | **~1.88 h** on M5 Max 128 GB (`layers_at_once=2`) |
| Peak MLX memory | ~**45 GB** |
| Metric | REAP = mean of `(gate_weight × ‖expert_activation‖)` over routed tokens |

Earlier 1k run preserved at `artifacts/step37-bf16-layerwise-v2/` (do not overwrite).

---

## What we are pruning

**Not whole layers.** Step MoE keeps every transformer layer; REAP drops **experts inside** each MoE layer independently.

- Dense layers **0–2**: never pruned (no MoE).
- MoE layers **3–44**: each has **288** experts; plans keep the highest-REAP experts and prune the lowest.

Plans are **nested** (verified): prune set at 10% ⊂ 15% ⊂ 20% ⊂ 25%.

| Plan file | Prune ratio | Keep / layer | Prune / layer |
| --- | ---: | ---: | ---: |
| `plan_p10.json` | 10% | **260** | 28 |
| `plan_p15.json` | 15% | **245** | 43 |
| `plan_p20.json` | 20% | **231** | 57 |
| `plan_p25.json` | 25% | **216** | 72 |

**Recommended first apply:** `plan_p10` or `plan_p15`. Use 25% only after a quality gate.

---

## Main findings

### 1. Every expert gets used

Across all 42 MoE layers, **zero** experts had zero routing hits on this 5k mix.  
Pruning is **not** “delete unused experts.” It is “drop the **lowest-saliency** among experts that still fire.”

### 2. Prune sets are layer-local (not a global kill list)

At 25% prune, Jaccard overlap of prune IDs is low across depth:

| Pair | Jaccard (prune sets) |
| --- | ---: |
| Layer 3 vs 24 | ~0.12 |
| Layer 24 vs 44 | ~0.15 |
| Layer 3 vs 44 | ~0.11 |
| In all three prune sets | **4** experts only |

**Implication:** you cannot pick 72 global expert IDs and delete them everywhere. Apply **per-layer plans** as written.

### 3. Raw REAP scores grow with depth — don’t compare across layers

Mean REAP (same metric, different scale):

| Layer | Mean REAP | p25 (≈25% cutoff) | Max |
| ---: | ---: | ---: | ---: |
| 3 (early) | ~0.17 | ~0.11 | ~0.94 |
| 12 | ~1.42 | ~0.83 | ~4.2 |
| 24 (mid) | ~6.2 | ~2.9 | ~28 |
| 36 | ~12.6 | ~6.2 | ~54 |
| 44 (last) | ~24 | ~8.0 | ~864 |

Ranking **within** each layer is what matters. Cross-layer score magnitudes are not comparable without normalization.

### 4. High routing frequency ≠ important

Classic REAP signal: some experts are **busy but low-saliency**.

Example — **layer 3, expert 67**:
- Very high routing frequency (~149k hits)
- **Lowest** REAP in that layer (~0.044)

So “often selected” does not mean “keep.” Gate × activation norm is the prune signal.

### 5. A few IDs lean weak or strong *often*, but none are universal dead weight

How often an expert lands in the **bottom 25%** or **top 25%** across 42 MoE layers (5k run):

**More often prune-zone (weak lean):**

| Expert ID | Times in bottom 25% / 42 |
| ---: | ---: |
| 72 | 20 |
| 275 | 18 |
| 87, 89, 168, 243 | 17 each |
| 47, 223, 210, 56 | 16 each |

**More often keep-zone (strong lean):**

| Expert ID | Times in top 25% / 42 |
| ---: | ---: |
| 190 | 20 |
| 152 | 19 |
| 250, 86 | 18 |
| 132, 257, 127 | 17 |

**No expert** was in the bottom 25% on *every* layer. Treat the weak list as a **hint**, not a global delete list.

### 6. Example keep / prune IDs (illustrative only)

**Layer 3** — lowest REAP (most pruneable): `67, 9, 169, 6, 260, 283, 37, 216, 22, 96`  
**Layer 3** — highest REAP: `58, 107, 124, 194, 135, 123, 16, 187, 274, 102`

**Layer 24** — lowest: `2, 220, 76, 135, 176, 278, 138, 23, 174, 39`  
**Layer 24** — highest: `77, 6, 261, 121, 243, 34, 65, 102, 89, 56`

**Layer 44** — lowest: `180, 259, 172, 57, 39, 171, 55, 53, 223, 166`  
**Layer 44** — highest: `86, 232, 88, 125, 134, 279, 224, 127, 197, 123`

Full keep/prune lists live in the plan JSON files.

---

## What this does *not* mean yet

- We have **not** applied weights — no smaller checkpoint on disk.
- We have **not** run held-out PPL / gen gates on a pruned model.
- 5k calib is strong for Mac; Cerebras-scale (~24k) may reorder a few boundary experts.
- Vision tower / shared expert / MTP dense blocks were **not** REAP targets in this pass.

---

## Files to use

```
artifacts/step37-bf16-layerwise-5k/
  saliency.json      # per-layer REAP scores + freqs
  plan_p10.json      # keep 260
  plan_p15.json      # keep 245  ← good first apply
  plan_p20.json      # keep 231
  plan_p25.json      # keep 216
  summary.json
  trace.json
```

Apply later when disk allows (~171 GB free today; full BF16 pruned copy needs ~300 GB free unless you free/move the base or quantize on write).

---

## Suggested next steps

1. Apply **`plan_p10` or `plan_p15`** to BF16 (external disk or free base after verify).
2. Smoke: coding + tool + short reasoning prompts vs base.
3. If gate passes, try **p20**; stop before **p25** if quality dips.
4. Optional: post-train / LoRA recovery on the winner.
5. Quantize (e.g. NVFP4) on Mac for local serving.
