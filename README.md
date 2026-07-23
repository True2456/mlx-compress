# LLM REAP & Frontier MoE Workspace

Layer-by-layer **REAP** (Router-weighted Expert Activation Pruning) and frontier MoE compression research for **MiniMax-M2/M2.7** and **Step-3.7-Flash** (198B VLM, 288 experts, top-$k=8$), built on [CerebrasResearch/reap](https://github.com/CerebrasResearch/reap) with local Apple Silicon MLX streaming pipelines.

---

## What's Here

| Path | Purpose |
|------|---------|
| `reap_stream/` | Local Mac streaming collector, plan generator, student builder (`Step-3.7-Flash`) |
| `vendor/cerebras-reap/` | Upstream REAP + MiniMax-M2 support patches |
| `models/` | Local checkpoints (Step-3.7-Flash, MiniMax-M2.7) |
| `docs/` | Research findings, mathematical audits, and technical proposals |
| `artifacts/` | Collected saliencies, plans, diagnostic logs, and pruned student builds |
| `calib/` | Multi-dataset calibration mixes (agentic, coding, math, reasoning, fixed multimodal) |

---

## Research & Documentation

Comprehensive analysis and mathematical derivations are available in `docs/`:

* 📄 [docs/step37-reap-dwq-summary.md](file:///Users/true/Desktop/LLM%20-%20Reap/docs/step37-reap-dwq-summary.md): Full session summary on Step-3.7 REAP, 4-bit affine/nvfp4 quantization, DWQ distillation lessons, and MLX memory optimizations (`mx.clear_cache()`).
* 📄 [docs/step37-reap-findings.md](file:///Users/true/Desktop/LLM%20-%20Reap/docs/step37-reap-findings.md): Saliency observations (zero dead experts, local Jaccard ~0.12, 15% REAP keep-245 decision).
* 📄 [docs/frontier-moe-pruning-theorems.md](file:///Users/true/Desktop/LLM%20-%20Reap/docs/frontier-moe-pruning-theorems.md): Mathematical derivations and proofs for 6 frontier "dark horse" MoE algorithms.
* 📄 [docs/frontier-moe-theorems-audit.md](file:///Users/true/Desktop/LLM%20-%20Reap/docs/frontier-moe-theorems-audit.md): Stress-test audit, edge-case corrections (SwiGLU, RMSNorm projection cancellation, top-8 hard thresholding bounds), and official Step-3.7 paper spec cross-check.
* 📄 [docs/gemini3.6_proposal.md](file:///Users/true/Desktop/LLM%20-%20Reap/docs/gemini3.6_proposal.md): Proposal for Router Bias Surgery + Scale-Normalized DPP-REAP, M5 Max 128 GB memory analysis, and technical counter-critique.

---

## Models & Targets

### 1. Step-3.7-Flash (StepFun 198B VLM)
- **Architecture**: 45 layers (42 MoE layers, dense 0–2), 288 routed experts + 1 shared expert per MoE layer, top-$k=8$ sigmoid routing with `e_score_correction_bias` (`router_bias`), SwiGLU, RMSNorm.
- **Base Checkpoint**: `models/Step-3.7-Flash` (375 GB BF16).
- **Pruned 15% Student**: `models/Step-3.7-p15-4bit` (92 GB 4-bit affine, 245 experts/layer, vision tower BF16).

### 2. MiniMax-M2.7-172B-BF16
- **Architecture**: 62 layers, 192 experts/layer (25% REAP of 256-expert 230B base), top-$k=8$.

---

## Hardware Requirements & Mac Streaming (M5 Max 128 GB)

Layer-wise REAP (`reap_stream`) streams one block window resident in GPU RAM at a time while carrying hidden states across windows, allowing full 198B MoE saliency collection on an **M5 Max Mac (128 GB RAM)**.

| Component | Precision / Shape | RAM Footprint | Memory Status |
|---|---|---|---|
| **Resident 15% Student** (`Step-3.7-p15-4bit`) | 4-bit affine (245 experts) | **92.0 GB** | Fits within 115 GB wired GPU limit |
| **Streaming Saliency Collector** | Layerwise block streaming | **~34.0 GB** | Optimized via `mx.clear_cache()` every 200 steps |
| **System Overhead** | macOS system memory | **~10.0 GB** | Reserved |

---

## Quick Start (Mac MLX Streaming REAP)

### 1. Run Corrected Saliency Collection (`headtail` 1024 tokens)
```bash
.venv/bin/python -m reap_stream.cli collect \
  --model models/Step-3.7-Flash \
  --dataset-file calib/cloud_reap_8k.jsonl \
  --output artifacts/step37-1024-headtail \
  --max-tokens 1024 \
  --truncation headtail \
  --layers-at-once 2
```

### 2. Build Fused Pruned + 4-bit Quantized Student
```bash
.venv/bin/python -m reap_stream.cli apply \
  --model models/Step-3.7-Flash \
  --plan artifacts/step37-1024-headtail/plan_p15.json \
  --output models/Step-3.7-p15-4bit \
  --mode affine \
  --group-size 64
```

---

## Refined Empirical Roadmap

1. **Co-occurrence Correlation Measurement**: Compute $\mathbf{C}_{i,j} = \frac{\sum_{t} (g_i h_i)^T (g_j h_j)}{\sqrt{\sum_t \|g_i h_i\|^2 \sum_t \|g_j h_j\|^2}}$ over top-8 co-occurrences to measure true cross-expert feature redundancy.
2. **RMSNorm Residual Projection Saliency**: Drop-in replacement metric $S_{\text{RMSNorm}}(e) = \mathbb{E}_t [g_{t,e} \|h_{t,e} - \frac{\langle x_t, h_{t,e}\rangle}{\|x_t\|_2^2} x_t\|]$ to filter out parallel energy rescaled by downstream RMSNorm.
3. **Trace Normalization**: Layer trace scaling $\frac{S_l(e)}{\operatorname{Tr}(S_l)}$ to resolve the $0.17 \to 24.0$ cross-depth magnitude explosion.

---

## References & Citations

- **REAP Paper**: [REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression](https://arxiv.org/abs/2510.13999) (Cerebras Research, ICLR 2026).
- **Upstream Code**: [CerebrasResearch/reap](https://github.com/CerebrasResearch/reap)
