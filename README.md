# LLM REAP Workspace

Layer-by-layer **REAP** (Router-weighted Expert Activation Pruning) for frontier MoE models, built on [CerebrasResearch/reap](https://github.com/CerebrasResearch/reap) with MiniMax-M2/M2.7 patches.

## What's here

| Path | Purpose |
|------|---------|
| `vendor/cerebras-reap/` | Upstream REAP + MiniMax-M2 support patches |
| `models/` | Local checkpoints (symlink to `MiniMax-M2.7-172B-BF16`) |
| `scripts/` | Setup + layer-wise prune launchers |
| `artifacts/` | Observations, plans, pruned outputs |
| `calib/cerebras_reap_mix.jsonl` | Cerebras agentic 6-way mix (24,576 rows) |

## Current checkpoint

`MiniMax-M2.7-172B-BF16` is already a **25% REAP** of MiniMax-M2.7:

- 192 experts / layer (from 256)
- 62 layers, top-k=8, ~172B BF16 (~321 GB on disk)

To prune the **full** 230B (256 experts), put `MiniMaxAI/MiniMax-M2.7` BF16 under `models/` and point `MODEL_PATH` at it.

## Hardware requirements

Layer-wise REAP (`python -m reap.layerwise_prune`) keeps the full model on **CPU RAM** and moves **one transformer block** to GPU at a time.

| Need | Why |
|------|-----|
| NVIDIA GPU (H100/H200/A100 class) | Calibration + prune apply |
| Host RAM ≥ model size | Full BF16 must fit on CPU (~320–450 GB for MiniMax-M2.7) |
| Free disk ≥ 1× model size | Pruned checkpoint write |

This Mac (M5 Max, 128 GB) **cannot** run MiniMax layer-wise REAP locally. Use HF Jobs / DGX / multi-GPU host (same pattern as `quantize-nvfp4-gb10-agentic.py`).

## Quick start (GPU host)

```bash
# 1) Build Cerebras REAP env (Python 3.12 +, CUDA)
bash scripts/setup_env.sh

# 2) Layer-wise REAP with the agentic 6-dataset mix (Cerebras HF recipe)
bash scripts/run_layerwise_reap.sh \
  --model "$(pwd)/models/MiniMax-M2.7-172B-BF16" \
  --ratio 0.25 \
  --gpus 0
```

Override examples:

```bash
# Full 230B BF16 path + 30% prune (256 → ~179 experts)
MODEL_PATH=/data/MiniMax-M2.7-BF16 RATIO=0.30 bash scripts/run_layerwise_reap.sh

# Observer only (collect saliency, skip prune/save)
RUN_OBSERVER_ONLY=1 bash scripts/run_layerwise_reap.sh --model ... --ratio 0.25
```

Default calibration mix (agentic):

```text
theblackcat102/evol-codealpaca-v1:4096,
Salesforce/xlam-function-calling-60k:4096,
open-r1/Mixture-of-Thoughts[code]:4096,
open-r1/Mixture-of-Thoughts[math]:4096,
open-r1/Mixture-of-Thoughts[science]:4096,
SWE-bench/SWE-smith-trajectories(tool):4096
```

## MiniMax patches (in `vendor/cerebras-reap`)

Upstream REAP did not register `MiniMaxM2ForCausalLM`. Local changes:

1. `MODEL_ATTRS` + observer registry for `MiniMaxM2SparseMoeBlock`
2. Sigmoid router scores + `e_score_correction_bias` for top-k (matches MiniMax routing)
3. Pruner preserves `MiniMaxM2Experts` subclass and slices MoE-level bias

## Disk pressure

This volume is nearly full (~12 GB free). Free space or point `ARTIFACTS_DIR` / pruned output to another disk **before** running — a pruned BF16 write needs hundreds of GB.

## References

- Paper: [REAP the Experts](https://arxiv.org/abs/2510.13999)
- Upstream: [CerebrasResearch/reap](https://github.com/CerebrasResearch/reap)
- Layer-wise CLI: `vendor/cerebras-reap/experiments/pruning-layerwise-cli.sh`
- Prior MiniMax REAP checkpoints: [cerebras/MiniMax-M2-REAP collection](https://huggingface.co/collections/cerebras/cerebras-reap)


## Mac streaming REAP (Gemma4 test)

Uses the same Cerebras 6-way mix locally (`calib/cerebras_reap_mix.jsonl`). Rebuild with:

```bash
.venv/bin/python scripts/build_cerebras_calib_mix.py
```

(xLAM via ungated mirror `NobodyExistsOnTheInternet/xlam-function-calling-60k`.)

```bash
bash scripts/reap_gemma4.sh
# or smoke on 3 layers:
.venv/bin/python -m reap_stream.cli collect \
  --model ~/.lmstudio/models/lmstudio-community/gemma-4-26B-A4B-it-QAT-MLX-4bit \
  --dataset-file calib/cerebras_reap_mix.jsonl \
  --output artifacts/gemma4-smoke \
  --ratio 0.25 --layers 0-2 --max-tokens 128
```

## Mac streaming REAP (Laguna-S-2.1)

`poolside/Laguna-S-2.1-NVFP4-mlx` (local LM Studio download, not on the
public Hub under this project's radar until now) — already native NVFP4
4-bit, 256 experts/layer, 47 MoE layers. Dedicated collect/apply pair
(ported from `collect_deepseek_v4.py`/`apply_deepseek_v4.py`, simpler:
no hyper-connections, no hash-routed layers — see
`docs/LAGUNA-REAP-FINDINGS.md` for the full writeup).

```bash
.venv/bin/python -m reap_stream.collect_laguna \
  --model ~/.lmstudio/models/poolside/Laguna-S-2.1-NVFP4-mlx \
  --output artifacts/laguna-reap \
  --dataset-file calib/cerebras_reap_mix.jsonl \
  --max-samples 2500 --max-tokens 1024 --ratio 0.375

.venv/bin/python -m reap_stream.apply_laguna \
  --model ~/.lmstudio/models/poolside/Laguna-S-2.1-NVFP4-mlx \
  --plan artifacts/laguna-reap/pruning-plan.json \
  --output models/Laguna-S-2.1-REAP
```

Measured result: 67GB → 45GB (256→160 experts, 37.5% prune), targeting a
64GB deploy machine. Not yet accuracy-tested beyond a load+generate smoke
check — see `docs/LAGUNA-REAP-FINDINGS.md` §7.
