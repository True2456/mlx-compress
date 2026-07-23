#!/usr/bin/env bash
# Layer-wise REAP prune for frontier MoE models (MiniMax-M2/M2.7 ready).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REAP_DIR="${ROOT}/vendor/cerebras-reap"
VENV_PY="${REAP_DIR}/.venv/bin/python"

MODEL_PATH="${MODEL_PATH:-${ROOT}/models/MiniMax-M2.7-172B-BF16}"
RATIO="${RATIO:-0.25}"
SEED="${SEED:-42}"
GPUS="${GPUS:-0}"
METHOD="${METHOD:-reap}"
RUN_OBSERVER_ONLY="${RUN_OBSERVER_ONLY:-false}"
DO_EVAL="${DO_EVAL:-false}"
BATCH_SIZE="${BATCH_SIZE:-8}"
BATCHES_PER_CATEGORY="${BATCHES_PER_CATEGORY:-128}"
LOW_CPU_MEM="${LOW_CPU_MEM:-True}"

# Cerebras agentic calibration mix used for published MiniMax REAP checkpoints.
DATASET="${DATASET:-theblackcat102/evol-codealpaca-v1:4096,Salesforce/xlam-function-calling-60k:4096,open-r1/Mixture-of-Thoughts[code]:4096,open-r1/Mixture-of-Thoughts[math]:4096,open-r1/Mixture-of-Thoughts[science]:4096,SWE-bench/SWE-smith-trajectories(tool):4096}"

usage() {
  cat <<USAGE
Usage: $0 [--model PATH] [--ratio 0.25] [--gpus 0] [--method reap] [--dataset SPEC]

Env overrides: MODEL_PATH RATIO SEED GPUS METHOD DATASET RUN_OBSERVER_ONLY DO_EVAL
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_PATH="$2"; shift 2 ;;
    --ratio) RATIO="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --method) METHOD="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ ! -x "${VENV_PY}" ]]; then
  echo "REAP venv missing. Run: bash scripts/setup_env.sh" >&2
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model path not found: ${MODEL_PATH}" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Layer-wise REAP requires an NVIDIA GPU host." >&2
  echo "This Mac cannot run MiniMax REAP; use HF Jobs / DGX / remote CUDA box." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPUS}"
FIRST_DEVICE="$(echo "${GPUS}" | cut -d',' -f1)"
PORT=$((8000 + FIRST_DEVICE))
OUTPUT_FILE="observations_${BATCHES_PER_CATEGORY}_cosine-seed_${SEED}.pt"
LOG_FILE="${ROOT}/artifacts/layerwise-reap-gpu${FIRST_DEVICE}.log"

mkdir -p "${ROOT}/artifacts"

echo "=== Layer-wise REAP ==="
echo "model:   ${MODEL_PATH}"
echo "ratio:   ${RATIO}"
echo "method:  ${METHOD}"
echo "gpus:    ${CUDA_VISIBLE_DEVICES}"
echo "dataset: ${DATASET}"
echo "log:     ${LOG_FILE}"

cd "${REAP_DIR}"
# shellcheck disable=SC1091
source .venv/bin/activate

python -m reap.layerwise_prune \
  --model-name "${MODEL_PATH}" \
  --dataset-name "${DATASET}" \
  --compression-ratio "${RATIO}" \
  --prune-method "${METHOD}" \
  --profile false \
  --vllm_port "${PORT}" \
  --server-log-file-name "${LOG_FILE}" \
  --do-eval "${DO_EVAL}" \
  --run_observer_only "${RUN_OBSERVER_ONLY}" \
  --seed "${SEED}" \
  --output_file_name "${OUTPUT_FILE}" \
  --batches_per_category "${BATCHES_PER_CATEGORY}" \
  --batch_size "${BATCH_SIZE}" \
  --low_cpu_mem_usage "${LOW_CPU_MEM}" \
  --renormalize_router_weights true \
  --record_pruning_metrics_only true \
  2>&1 | tee -a "${LOG_FILE}"

echo "Done. Check vendor/cerebras-reap/artifacts/ for pruned models + observations."
