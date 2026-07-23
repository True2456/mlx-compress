#!/usr/bin/env bash
# Build the Cerebras REAP environment inside vendor/cerebras-reap.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REAP_DIR="${ROOT}/vendor/cerebras-reap"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: no nvidia-smi — REAP layer-wise prune needs an NVIDIA GPU host." >&2
fi

cd "${REAP_DIR}"
bash scripts/build.sh

echo
echo "REAP env ready: ${REAP_DIR}/.venv"
echo "Activate with: source ${REAP_DIR}/.venv/bin/activate"
