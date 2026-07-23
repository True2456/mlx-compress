#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python"
MODEL="${MODEL:-$HOME/.lmstudio/models/lmstudio-community/gemma-4-26B-A4B-it-QAT-MLX-4bit}"
OUT="${OUT:-$ROOT/artifacts/gemma4}"
RATIO="${RATIO:-0.25}"
MODE="${MODE:-layerwise}"

DATASET="${DATASET:-$ROOT/calib/cerebras_reap_mix.jsonl}"

echo "Collecting REAP ($MODE) on: $MODEL"
echo "Calib: $DATASET"
"$PY" -m reap_stream.cli collect \
  --model "$MODEL" \
  --output "$OUT" \
  --ratio "$RATIO" \
  --mode "$MODE" \
  --dataset-file "$DATASET" \
  "$@"
echo
echo "Plan ready. Apply with:"
echo "  $PY -m reap_stream.cli apply --model \"$MODEL\" --plan \"$OUT/pruning-plan.json\" --output \"$OUT/pruned\""
