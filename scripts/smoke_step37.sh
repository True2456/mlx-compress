#!/usr/bin/env bash
# Smoke-test Step-3.7 streaming REAP plumbing on the local already-REAPed MLX
# checkpoint. Does NOT write a pruned model into LM Studio — collect + dry-run only.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python"
MODEL="${MODEL:-$HOME/.lmstudio/models/mlx-community/Step-3.7-Flash-148B-MLX}"
OUT="${OUT:-$ROOT/artifacts/step37-smoke}"

echo "== inspect =="
"$PY" - <<PY
from reap_stream.collect_step3p7 import inspect_model
import json
print(json.dumps(inspect_model("$MODEL"), indent=2))
PY

echo
echo "== collect (layers 3 only, tiny tokens) =="
"$PY" -m reap_stream.cli collect \
  --arch step3p7 \
  --model "$MODEL" \
  --output "$OUT" \
  --layers 3 \
  --layers-at-once 1 \
  --max-tokens 64 \
  --ratio 0.10 \
  --min-experts 8

echo
echo "== apply dry-run =="
"$PY" -m reap_stream.cli apply \
  --arch step3p7 \
  --model "$MODEL" \
  --plan "$OUT/pruning-plan.json" \
  --output "$OUT/pruned-dry" \
  --dry-run

echo
echo "Smoke OK. Artifacts in $OUT (no LM Studio model was modified)."
