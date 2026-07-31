#!/usr/bin/env bash
# Launcher for the Step-3.7 QLoRA run (custom mlx_vlm-native trainer).
#
# mlx_lm.lora cannot load step3p7 (VLM), and mlx_vlm.lora's data pipeline is
# image-text oriented, so training goes through reap_stream.lora_sft_step37
# (text-only masked-CE SFT, mlx_vlm adapter attach/save). See docs/LORA-SETUP.md.
#
# Usage:
#   scripts/run_lora.sh smoke                 # 30 iters, validate pipeline+memory
#   scripts/run_lora.sh full [EPOCHS]         # iters = train_rows * EPOCHS
#   scripts/run_lora.sh custom --iters N ...  # pass trainer args through
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
MODEL=models/Step-3.7-p15-4bit-vblend-shared8
DATA=data/lora_step37
ADAPTERS=artifacts/lora/adapters
LOG=artifacts/lora/train.log
MODE="${1:-smoke}"

# --- real OS-level memory sampler (mx.get_peak_memory under-reports >2x) ---
start_mon() {
  ( while true; do
      pid=$(pgrep -f "lora_sft_step37" | head -1 || true)
      [ -n "${pid:-}" ] && footprint "$pid" 2>/dev/null | grep -E "phys_footprint:" \
        | sed "s/^/[footprint $(date +%H:%M:%S)] /" || true
      sleep 30
    done ) & MON=$!
  trap 'kill $MON 2>/dev/null || true' EXIT
}

# seq 2048 is the memory-validated ceiling: ~110GB resident with grad
# checkpointing, safely under the ~115GB wired limit. 8192 blew past it (241GB).
COMMON="--model $MODEL --data $DATA --adapter-dir $ADAPTERS --max-seq-length 2048 --lr 2.0e-5"

case "$MODE" in
  smoke)
    ARGS="$COMMON --iters 30 --steps-per-report 5 --steps-per-eval 20 --val-batches 5 --save-every 15"
    echo "[run_lora] SMOKE"
    ;;
  full)
    EPOCHS="${2:-1}"
    ROWS=$(wc -l < "$DATA/train.jsonl" | tr -d ' ')
    ITERS=$(( ROWS * EPOCHS ))
    ARGS="$COMMON --iters $ITERS --steps-per-report 10 --steps-per-eval 100 --val-batches 25 --save-every 100"
    echo "[run_lora] FULL: $EPOCHS epoch(s) x $ROWS rows = $ITERS iters"
    ;;
  custom)
    shift
    ARGS="$COMMON $*"
    echo "[run_lora] CUSTOM: $ARGS"
    ;;
  *)
    echo "unknown mode: $MODE (smoke|full|custom)"; exit 1 ;;
esac

start_mon
caffeinate -s $PY -m reap_stream.lora_sft_step37 $ARGS 2>&1 | tee -a "$LOG"
