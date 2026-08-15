#!/bin/bash
# Relaunch distributed DWQ across Thunderbolt link flaps; the trainer resumes
# from its last checkpoint (params + optimizer state).
CKPT="$1"; MAXSTEPS="$2"; TARGETS="$3"; shift 3
LOCAL=169.254.210.199; PEER=169.254.146.230
CD="/Users/true/Desktop/LLM - Reap"
LOG="$CD/artifacts/dwq_rank0.log"
for attempt in $(seq 1 200); do
  grep -q "done in" "$LOG" 2>/dev/null && { echo "[wrap] complete"; break; }
  echo "[wrap] === attempt $attempt at $(date +%T) ==="
  pkill -9 -f dwq_train_student_deepseek_v4_distributed 2>/dev/null
  pkill -9 -f "mlx.launch" 2>/dev/null
  ssh -o ConnectTimeout=10 $PEER 'pkill -9 -f dwq_train_student_deepseek_v4_distributed; pkill -9 -f mlx.launch' 2>/dev/null
  sleep 5
  for i in $(seq 1 60); do ping -c 1 -t 3 $PEER >/dev/null 2>&1 && break; echo "[wrap] link down ($i)"; sleep 10; done
  mlx.launch --hosts $LOCAL,$PEER --backend ring --cwd "$CD" -- \
    "$CD/scripts/dwq/rank_wrapper.sh" \
    --student models/DeepSeek-V4-Flash-0731-awq2bit3bit-v2 \
    --targets "$TARGETS" --ckpt-dir "$CKPT" --max-steps "$MAXSTEPS" "$@"
  grep -q "done in" "$LOG" 2>/dev/null && { echo "[wrap] completed"; break; }
  echo "[wrap] attempt $attempt ended; last: $(grep -oE 'step [0-9]+/' "$LOG" 2>/dev/null | tail -1)"
  sleep 20
done
echo "[wrap] finished $(date +%T)"
