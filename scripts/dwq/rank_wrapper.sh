#!/bin/bash
cd "/Users/true/Desktop/LLM - Reap"
exec ./.venv/bin/python -u -m reap_stream.dwq_train_student_deepseek_v4_distributed "$@" \
  > "/Users/true/Desktop/LLM - Reap/artifacts/dwq_rank${MLX_RANK}.log" 2>&1
