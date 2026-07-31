#!/usr/bin/env bash
# One-time environment setup for the Gemma-4 QLoRA cloud box (CUDA/Unsloth).
# Run this ON the rented GPU instance, not locally -- this installs a CUDA
# stack that has nothing to do with the local MLX side of this project.
set -euo pipefail

echo "[setup] python/pip versions:"
python3 --version
pip3 --version

echo "[setup] installing Unsloth + training stack..."
pip3 install --upgrade pip
# Unsloth's own installer pins compatible torch/xformers/bitsandbytes for the
# detected CUDA version -- let it manage those rather than pinning ourselves.
pip3 install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git" || \
  pip3 install unsloth
pip3 install trl peft accelerate bitsandbytes datasets huggingface_hub

echo "[setup] GPU check:"
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); print('VRAM GB:', torch.cuda.get_device_properties(0).total_mem/1e9 if torch.cuda.is_available() else 0)"

echo "[setup] downloading Gemma-4 QAT checkpoint (62.6GB) -- do this on a"
echo "        persistent volume, not ephemeral disk, if using spot/preemptible."
echo "        huggingface-cli download google/gemma-4-31B-it-qat-q4_0-unquantized --local-dir ./gemma4-qat-bf16"
echo
echo "[setup] done. Next: pull training data + launch scripts/cloud_train_gemma4.py"
