"""Diagnostic: baseline KL (untouched student) on specific target files."""
import json
import sys
from pathlib import Path

import mlx.core as mx
from mlx_vlm import load

from .dwq_train_student import _kl_topk

STUDENT = "models/Step-3.7-p15-4bit"
TARGETS = "artifacts/dwq-targets"
FILES = ["00194", "00205", "00216", "00226", "00237", "00249"]

meta = json.loads((Path(TARGETS) / "targets_meta.json").read_text())
pad_len = int(meta["max_tokens"])

print("[diag] loading untouched student")
model, _ = load(STUDENT, lazy=False)
lm = model.language_model

for name in FILES:
    d = mx.load(f"{TARGETS}/{name}.safetensors")
    n = d["input_ids"].shape[0]
    pad = max(pad_len - n, 0)
    inp = mx.pad(d["input_ids"], (0, pad))[:pad_len][None]
    idx = mx.pad(d["topk_idx"], [(0, pad), (0, 0)])[:pad_len]
    vals = mx.pad(d["topk_vals"], [(0, pad), (0, 0)])[:pad_len]
    mask = mx.concatenate([mx.ones(min(n, pad_len)), mx.zeros(max(pad_len - n, 0))])[:pad_len]

    out = lm(input_ids=inp)
    loss = _kl_topk(out.logits[0], idx, vals, mask, 1.0)
    mx.eval(loss)
    print(f"[diag] {name} seq_len={n:4d} baseline_kl={loss.item():.4f}")
