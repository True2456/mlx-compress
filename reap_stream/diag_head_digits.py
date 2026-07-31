"""Is the 4-bit output head what's eating digits?

The shared8 tomography result found the real headroom was in the weight class
that fires on *every* token (share_expert, dense MLPs, router gates) -- no
top-k sparsity, so no implicit smoothing. `embed_tokens` and `lm_head` are in
that same class, but they were never a sweep variant: the build predicate
returns `True` for them, so they sit at 4-bit gs64 like a routed expert.

Digit tokens are the worst case for a quantized output head. They form a tight,
near-equidistant cluster of rows (the model must separate '4' from '5' from 'a'
using small margins), and they need exact argmax, not a good distribution --
one rank flip corrupts the whole number. Aggregate PPL barely moves, because
digits are a rounding error in token count. That matches the reported symptom:
everything fine except numbers.

This measures lm_head/embed_tokens row reconstruction error, grouped by token
class, straight off disk -- no model load, no generation. If digit rows are not
worse than baseline rows, the head is exonerated and the cause is elsewhere
(see docs/LLGUIDANCE-CRASH-DIAGNOSIS.md for the structured-output path).

Indicator, not proof. Proof is the A/B rebuild with bits=8 on both tensors.

Usage:
    .venv/bin/python -m reap_stream.diag_head_digits \
        --bf16  models/Step-3.7-Flash \
        --quant models/Step-3.7-p15-4bit-vblend-shared8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
from transformers import AutoTokenizer

# Name pairs differ between the source checkpoint and the mlx_vlm student.
TENSORS = [
    ("lm_head", "lm_head.weight", "language_model.lm_head.weight"),
    ("embed_tokens", "model.embed_tokens.weight", "language_model.model.embed_tokens.weight"),
]


def _shard_of(model_dir: Path, key: str) -> Path:
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())
    return model_dir / idx["weight_map"][key]


def _load(model_dir: Path, key: str):
    return mx.load(str(_shard_of(model_dir, key)))[key]


def _load_quant(model_dir: Path, key: str, group_size: int, bits: int):
    shard = mx.load(str(_shard_of(model_dir, key)))
    stem = key.rsplit(".weight", 1)[0]
    return mx.dequantize(
        shard[key], shard[f"{stem}.scales"], shard[f"{stem}.biases"],
        group_size=group_size, bits=bits,
    )


def token_classes(tokenizer):
    """Row indices for digit-bearing tokens vs a same-size random control."""
    vocab = tokenizer.get_vocab()
    digit, alnum_confusable = [], []
    for tok, i in vocab.items():
        s = tokenizer.convert_tokens_to_string([tok])
        if not s:
            continue
        if any(c.isdigit() for c in s):
            digit.append(i)
        elif s.strip() and all(c.isalnum() for c in s.strip()):
            alnum_confusable.append(i)
    return {"digit": digit, "alpha": alnum_confusable}


def row_error(ref, deq, idx):
    """Mean relative L2 error and mean cosine similarity over the given rows."""
    r = ref[mx.array(idx)].astype(mx.float32)
    d = deq[mx.array(idx)].astype(mx.float32)
    err = mx.linalg.norm(r - d, axis=-1) / (mx.linalg.norm(r, axis=-1) + 1e-9)
    cos = mx.sum(r * d, axis=-1) / (
        mx.linalg.norm(r, axis=-1) * mx.linalg.norm(d, axis=-1) + 1e-9
    )
    return float(mx.mean(err)), float(mx.mean(cos))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--quant", required=True)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    bf16, quant = Path(a.bf16), Path(a.quant)
    tokenizer = AutoTokenizer.from_pretrained(str(bf16), trust_remote_code=True)
    classes = token_classes(tokenizer)
    print(f"[INFO] digit-bearing tokens: {len(classes['digit'])}, "
          f"alphanumeric control: {len(classes['alpha'])}")

    report = {}
    for label, ref_key, q_key in TENSORS:
        print(f"\n=== {label} ===")
        ref = _load(bf16, ref_key)
        deq = _load_quant(quant, q_key, a.group_size, a.bits)
        if ref.shape != deq.shape:
            raise ValueError(f"{label}: shape mismatch {ref.shape} vs {deq.shape}")

        all_idx = list(range(ref.shape[0]))
        rows = {"all": all_idx, **classes}
        report[label] = {}
        print(f"{'rows':<10}{'n':>8}{'rel L2':>12}{'cosine':>12}")
        for name, idx in rows.items():
            err, cos = row_error(ref, deq, idx)
            report[label][name] = {"n": len(idx), "rel_l2": err, "cosine": cos}
            print(f"{name:<10}{len(idx):>8}{err:>12.5f}{cos:>12.6f}")

        d, base = report[label]["digit"]["rel_l2"], report[label]["all"]["rel_l2"]
        print(f"digit rows are {d / base:.3f}x the all-row error")
        del ref, deq
        mx.clear_cache()

    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2))
        print(f"\n[INFO] wrote {a.out}")


if __name__ == "__main__":
    main()
