"""Collect an oQe-compatible importance matrix by STREAMING the model
block-by-block, so a checkpoint far larger than unified memory can be
calibrated on one machine.

Why this exists: oMLX's own oQe calibration loads the model for ordinary
forward passes, and on DeepSeek-V4-Flash-0731 that fails outright --
its RAM-safe proxy is 151.5GB against a 90GB calibration limit, because a
uniform 4-bit proxy of an *already*-mxfp4 checkpoint doesn't shrink. No memory
setting fixes that (measured: capacity 128.8GB, so even fraction=1.0 falls
short).

But an imatrix is only a running sum of squared input activations per weight
tensor -- verified against oq.py's own collector:

    entry.in_sum2 += np.square(x).sum(axis=0)        # dense: [in_features]
    entry.in_sum2[expert] += x_sq[rows].sum(axis=0)  # MoE:  [n_experts, in_features]

No global state, nothing that needs the whole model resident. Streaming
layer-by-layer therefore computes *identical* statistics at ~6GB instead of
151.5GB: at oQ's defaults (128 samples x 512 tokens) the hidden states are
128*512*hc_mult(4)*4096*2B ~= 2.1GB, plus one layer (~3.6GB).

This reuses oq.py's OQImatrixCollector and _save_oqe_imatrix directly rather
than reimplementing them, so the output is byte-compatible with what oQ writes
itself and is accepted by its cache validator (plain metadata key equality).

An imatrix is activation statistics, not a quantization method, so the result
also drives GGUF/llama.cpp, a custom mixed-precision pass, or answers the open
bit-allocation question (which projections deserve more bits) directly.

Usage:
    PYTHONPATH=/Applications/oMLX.app/Contents/Resources \
    .venv/bin/python -m reap_stream.collect_imatrix_streamed \
        --model ~/Desktop/models/DeepSeek-V4-Flash-0731 \
        --dataset calib/ds4_agentic.jsonl \
        --out artifacts/imatrix_ds4_agentic.npz \
        --num-samples 128 --seq-length 512
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm import load
from mlx_vlm.models.base import create_attention_mask

OMLX = Path("/Applications/oMLX.app/Contents/Resources")


def _oq():
    if str(OMLX) not in sys.path:
        sys.path.insert(0, str(OMLX))
    import omlx.oq as oq
    return oq


def _text_model(model):
    lm = getattr(model, "language_model", None) or model
    return getattr(lm, "model", lm)


def _expand_hc(h, hc_mult):
    h = mx.broadcast_to(h[:, :, None, :], (h.shape[0], h.shape[1], hc_mult, h.shape[2]))
    return mx.contiguous(h)


class _Freed(nn.Module):
    def __call__(self, *a, **k):
        raise RuntimeError("freed layer invoked -- streaming bug")


def _load_prompts(dataset: str, n: int, oq, seq_length: int, tok):
    """Prefer the user's own corpus; fall back to oQ's bundled calibration."""
    texts: list[str] = []
    p = Path(dataset)
    if p.exists():
        for line in p.open():
            if not line.strip():
                continue
            rec = json.loads(line)
            t = rec.get("text") or rec.get("rendered") or ""
            if t.strip():
                texts.append(t)
            if len(texts) >= n:
                break
    if not texts:
        data = json.loads((OMLX / "omlx" / "oqe_calibration_data.json").read_text())
        for cat in ("tool_calling", "code", "reasoning", "chat", "en"):
            texts.extend(data.get(cat, []))
        texts = texts[:n]
    return [tok.encode(t)[:seq_length] for t in texts[:n]]


def run(model_path: str, dataset: str, out_path: str,
        num_samples: int, seq_length: int) -> None:
    oq = _oq()
    t0 = time.time()

    model, processor = load(model_path, lazy=True)
    tok = getattr(processor, "tokenizer", processor)
    text = _text_model(model)
    lm = getattr(model, "language_model", None) or model
    hc_mult = text.args.hc_mult
    sliding = text.args.sliding_window
    n_layers = len(text.layers)

    prompts = _load_prompts(dataset, num_samples, oq, seq_length, tok)
    prompts = [p for p in prompts if len(p) > 8]
    print(f"[imatrix] {len(prompts)} calibration samples, seq<={seq_length}, "
          f"{n_layers} layers", flush=True)

    # Reuse oQ's own collector so the output is format-identical.
    collector = oq.OQImatrixCollector()
    n_probes = collector.install(model)
    print(f"[imatrix] installed {n_probes} capture probes "
          f"({dict(collector.capture_module_classes)})", flush=True)

    # Embed all prompts once, then push them through one layer at a time.
    hidden, ids_list, masks = [], [], []
    for toks in prompts:
        a = mx.array(toks)[None]
        h = _expand_hc(text.embed_tokens(a), hc_mult)
        m = create_attention_mask(h[:, :, 0, :], None, window_size=sliding,
                                  return_array=True)
        mx.eval(h, m)
        hidden.append(h); ids_list.append(a); masks.append(m)
    print(f"[imatrix] embedded, active={mx.get_active_memory()/1e9:.1f}GB", flush=True)

    for li in range(n_layers):
        layer = text.layers[li]
        new = []
        for h, m, a in zip(hidden, masks, ids_list):
            o = layer(h, m, None, a)
            mx.eval(o)
            new.append(o)
        hidden = new
        text.layers[li] = _Freed()      # probe already fired; weights no longer needed
        # install() stashes every wrapped module in _original_modules; without
        # dropping this layer's entries those references keep the weights alive
        # and the stream leaks (measured: 3.4GB -> 98.3GB over 43 blocks).
        stale = [k for k in collector._original_modules if f".layers.{li}." in k]
        for k in stale:
            del collector._original_modules[k]
        gc.collect(); mx.clear_cache()
        if li % 5 == 0 or li == n_layers - 1:
            print(f"[imatrix] block {li}/{n_layers-1} "
                  f"entries={len(collector.entries)} "
                  f"active={mx.get_active_memory()/1e9:.1f}GB "
                  f"({time.time()-t0:.0f}s)", flush=True)

    entries = collector.entries
    if not entries:
        raise RuntimeError("no imatrix entries collected -- probes never fired")

    # Metadata: reuse oQ's own signature so its cache validator accepts this.
    cfg = json.loads((Path(model_path).expanduser() / "config.json").read_text())
    expected = oq._source_imatrix_signature(
        Path(model_path).expanduser(), cfg,
        num_samples=len(prompts), seq_length=seq_length,
        calib_dataset=oq._OQE_CALIB_DATASET,
    )
    coverage = {n: int(np.count_nonzero(e.counts)) for n, e in entries.items()
                if getattr(e.counts, "size", 1) > 1}
    metadata = {
        **expected,
        "entry_count": len(entries),
        "collection": {"streamed": True, "source": dataset,
                       "processed_samples": len(prompts)},
        "expert_coverage": coverage,
        "requires_expert_counts": bool(coverage),
        "processed_samples": len(prompts),
    }
    out = Path(out_path).expanduser()
    oq._save_oqe_imatrix(out, entries, metadata)
    print(f"[imatrix] wrote {out} ({len(entries)} entries) in "
          f"{time.time()-t0:.0f}s", flush=True)

    dense = sum(1 for e in entries.values() if np.ndim(e.in_sum2) == 1)
    print(f"[imatrix] dense tensors={dense}  MoE tensors={len(entries)-dense}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="calib/ds4_agentic.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-samples", type=int, default=128)
    ap.add_argument("--seq-length", type=int, default=512)
    a = ap.parse_args()
    run(a.model, a.dataset, a.out, a.num_samples, a.seq_length)


if __name__ == "__main__":
    main()
