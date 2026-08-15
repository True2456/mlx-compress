"""One-off: test whether protecting a given set of layers at native mxfp4
rescues coherence for the 2-bit gate/up + 3-bit down_proj policy, without
saving to disk (in-memory only). See reap_stream/build_deepseek_v4_quant98.py
for the base policy this modifies.

Usage:
    .venv/bin/python scripts/test_layer_protection_dsv4.py --protect 0 1 2
"""
from __future__ import annotations

import argparse
import gc
import time

import mlx.core as mx
from mlx_vlm import load


def dequant(m):
    return mx.dequantize(m.weight, m.scales, m.get("biases"), group_size=m.group_size, bits=m.bits, mode=m.mode)


def requant(m, bits, gs):
    full = dequant(m)
    mx.eval(full)
    out = mx.quantize(full, group_size=gs, bits=bits, mode="affine")
    m.weight, m.scales = out[0], out[1]
    m.group_size, m.bits, m.mode = gs, bits, "affine"
    m.biases = out[2] if len(out) > 2 else None
    # Force evaluation of the NEW quantized weight before freeing `full` --
    # otherwise m.weight/m.scales stay lazy graph nodes that still reference
    # `full` transitively, so clear_cache() frees nothing and every layer's
    # dequantized float tensor accumulates (the exact OOM this project's own
    # build_deepseek_v4_quant98.py docstring already documents and avoids).
    mx.eval(m.weight, m.scales)
    del full, out
    gc.collect()
    mx.clear_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protect", type=int, nargs="*", default=[])
    ap.add_argument("--model", default="/Users/true/Desktop/models/DeepSeek-V4-Flash-0731")
    a = ap.parse_args()
    protect = set(a.protect)

    model, processor = load(a.model, lazy=True)
    tok = getattr(processor, "tokenizer", processor)
    lm = model.language_model
    text = lm.model
    n_layers = len(text.layers)

    t0 = time.time()
    for i in range(n_layers):
        if i in protect:
            print(f"[test] layer {i:02d} PROTECTED (native)", flush=True)
            continue
        sg = text.layers[i].ffn.switch_mlp
        requant(sg.gate_proj, 2, 128)
        requant(sg.up_proj, 2, 128)
        requant(sg.down_proj, 3, 128)
        print(f"[test] layer {i:02d}/{n_layers - 1} quantized ({time.time() - t0:.0f}s)", flush=True)

    gc.collect()
    mx.clear_cache()
    print(f"[test] all layers processed, protected={sorted(protect)} ({time.time() - t0:.0f}s)", flush=True)

    prompt = "Hello, my name is"
    inp = mx.array(tok.encode(prompt))[None]
    out = lm(inputs=inp)
    logits = out.logits[0, -1]
    mx.eval(logits)
    top5 = mx.argsort(-logits)[:5]
    mx.eval(top5)
    print(f"[test] top5 with protected={sorted(protect)}:", flush=True)
    for t in top5.tolist():
        print(" ", t, repr(tok.decode([t])), flush=True)


if __name__ == "__main__":
    main()
