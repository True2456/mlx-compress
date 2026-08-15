"""Where does the ~13.5K-token teacher-vs-quantized divergence enter?

*** CAUTION: this divergence turned out to be a single-chunk testing
*** artifact, not a real property of the model -- see
*** docs/LING3-LONG-CONTEXT-QUANT-FINDINGS.md and
*** reap_stream/test_chunked_prefill_divergence.py. Kept because the
*** layer-by-layer localization *technique* is still useful for any future
*** divergence that survives chunked-prefill testing; the specific finding
*** in this docstring (gradual, non-localized growth) describes the
*** artifact, not a real model property.

test_long_context_divergence.py established (and verified against a
harness-faithfulness check) that Ling-3.0-flash's 8fixed-5routed quantized
checkpoint diverges sharply from the BF16 teacher's output distribution at
an absolute context length of ~13.5K-13.9K tokens -- same absolute position
regardless of total document length.

This script narrows down WHERE across the 42 decoder layers that divergence
first appears, by capturing the hidden state at two probe positions (one
just before the cliff, one just after) after every single layer, for both
the teacher and the quantized model, and comparing them layer by layer.

A divergence that's already large at layer 0 (pure embedding) would mean
the two probe positions are just different inputs -- uninteresting. A
divergence that's small for many layers then jumps at one specific layer
points at that layer's quantization (which experts, which projection) as
the proximate cause. A divergence that grows gradually across many layers
suggests compounding error rather than a single bad layer.

Usage:
    .venv/bin/python -m reap_stream.localize_divergence \
        --teacher models/Ling-3.0-flash \
        --quantized artifacts/ling3-8fixed-5routed \
        --pos-before 13439 --pos-after 13758 --target-tokens 16000
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load

from .collect_ling3 import _free_layer, _run_layer, _text_model, load_lazy
from .test_long_context_divergence import _build_long_document


def _run_capturing(model, inp, idx_arr, layers_at_once: int, free_as_we_go: bool):
    """Stream all decoder layers, capturing the hidden state at idx_arr
    positions after every single layer. Returns [n_layers+1, n_probes, hidden]
    (index 0 = post-embedding, before any decoder layer)."""
    text = _text_model(model)
    n_layers = len(text.layers)
    h = text.word_embeddings(inp)
    mx.eval(h)

    snapshots = [mx.take(h[0], idx_arr, axis=0)]
    mx.eval(snapshots[-1])

    for w0 in range(0, n_layers, layers_at_once):
        window = range(w0, min(w0 + layers_at_once, n_layers))
        for li in window:
            h = _run_layer(text.layers[li], h)
            snapshots.append(mx.take(h[0], idx_arr, axis=0))
            mx.eval(snapshots[-1])
        if free_as_we_go:
            for li in window:
                _free_layer(text, li)
            gc.collect()
            mx.clear_cache()
        act = mx.get_active_memory() / 1e9
        print(f"[localize] layers {w0}-{window[-1]}/{n_layers-1} active={act:.1f} GB",
              flush=True)

    return mx.stack(snapshots)  # [n_layers+1, n_probes, hidden]


def run(teacher_path: str, quantized_path: str, pos_before: int, pos_after: int,
        target_tokens: int, layers_at_once: int):
    idx_arr = mx.array([pos_before, pos_after])
    repo_root = Path(__file__).resolve().parents[1]

    print("[localize] tokenizing document...", flush=True)
    _, tok = load(quantized_path, lazy=True)
    doc = _build_long_document(repo_root, target_tokens)
    tokens = tok.encode(doc)
    if isinstance(tokens, dict):
        tokens = tokens["input_ids"]
    tokens = tokens[:target_tokens]
    inp = mx.array(tokens)[None]
    print(f"[localize] probes at positions {pos_before} (pre-cliff), "
          f"{pos_after} (post-cliff), seq_len={len(tokens)}", flush=True)

    print(f"[localize] teacher (streamed, free-as-we-go): {teacher_path}", flush=True)
    teacher_model, _ = load_lazy(teacher_path)
    teacher_snaps = _run_capturing(teacher_model, inp, idx_arr, layers_at_once,
                                    free_as_we_go=True)
    del teacher_model
    gc.collect()
    mx.clear_cache()

    print(f"[localize] quantized (streamed, resident so no need to free): {quantized_path}",
          flush=True)
    from .bailing_swiglu_clamp import apply_bailing_swiglu_clamp
    apply_bailing_swiglu_clamp()
    q_model, _ = load(quantized_path, lazy=False)
    q_snaps = _run_capturing(q_model, inp, idx_arr, layers_at_once, free_as_we_go=False)

    n_layers = teacher_snaps.shape[0] - 1
    print(f"\n[localize] layer | pos_before relerr | pos_after relerr | pos_after cos_sim")
    t32 = teacher_snaps.astype(mx.float32)
    q32 = q_snaps.astype(mx.float32)
    diff = t32 - q32
    rel_err = mx.sqrt((diff ** 2).sum(axis=-1)) / mx.maximum(
        mx.sqrt((t32 ** 2).sum(axis=-1)), 1e-6)
    dot = (t32 * q32).sum(axis=-1)
    norm_prod = mx.sqrt((t32 ** 2).sum(axis=-1)) * mx.sqrt((q32 ** 2).sum(axis=-1))
    cos_sim = dot / mx.maximum(norm_prod, 1e-6)
    mx.eval(rel_err, cos_sim)

    for li in range(n_layers + 1):
        label = "embed" if li == 0 else f"{li-1:>3d}"
        print(f"  {label:>5} | {rel_err[li, 0].item():>16.5f} | "
              f"{rel_err[li, 1].item():>16.5f} | {cos_sim[li, 1].item():>16.5f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--quantized", required=True)
    ap.add_argument("--pos-before", type=int, required=True)
    ap.add_argument("--pos-after", type=int, required=True)
    ap.add_argument("--target-tokens", type=int, default=16000)
    ap.add_argument("--layers-at-once", type=int, default=1)
    a = ap.parse_args()
    run(a.teacher, a.quantized, a.pos_before, a.pos_after, a.target_tokens,
        a.layers_at_once)
