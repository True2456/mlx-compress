"""Teacher-vs-quantized divergence under REALISTIC chunked prefill.

test_long_context_divergence.py / localize_divergence.py / divergence_rate_by_length.py
all force the entire test sequence through KDA's recurrent kernel as ONE
giant single chunk with no cache -- convenient for a one-shot KL comparison,
but NOT how real serving processes long context. Real inference always
chunks a long prompt and carries the recurrent/KV state between chunks via
a real cache object (mlx_lm.models.cache.KVCache / ArraysCache).

This turned out to matter enormously: single-mega-chunk testing on
Ling-3.0-flash showed the quantized checkpoint catastrophically diverging
from the BF16 teacher past ~13.5K tokens (KL exploding to 3-6, argmax
flips, 100% of sampled positions broken by 14-16K) -- see
docs/LING3-LONG-CONTEXT-QUANT-FINDINGS.md for the full (now-superseded)
investigation. Seven hypotheses about quantization scheme were tested and
eliminated (routed-expert bits, KDA attention precision, router precision,
kernel choice, a real missing safe-gate clamp, document content) before
finding the actual cause: the single-chunk test methodology itself. Rerun
with real chunked prefill (this script), the divergence completely
disappeared -- every bucket dropped to the same near-zero KL seen at short
context. The existing quantization was fine all along; the test wasn't
representative of real usage.

**Use this script, not the single-chunk ones, to evaluate whether a
quantized checkpoint is safe at long context.** The single-chunk scripts
are kept because they're still useful for other things (e.g. localizing
where a genuine divergence originates, once you have one that survives
chunked testing) but will show false failures on their own if used to
judge long-context quality.

Usage:
    .venv/bin/python -m reap_stream.test_chunked_prefill_divergence \
        --teacher models/Ling-3.0-flash \
        --quantized artifacts/ling3-8fixed-5routed \
        --target-tokens 16000 --chunk-size 1024 --n-buckets 8 --layers-at-once 3
"""
from __future__ import annotations

import argparse
import gc
import random
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache

from .collect_ling3 import _free_layer, _text_model, load_lazy
from .test_long_context_divergence import _build_long_document, _kl_from_full


def _run_chunked(text, lm_head, tokens: list[int], chunk_size: int,
                  layers_at_once: int, free_as_we_go: bool):
    """Chunked-prefill forward: real KVCache/ArraysCache carried between
    chunks, one layer window resident at a time. Returns final-norm'd hidden
    states for the full sequence."""
    n_layers = len(text.layers)
    hidden_chunks = []
    for i in range(0, len(tokens), chunk_size):
        h = text.word_embeddings(mx.array(tokens[i:i + chunk_size])[None])
        mx.eval(h)
        hidden_chunks.append(h)

    for w0 in range(0, n_layers, layers_at_once):
        window = range(w0, min(w0 + layers_at_once, n_layers))
        for li in window:
            layer = text.layers[li]
            cache = KVCache() if layer.is_global else ArraysCache(size=4)
            for ci in range(len(hidden_chunks)):
                h = hidden_chunks[ci]
                if layer.is_global:
                    mask = create_attention_mask(h, cache, return_array=True)
                else:
                    mask = create_ssm_mask(h, cache)
                h_out = layer(h, mask=mask, cache=cache)
                mx.eval(h_out)
                hidden_chunks[ci] = h_out
        if free_as_we_go:
            for li in window:
                _free_layer(text, li)
        gc.collect()
        mx.clear_cache()
        act = mx.get_active_memory() / 1e9
        print(f"[chunked] layers {w0}-{window[-1]}/{n_layers-1} active={act:.1f} GB",
              flush=True)

    h_full = mx.concatenate(hidden_chunks, axis=1)
    return text.norm(h_full)


def run(teacher_path: str, quantized_path: str, target_tokens: int, chunk_size: int,
        n_buckets: int, layers_at_once: int, kl_threshold: float, seed: int):
    rng = random.Random(seed)
    repo_root = Path(__file__).resolve().parents[1]

    print("[chunked] tokenizing document...", flush=True)
    _, tok = load(quantized_path, lazy=True)
    doc = _build_long_document(repo_root, target_tokens)
    tokens = tok.encode(doc)
    if isinstance(tokens, dict):
        tokens = tokens["input_ids"]
    tokens = tokens[:target_tokens]
    seq_len = len(tokens)

    bucket_edges = [int(seq_len * i / n_buckets) for i in range(n_buckets + 1)]
    buckets = list(zip(bucket_edges[:-1], bucket_edges[1:]))
    n_per = 8
    positions = []
    for lo, hi in buckets:
        lo = max(lo, 8)
        hi = max(hi, lo + n_per + 1)
        positions += sorted(rng.sample(range(lo, hi), min(n_per, hi - lo)))
    idx_arr = mx.array(positions)
    print(f"[chunked] seq_len={seq_len}, chunk_size={chunk_size}, "
          f"{len(buckets)} buckets, {len(positions)} probe positions", flush=True)

    print(f"[chunked] teacher (streamed, chunked prefill): {teacher_path}", flush=True)
    model, _ = load_lazy(teacher_path)
    text = _text_model(model)
    h_full = _run_chunked(text, model.lm_head, tokens, chunk_size, layers_at_once,
                           free_as_we_go=True)
    teacher_logits = model.lm_head(mx.take(h_full[0], idx_arr, axis=0))
    mx.eval(teacher_logits)
    del model, text, h_full
    gc.collect()
    mx.clear_cache()

    print(f"[chunked] quantized (resident, chunked prefill): {quantized_path}", flush=True)
    from .bailing_swiglu_clamp import apply_bailing_swiglu_clamp
    apply_bailing_swiglu_clamp()
    q_model, _ = load(quantized_path, lazy=False)
    q_text = _text_model(q_model)
    q_h_full = _run_chunked(q_text, q_model.lm_head, tokens, chunk_size, layers_at_once,
                             free_as_we_go=False)
    q_logits = q_model.lm_head(mx.take(q_h_full[0], idx_arr, axis=0))
    mx.eval(q_logits)

    print(f"\n--- chunked prefill (chunk_size={chunk_size}), {quantized_path} ---")
    print("bucket | mean_kl | rate(KL>{:.1f})".format(kl_threshold))
    off = 0
    for lo, hi in buckets:
        n = min(n_per, hi - max(lo, 8))
        kls = [_kl_from_full(teacher_logits[off + i], q_logits[off + i])[0] for i in range(n)]
        bad = sum(1 for k in kls if k > kl_threshold)
        print(f"  {lo:>7}-{hi:<7} | {sum(kls)/n:>7.4f} | {bad}/{n}")
        off += n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--quantized", required=True)
    ap.add_argument("--target-tokens", type=int, default=16000)
    ap.add_argument("--chunk-size", type=int, default=1024)
    ap.add_argument("--n-buckets", type=int, default=8)
    ap.add_argument("--layers-at-once", type=int, default=3)
    ap.add_argument("--kl-threshold", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    run(a.teacher, a.quantized, a.target_tokens, a.chunk_size, a.n_buckets,
        a.layers_at_once, a.kl_threshold, a.seed)
