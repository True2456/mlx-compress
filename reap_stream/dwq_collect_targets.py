"""DWQ phase 1: stream the (BF16) teacher block-by-block on one Mac and cache
top-k logit targets for each calib prompt.

Never holds the full teacher resident — reuses the layerwise streaming pattern
from collect_step3p7 (load a window of blocks, push all prompts through, free
the window). At the end applies final norm + lm_head, keeps only the top-k
logits per position, and writes one safetensors per prompt.

Phase 2 (dwq_train_student) reloads these and trains the 4-bit student's affine
scales to match, via KL over the cached top-k.

Usage:
    .venv/bin/python -m reap_stream.dwq_collect_targets \
        --teacher models/Step-3.7-Flash \
        --dataset calib/cloud_reap_8k.jsonl \
        --out artifacts/dwq-targets \
        --n-prompts 1500 --max-tokens 384 --topk 128 --layers-at-once 3
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm import load

from .collect_step3p7 import (
    _free_layer,
    _run_layer,
    _text_config,
    _text_model,
    _tokenize_prompts,
)
from .dataset import load_prompt_texts


def collect_targets(
    teacher_path: str,
    dataset_file: str,
    out_dir: str,
    n_prompts: int,
    max_tokens: int,
    topk: int,
    layers_at_once: int,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[dwq-collect] loading teacher (lazy): {teacher_path}", flush=True)
    model, processor = load(teacher_path, lazy=True)
    text = _text_model(model)
    lm_head = model.language_model.lm_head
    final_norm = text.norm
    cfg = _text_config(model)
    sliding_window = getattr(cfg, "sliding_window", None)
    n_layers = len(text.layers)

    prompts = load_prompt_texts(dataset_file, limit=n_prompts)
    token_batches = _tokenize_prompts(processor, prompts, max_tokens)
    print(f"[dwq-collect] {len(token_batches)} prompts, {n_layers} layers, "
          f"topk={topk}, layers_at_once={layers_at_once}", flush=True)

    # keep the input tokens so phase 2 feeds the student identical sequences
    input_ids = [mx.array(t)[None] for t in token_batches]

    t0 = time.time()
    hidden = []
    for tok in input_ids:
        h = text.embed_tokens(tok)
        mx.eval(h)
        hidden.append(h)

    # stream all decoder blocks, windowed; free each window after use
    for w0 in range(0, n_layers, layers_at_once):
        window = range(w0, min(w0 + layers_at_once, n_layers))
        for li in window:
            layer = text.layers[li]
            hidden = [_run_layer(layer, h, sliding_window) for h in hidden]
        for li in window:
            _free_layer(text, li)
        gc.collect()
        mx.clear_cache()
        act = mx.get_active_memory() / 1e9
        print(f"[dwq-collect] blocks {w0}-{window[-1]}/{n_layers-1} "
              f"active={act:.1f} GB", flush=True)

    # final norm + lm_head -> top-k per position; write one file per prompt
    print("[dwq-collect] projecting to logits + writing top-k targets", flush=True)
    for i in range(len(hidden)):
        h = final_norm(hidden[i])            # [1, seq, d]
        logits = lm_head(h)[0]               # [seq, vocab]
        vals, idx = mx.top_k_with_indices(logits, topk, axis=-1) \
            if hasattr(mx, "top_k_with_indices") else _topk(logits, topk)
        mx.eval(vals, idx)
        mx.save_safetensors(
            str(out / f"{i:05d}.safetensors"),
            {
                "input_ids": input_ids[i][0].astype(mx.int32),
                "topk_vals": vals.astype(mx.float16),
                "topk_idx": idx.astype(mx.int32),
            },
        )
        hidden[i] = None                     # free as we go
        if (i + 1) % 100 == 0:
            print(f"[dwq-collect] wrote {i+1}/{len(hidden)}", flush=True)

    meta = {
        "teacher": teacher_path,
        "dataset": dataset_file,
        "n_prompts": len(input_ids),
        "max_tokens": max_tokens,
        "topk": topk,
        "vocab_size": int(cfg.vocab_size),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out / "targets_meta.json").write_text(__import__("json").dumps(meta, indent=2))
    print(f"[dwq-collect] done in {meta['elapsed_sec']}s -> {out}", flush=True)


def _topk(logits: mx.array, k: int):
    """Fallback top-k (values + indices) along last axis."""
    idx = mx.argpartition(-logits, k, axis=-1)[..., :k]
    vals = mx.take_along_axis(logits, idx, axis=-1)
    # sort the k by value desc for determinism
    order = mx.argsort(-vals, axis=-1)
    return mx.take_along_axis(vals, order, axis=-1), mx.take_along_axis(idx, order, axis=-1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-prompts", type=int, default=1500)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--topk", type=int, default=128)
    ap.add_argument("--layers-at-once", type=int, default=3)
    a = ap.parse_args()
    collect_targets(a.teacher, a.dataset, a.out, a.n_prompts, a.max_tokens,
                    a.topk, a.layers_at_once)
