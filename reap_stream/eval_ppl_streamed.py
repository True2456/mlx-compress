"""Perplexity evaluation for Step-3.7, with per-category breakdown.

Two modes, same math, so numbers are directly comparable:
  --stream : BF16 (or any too-large-to-fit model) evaluated by streaming decoder
             blocks one window at a time (~34GB peak). This is the zero-loss
             reference baseline.
  default  : resident model (quantized students) -- normal forward pass.

Per-category reporting matters: an aggregate PPL can hide "coding fine,
agentic degraded 5%", and both the vision and truncation findings landed
hardest on agentic.

Held-out data: rows 5000+ of cloud_reap_8k were never used for calibration
(saliency used the first 2500, DWQ the first 1500). Multimodal rows are
EXCLUDED -- they are the misaligned/imageless ones and are unanswerable as
text, which would pollute the comparison.

Usage:
    # BF16 reference (streamed, ~17 min for 500 prompts)
    .venv/bin/python -m reap_stream.eval_ppl_streamed \
        --model models/Step-3.7-Flash --stream \
        --out artifacts/ppl-bf16.json --n-prompts 500

    # resident quantized student (fast)
    .venv/bin/python -m reap_stream.eval_ppl_streamed \
        --model models/Step-3.7-p15-4bit \
        --out artifacts/ppl-p15-4bit.json --n-prompts 500
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm import load

from .collect_step3p7 import (
    _CACHE_EVERY,
    _free_layer,
    _run_layer,
    _text_config,
    _text_model,
    _truncate,
)

SKIP_CATEGORIES = {"multimodal"}   # misaligned + imageless -> unanswerable as text


def load_rows(dataset, start, n, max_tokens, truncation, tokenizer, raw_text=True):
    """Held-out rows with their category, tokenized."""
    out = []
    with open(dataset) as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("category") in SKIP_CATEGORIES:
                continue
            text = rec.get("text")
            if not text:
                continue
            if raw_text:
                # The rows already carry their own SYSTEM:/USER:/ASSISTANT:
                # structure. Wrapping them in apply_chat_template nests that
                # inside a user turn, producing input the model never saw in
                # training -- measured PPL 221 vs 8 on agentic prompts.
                templated = text
            else:
                try:
                    templated = tokenizer.apply_chat_template(
                        [{"role": "user", "content": text}],
                        tokenize=False, add_generation_prompt=True)
                except Exception:
                    templated = text
            toks = tokenizer.encode(templated)
            if isinstance(toks, dict):
                toks = toks["input_ids"]
            toks = _truncate(list(toks), max_tokens, truncation)
            if len(toks) < 16:      # too short to score meaningfully
                continue
            out.append({"category": rec.get("category", "?"), "tokens": toks})
            if len(out) >= n:
                break
    return out


def _nll_from_logits(logits, tokens):
    """Sum NLL and token count for next-token prediction. logits: [seq, vocab]."""
    tgt = mx.array(tokens[1:])
    lg = logits[:-1].astype(mx.float32)
    lse = mx.logsumexp(lg, axis=-1)
    picked = mx.take_along_axis(lg, tgt[:, None], axis=-1)[:, 0]
    nll = (lse - picked)
    return float(nll.sum().item()), int(tgt.shape[0])


def eval_resident(model, rows):
    lm = model.language_model
    per_cat = defaultdict(lambda: [0.0, 0])
    for i, r in enumerate(rows):
        out = lm(input_ids=mx.array(r["tokens"])[None])
        s, n = _nll_from_logits(out.logits[0], r["tokens"])
        per_cat[r["category"]][0] += s
        per_cat[r["category"]][1] += n
        if (i + 1) % _CACHE_EVERY == 0:
            mx.clear_cache()
        if (i + 1) % 100 == 0:
            print(f"[ppl] {i+1}/{len(rows)}", flush=True)
    return per_cat


def eval_streamed(model, rows, layers_at_once):
    """Stream decoder blocks; carry hidden states for all prompts across windows."""
    text = _text_model(model)
    lm = model.language_model
    cfg = _text_config(model)
    sliding = getattr(cfg, "sliding_window", None)
    n_layers = len(text.layers)

    hidden = []
    for r in rows:
        h = text.embed_tokens(mx.array(r["tokens"])[None])
        mx.eval(h)
        hidden.append(h)
    print(f"[ppl] embedded {len(hidden)} prompts", flush=True)

    for w0 in range(0, n_layers, layers_at_once):
        window = range(w0, min(w0 + layers_at_once, n_layers))
        for li in window:
            layer = text.layers[li]
            for i in range(len(hidden)):
                hidden[i] = _run_layer(layer, hidden[i], sliding)
                if (i + 1) % _CACHE_EVERY == 0:
                    mx.clear_cache()
        for li in window:
            _free_layer(text, li)
        gc.collect()
        mx.clear_cache()
        print(f"[ppl] blocks {w0}-{list(window)[-1]}/{n_layers-1} "
              f"active={mx.get_active_memory()/1e9:.1f}GB", flush=True)

    per_cat = defaultdict(lambda: [0.0, 0])
    for i, r in enumerate(rows):
        h = text.norm(hidden[i])
        logits = lm.lm_head(h)[0]
        s, n = _nll_from_logits(logits, r["tokens"])
        per_cat[r["category"]][0] += s
        per_cat[r["category"]][1] += n
        hidden[i] = None
        if (i + 1) % _CACHE_EVERY == 0:
            mx.clear_cache()
    return per_cat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="calib/cloud_reap_8k.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-prompts", type=int, default=500)
    ap.add_argument("--start-row", type=int, default=5000, help="held-out offset")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--truncation", default="head")
    ap.add_argument("--chat-template", action="store_true",
                    help="re-wrap rows in the chat template (they already carry roles)")
    ap.add_argument("--stream", action="store_true",
                    help="stream decoder blocks (for BF16 / too-large models)")
    ap.add_argument("--layers-at-once", type=int, default=2)
    ap.add_argument("--adapter-path", default=None,
                    help="optional LoRA adapter dir to load on top of the base "
                         "model (forgetting check vs the un-adapted baseline)")
    a = ap.parse_args()

    t0 = time.time()
    from .tiered import maybe_patch_tiered
    if maybe_patch_tiered(a.model):
        print("[ppl] tiered-bank model detected, MoE class patched", flush=True)
    print(f"[ppl] loading {a.model} (lazy={a.stream})"
          + (f" +adapter {a.adapter_path}" if a.adapter_path else ""), flush=True)
    load_kwargs = {"adapter_path": a.adapter_path} if a.adapter_path else {}
    model, processor = load(a.model, lazy=a.stream, **load_kwargs)
    tok = getattr(processor, "tokenizer", processor)

    rows = load_rows(a.dataset, a.start_row, a.n_prompts, a.max_tokens,
                     a.truncation, tok, raw_text=not a.chat_template)
    ntok = sum(len(r["tokens"]) for r in rows)
    print(f"[ppl] {len(rows)} held-out prompts, {ntok/1e3:.0f}k tokens "
          f"(rows {a.start_row}+, multimodal excluded)", flush=True)

    per_cat = (eval_streamed(model, rows, a.layers_at_once) if a.stream
               else eval_resident(model, rows))

    tot_nll = sum(v[0] for v in per_cat.values())
    tot_n = sum(v[1] for v in per_cat.values())
    res = {
        "model": a.model, "streamed": a.stream, "n_prompts": len(rows),
        "n_tokens": tot_n, "start_row": a.start_row,
        "max_tokens": a.max_tokens, "truncation": a.truncation,
        "overall": {"nll": tot_nll / tot_n, "ppl": math.exp(tot_nll / tot_n)},
        "per_category": {
            c: {"nll": s / n, "ppl": math.exp(s / n), "n_tokens": n}
            for c, (s, n) in sorted(per_cat.items())
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    Path(a.out).write_text(json.dumps(res, indent=2))

    print(f"\n{'category':22s} {'PPL':>10} {'NLL':>8} {'tokens':>10}")
    for c, d in res["per_category"].items():
        print(f"{c:22s} {d['ppl']:>10.4f} {d['nll']:>8.4f} {d['n_tokens']:>10d}")
    print(f"{'OVERALL':22s} {res['overall']['ppl']:>10.4f} "
          f"{res['overall']['nll']:>8.4f} {tot_n:>10d}")
    print(f"[ppl] done in {res['elapsed_sec']}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
