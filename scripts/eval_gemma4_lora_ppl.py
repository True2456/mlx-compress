"""Held-out perplexity for the Gemma-4-12B agentic LoRA: base vs adapted.

Evaluates on data/lora_gemma4/test.jsonl -- rows never used in training or
validation. Reuses mlx_vlm's own VisionDataset + the exact model-call
convention from trainer/sft_trainer.py's vision_language_loss_fn (confirmed
by reading it, not assumed: gemma4_unified needs attention_mask=None, and
completion_mask -- not a hardcoded assistant_id -- is what actually gates
train_on_completions in this codepath) so the eval matches training math
exactly, just per-row instead of aggregated per-batch.

NOTE: perplexity alone can be misleading -- this project rejected a REAM
merge scheme earlier specifically because its PPL gain turned out to be
smoothing, not real quality. Treat this as one signal, not the verdict.
Pair with scripts/eval_gemma4_humaneval.py for a real functional check.

Usage:
    .venv/bin/python scripts/eval_gemma4_lora_ppl.py \
        --model ~/.lmstudio/models/mlx-community/gemma-4-12B-it-qat-4bit \
        --adapter-path adapters/gemma4-12b-agentic \
        --data data/lora_gemma4/test.jsonl --n 200 --out artifacts/ppl_gemma4_12b.json
"""
from __future__ import annotations

import argparse
import json
import random

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from datasets import load_dataset
from mlx_vlm import load
from mlx_vlm.trainer.datasets import VisionDataset


def _model_type(model):
    config = getattr(model, "config", None)
    if isinstance(config, dict):
        return config.get("model_type")
    return getattr(config, "model_type", None)


def row_nll(model, item, model_type):
    input_ids = item["input_ids"]
    if input_ids.ndim == 1:
        input_ids = input_ids[None]
    attention_mask = item.get("attention_mask")
    if attention_mask is not None and attention_mask.ndim == 1:
        attention_mask = attention_mask[None]
    completion_mask = item.get("completion_mask")
    if completion_mask is not None and completion_mask.ndim == 1:
        completion_mask = completion_mask[None]
    pixel_values = item.get("pixel_values")

    seq_in = input_ids[:, :-1]
    labels = input_ids[:, 1:]
    model_attention_mask = None if model_type == "gemma4_unified" else (
        attention_mask[:, :-1] if attention_mask is not None else None
    )

    outputs = model(seq_in, pixel_values, model_attention_mask)
    logits = outputs.logits.astype(mx.float32)
    if logits.shape[1] != labels.shape[1]:
        logits = logits[:, -labels.shape[1]:, :]

    ce = nn.losses.cross_entropy(logits, labels)  # (1, seq)

    if completion_mask is not None:
        mask = completion_mask[:, 1:]
    else:
        mask = mx.ones_like(labels)
    if attention_mask is not None:
        mask = mask * attention_mask[:, 1:]

    n_tok = float(mx.sum(mask))
    if n_tok == 0:
        return None, 0
    nll = float((ce * mask).sum()) / n_tok
    return nll, n_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--data", default="data/lora_gemma4/test.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    print(f"[eval] loading {a.model} (adapter={a.adapter_path})", flush=True)
    model, processor = load(
        a.model, adapter_path=a.adapter_path, processor_config={"trust_remote_code": True}
    )
    config = model.config.__dict__
    model_type = _model_type(model)

    ds = load_dataset("json", data_files=a.data, split="train")
    idx = list(range(len(ds)))
    random.Random(a.seed).shuffle(idx)
    idx = idx[: a.n]

    vd = VisionDataset(ds, config, processor, train_on_completions=True)

    nlls, weights = [], []
    for j, i in enumerate(idx):
        item = vd[i]
        nll, n_tok = row_nll(model, item, model_type)
        if nll is not None:
            nlls.append(nll)
            weights.append(n_tok)
        if (j + 1) % 25 == 0:
            print(f"  {j+1}/{len(idx)}", flush=True)

    nlls = np.array(nlls)
    weights = np.array(weights)
    mean_nll = float(np.average(nlls, weights=weights))
    ppl = float(np.exp(mean_nll))
    result = {
        "model": a.model,
        "adapter_path": a.adapter_path,
        "n_rows": len(nlls),
        "total_completion_tokens": int(weights.sum()),
        "mean_nll": mean_nll,
        "perplexity": ppl,
    }
    print(json.dumps(result, indent=2))
    if a.out:
        json.dump(result, open(a.out, "w"), indent=2)
        print(f"[eval] wrote -> {a.out}")


if __name__ == "__main__":
    main()
