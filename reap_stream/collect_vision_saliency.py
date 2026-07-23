"""Vision-only REAP saliency: which experts fire on IMAGE-derived tokens?

The normal collector (collect_step3p7.py) builds embeddings with
`text.embed_tokens(...)` and never touches the vision tower -- so experts that
specialise in image tokens get ~0 saliency and would be pruned first. But image
features are `_masked_scatter`'d into inputs_embeds and flow through the SAME
42 MoE layers (step3p7.py:105-111), so those experts definitely exist in the
routing path.

This runs the identical layer-streaming saliency collection, but seeds the
hidden states from the full vision path:
    processor(text, images) -> pixel_values/patch_pixel_values/num_patches
    model.get_input_embeddings(input_ids, pixel_values, ...) -> merged embeds

Output is a saliency map directly comparable to the text-only one, answering:
are vision experts SEPARATE (blind spot -> must protect them from pruning) or
MIXED with text experts (text-only saliency already covers them)?

Usage:
    .venv/bin/python -m reap_stream.collect_vision_saliency \
        --model models/Step-3.7-Flash \
        --dataset calib/multimodal_fixed/multimodal_fixed.jsonl \
        --output artifacts/step37-vision-saliency \
        --max-samples 300 --layers-at-once 2
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image
from mlx_vlm import load

from .collect_step3p7 import (
    _CACHE_EVERY,
    _MoEProbe,
    _free_layer,
    _mem_mb,
    _moe_layer_ids,
    _num_experts,
    _run_layer,
    _text_config,
    _text_model,
)
from .saliency import LayerSaliency


def _load_rows(dataset_file: str, limit: int | None):
    rows = []
    with open(dataset_file) as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def _vision_embed(model, processor, row, root: Path):
    """Run the FULL vision path -> merged input embeddings for one row."""
    img_path = root / row["image"] if not Path(row["image"]).is_absolute() else Path(row["image"])
    image = Image.open(img_path).convert("RGB")
    inputs = processor(text=row["text"], images=[image])

    def _mx(key):
        v = inputs.get(key)
        if v is None:
            return None
        return v if isinstance(v, mx.array) else mx.array(np.asarray(v))

    input_ids = _mx("input_ids")
    if input_ids.ndim == 1:
        input_ids = input_ids[None]
    kwargs = {}
    for k in ("patch_pixel_values", "num_patches"):
        val = inputs.get(k)
        if val is not None:
            kwargs[k] = _mx(k) if k != "num_patches" else list(np.asarray(val).reshape(-1))
    feats = model.get_input_embeddings(input_ids, _mx("pixel_values"), **kwargs)
    return feats.inputs_embeds


def collect(model_path, dataset_file, output_dir, max_samples, layers_at_once):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path.cwd()

    print(f"[vis] loading {model_path} (lazy)", flush=True)
    model, processor = load(model_path, lazy=True)
    text = _text_model(model)
    cfg = _text_config(model)
    sliding_window = getattr(cfg, "sliding_window", None)
    n_layers = len(text.layers)
    moe_ids = _moe_layer_ids(text)
    n_experts = _num_experts(text)
    stats = {i: LayerSaliency(num_experts=n_experts) for i in moe_ids}

    rows = _load_rows(dataset_file, max_samples)
    print(f"[vis] {len(rows)} image rows, {len(moe_ids)} MoE layers, {n_experts} experts",
          flush=True)

    t0 = time.time()
    hidden = []
    for i, row in enumerate(rows):
        h = _vision_embed(model, processor, row, root)
        mx.eval(h)
        hidden.append(h)
        if (i + 1) % 50 == 0:
            print(f"[vis] embedded {i+1}/{len(rows)} (vision tower) "
                  f"active={mx.get_active_memory()/1e9:.1f}GB", flush=True)
        if (i + 1) % _CACHE_EVERY == 0:
            mx.clear_cache()
    print(f"[vis] all images embedded in {time.time()-t0:.0f}s", flush=True)

    # free the vision tower -- it has done its job, the rest is pure decoder
    model.vision_model = nn.Identity()
    gc.collect()
    mx.clear_cache()

    for w0 in range(0, n_layers, layers_at_once):
        window = list(range(w0, min(w0 + layers_at_once, n_layers)))
        for li in window:
            if li in stats:
                text.layers[li].mlp = _MoEProbe(text.layers[li].mlp, li, stats)
        for li in window:
            layer = text.layers[li]
            for i in range(len(hidden)):
                hidden[i] = _run_layer(layer, hidden[i], sliding_window)
                if (i + 1) % _CACHE_EVERY == 0:
                    mx.clear_cache()
            hits = int(stats[li].freq.sum()) if li in stats else 0
            mem = _mem_mb()
            print(f"[vis] layer {li:02d}/{n_layers-1} moe_hits={hits} "
                  f"active_mb={mem['active_mb']:.0f}", flush=True)
        for li in window:
            _free_layer(text, li)
        gc.collect()
        mx.clear_cache()

    sal = {str(k): v.to_dict() for k, v in stats.items()}
    (out / "saliency.json").write_text(json.dumps(sal))
    (out / "run_meta.json").write_text(json.dumps({
        "model": model_path, "dataset": dataset_file, "n_images": len(rows),
        "modality": "vision", "layers_at_once": layers_at_once,
        "elapsed_sec": round(time.time() - t0, 1),
    }, indent=2))
    print(f"[vis] done in {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Step-3.7-Flash")
    ap.add_argument("--dataset", default="calib/multimodal_fixed/multimodal_fixed.jsonl")
    ap.add_argument("--output", default="artifacts/step37-vision-saliency")
    ap.add_argument("--max-samples", type=int, default=300)
    ap.add_argument("--layers-at-once", type=int, default=2)
    a = ap.parse_args()
    collect(a.model, a.dataset, a.output, a.max_samples, a.layers_at_once)
