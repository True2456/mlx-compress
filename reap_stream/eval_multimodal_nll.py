"""Answer-token NLL on image-conditioned prompts, for RESIDENT models.

The one measurement text perplexity structurally cannot make: feed real images
through the vision tower (kept BF16 in every student), merge into the MoE
stream, and score ONLY the answer tokens. Comparing two students that differ
only in their reap plan isolates the plan's effect on vision quality.

Answer span is located by the prefix-property: processor(prefix) token ids are
a strict prefix of processor(full) ids (verified per row), so the answer is
exactly the trailing len(full) - len(prefix) tokens.

The forward pass reuses the same manual layer loop as eval_ppl_streamed's
streamed mode (model here is resident, so no block windowing needed), which
sidesteps any inputs_embeds signature differences across mlx_vlm versions.

Usage:
    .venv/bin/python -m reap_stream.eval_multimodal_nll \
        --model models/Step-3.7-p15-4bit \
        --dataset calib/multimodal_eval/multimodal_eval.jsonl \
        --out artifacts/mmeval-p15-old.json
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image
from mlx_vlm import load

from .collect_step3p7 import _CACHE_EVERY, _run_layer, _text_config, _text_model


def _proc_ids(processor, text, image):
    ids = processor(text=text, images=[image])["input_ids"]
    ids = np.asarray(ids).reshape(-1)
    return ids


def _vision_embed(model, processor, text, image):
    inputs = processor(text=text, images=[image])

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
    return feats.inputs_embeds, np.asarray(input_ids).reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="calib/multimodal_eval/multimodal_eval.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--adapter-path", default=None,
                    help="optional LoRA adapter dir to load on top of the base model")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.dataset) if l.strip()]
    if a.max_rows:
        rows = rows[: a.max_rows]

    from .tiered import maybe_patch_tiered
    if maybe_patch_tiered(a.model):
        print("[mm] tiered-bank model detected, MoE class patched", flush=True)
    print(f"[mm] loading {a.model} (resident)"
          + (f" +adapter {a.adapter_path}" if a.adapter_path else ""), flush=True)
    load_kwargs = {"adapter_path": a.adapter_path} if a.adapter_path else {}
    model, processor = load(a.model, lazy=False, **load_kwargs)
    text = _text_model(model)
    lm = model.language_model
    sliding = getattr(_text_config(model), "sliding_window", None)

    per_cat = defaultdict(lambda: [0.0, 0])
    skipped = 0
    t0 = time.time()
    for i, r in enumerate(rows):
        image = Image.open(r["image"]).convert("RGB")
        full_ids_chk = _proc_ids(processor, r["text"], image)
        prefix_ids = _proc_ids(processor, r["prefix"], image)
        k = len(full_ids_chk) - len(prefix_ids)
        if k <= 0 or not np.array_equal(full_ids_chk[: len(prefix_ids)], prefix_ids):
            skipped += 1
            continue

        h, full_ids = _vision_embed(model, processor, r["text"], image)
        for layer in text.layers:
            h = _run_layer(layer, h, sliding)
        logits = lm.lm_head(text.norm(h))[0]

        # logits[t] predicts token t+1; answer occupies the last k positions
        lg = logits[-k - 1 : -1].astype(mx.float32)
        tgt = mx.array(full_ids[-k:])
        lse = mx.logsumexp(lg, axis=-1)
        picked = mx.take_along_axis(lg, tgt[:, None], axis=-1)[:, 0]
        nll = float((lse - picked).sum().item())

        per_cat[r["category"]][0] += nll
        per_cat[r["category"]][1] += k
        mx.clear_cache()
        if (i + 1) % 25 == 0:
            print(f"[mm] {i+1}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)

    result = {
        "model": a.model,
        "dataset": a.dataset,
        "n_rows": len(rows) - skipped,
        "skipped": skipped,
        "per_category": {},
        "elapsed_sec": time.time() - t0,
    }
    tot_s = tot_n = 0.0
    for cat, (s, n) in sorted(per_cat.items()):
        result["per_category"][cat] = {"nll": s / n, "ppl": float(np.exp(s / n)), "n_tokens": n}
        tot_s += s
        tot_n += n
    result["overall"] = {"nll": tot_s / tot_n, "ppl": float(np.exp(tot_s / tot_n))}
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(f"[mm] overall answer NLL {tot_s/tot_n:.4f} "
          f"({int(tot_n)} answer tokens, {skipped} skipped) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
