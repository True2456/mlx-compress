"""Build a HELD-OUT multimodal eval set (never used in any calibration).

Sources, chosen to be disjoint from every calibration pass in this repo:
  - HuggingFaceM4/ChartQA **test** split (all calib/saliency passes used train)
  - merve/vqav2-small validation[1000:] (vision saliency used validation[:300])

Image + query + answer are read from the SAME dataset row in one pass
(the build_calib_mix.py desync bug cannot occur here). Rows carry an extra
"answer" field so the evaluator can locate the answer span exactly.

Usage:
    .venv/bin/python scripts/build_multimodal_eval_set.py \
        --n-chart 150 --n-natural 100 --out-dir calib/multimodal_eval
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _rows_from(ds, n, img_dir, prefix, source, category):
    rows = []
    for i in range(len(ds)):
        if len(rows) >= n:
            break
        ex = ds[i]
        query = str(ex.get("query") or ex.get("question") or "").strip()
        label = ex.get("label") if ex.get("label") is not None else ex.get("multiple_choice_answer")
        answer = ", ".join(str(x) for x in label) if isinstance(label, list) else str(label or "").strip()
        image = ex.get("image")
        if not query or not answer or image is None:
            continue
        img_path = img_dir / f"{prefix}_{len(rows):04d}.png"
        image.convert("RGB").save(img_path)
        rows.append({
            "text": f"<im_patch>\nUSER:\n{query}\n\nASSISTANT:\n{answer}",
            "prefix": f"<im_patch>\nUSER:\n{query}\n\nASSISTANT:\n",
            "answer": answer,
            "category": category,
            "image": str(img_path),
            "source": source,
            "orig_index": i,
        })
    return rows


def build(n_chart: int, n_natural: int, out_dir: str):
    from datasets import load_dataset

    out = Path(out_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"[build] ChartQA test[:{n_chart*2}] (held out: calib used train)")
    ds = load_dataset("HuggingFaceM4/ChartQA", split=f"test[:{n_chart*2}]")
    rows = _rows_from(ds, n_chart, img_dir, "chartqa_test", "HuggingFaceM4/ChartQA:test", "chartqa")

    print(f"[build] vqav2-small validation[1000:{1000+n_natural*3}] (saliency used [:300])")
    ds = load_dataset("merve/vqav2-small", split=f"validation[1000:{1000+n_natural*3}]")
    rows += _rows_from(ds, n_natural, img_dir, "vqav2_val", "merve/vqav2-small:validation+1000", "vqa_natural")

    out_file = out / "multimodal_eval.jsonl"
    with out_file.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[build] wrote {len(rows)} rows -> {out_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-chart", type=int, default=150)
    ap.add_argument("--n-natural", type=int, default=100)
    ap.add_argument("--out-dir", default="calib/multimodal_eval")
    a = ap.parse_args()
    build(a.n_chart, a.n_natural, a.out_dir)
