"""Build a correctly-paired multimodal calib set from HuggingFaceM4/ChartQA.

Fixes the bug in build_calib_mix.py's build_chartqa_rows(), which paired local
PNGs (sorted by filename) with a freshly re-loaded dataset by raw index --
silently desyncing image and Q&A whenever the local PNGs weren't saved from
that exact same load. Here, image + query + answer are all read from the SAME
row in one pass, so they cannot desync.

Text is formatted with a literal "<im_patch>" placeholder, as required by
Step3VLProcessor.__call__ (it splits on this token and injects the real
patch/image-token sequence in its place).

Usage:
    .venv/bin/python scripts/build_multimodal_calib_fixed.py \
        --n 300 --out-dir calib/multimodal_fixed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build(n: int, out_dir: str, seed: int):
    from datasets import load_dataset

    out = Path(out_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"[build] loading HuggingFaceM4/ChartQA train[:{n*2}] (headroom for skips)")
    ds = load_dataset("HuggingFaceM4/ChartQA", split=f"train[:{n*2}]")

    rows = []
    for i in range(len(ds)):
        if len(rows) >= n:
            break
        ex = ds[i]
        query = str(ex.get("query") or "").strip()
        label = ex.get("label")
        answer = ", ".join(str(x) for x in label) if isinstance(label, list) else str(label or "").strip()
        image = ex.get("image")
        if not query or not answer or image is None:
            continue

        img_path = img_dir / f"chartqa_fixed_{len(rows):04d}.png"
        image.convert("RGB").save(img_path)

        text = f"<im_patch>\nUSER:\n{query}\n\nASSISTANT:\n{answer}"
        rows.append({
            "text": text,
            "category": "multimodal",
            "image": str(img_path.relative_to(Path.cwd())) if img_path.is_absolute() else str(img_path),
            "source": "HuggingFaceM4/ChartQA",
            "orig_index": i,
        })

    out_file = out / "multimodal_fixed.jsonl"
    with out_file.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[build] wrote {len(rows)} correctly-paired rows -> {out_file}")
    print(f"[build] images -> {img_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out-dir", default="calib/multimodal_fixed")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a.n, a.out_dir, a.seed)
