"""Build a MIXED multimodal calib set: synthetic charts + natural photos.

Why mixed: ChartQA alone is a narrow slice of "vision" (plots/axes/legends).
Natural photographs exercise different visual features (objects, scenes,
textures), so a vision-only saliency pass built on charts alone could miss
vision experts that only fire on natural imagery -- and would risk a
falsely-reassuring "MIXED" verdict.

Default composition: 200 ChartQA + 100 VQAv2 (natural images) = 300.

Image + question + answer are always read from the SAME dataset row in one
pass, so they cannot desync (the bug in build_calib_mix.py's build_chartqa_rows,
which paired local PNGs to a re-loaded dataset by raw index).

Text carries the literal "<im_patch>" placeholder required by
Step3VLProcessor.__call__ (it splits on this token, one per image).

Usage:
    .venv/bin/python scripts/build_multimodal_calib_mixed.py \
        --n-chart 200 --n-natural 100 --out-dir calib/multimodal_mixed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _rows_from(ds, n, img_dir, prefix, q_key, a_key, source, start_idx):
    rows, i = [], 0
    while len(rows) < n and i < len(ds):
        ex = ds[i]
        i += 1
        q = str(ex.get(q_key) or "").strip()
        a = ex.get(a_key)
        a = ", ".join(str(x) for x in a) if isinstance(a, list) else str(a or "").strip()
        img = ex.get("image")
        if not q or not a or img is None:
            continue
        idx = start_idx + len(rows)
        p = img_dir / f"{prefix}_{idx:04d}.png"
        img.convert("RGB").save(p)
        rows.append({
            "text": f"<im_patch>\nUSER:\n{q}\n\nASSISTANT:\n{a}",
            "category": "multimodal",
            "subtype": prefix,
            "image": str(p),
            "source": source,
            "orig_index": i - 1,
        })
    return rows


def build(n_chart, n_natural, out_dir):
    from datasets import load_dataset

    out = Path(out_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    print(f"[build] ChartQA (synthetic plots): target {n_chart}")
    ds = load_dataset("HuggingFaceM4/ChartQA", split=f"train[:{max(n_chart * 2, 10)}]")
    rows += _rows_from(ds, n_chart, img_dir, "chart", "query", "label",
                       "HuggingFaceM4/ChartQA", 0)
    print(f"[build]   got {len(rows)}")

    print(f"[build] VQAv2 (natural photos): target {n_natural}")
    ds = load_dataset("merve/vqav2-small", split=f"validation[:{max(n_natural * 3, 10)}]")
    nat = _rows_from(ds, n_natural, img_dir, "natural", "question",
                     "multiple_choice_answer", "merve/vqav2-small", 0)
    rows += nat
    print(f"[build]   got {len(nat)}")

    out_file = out / "multimodal_mixed.jsonl"
    with out_file.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n_c = sum(1 for r in rows if r["subtype"] == "chart")
    n_n = sum(1 for r in rows if r["subtype"] == "natural")
    print(f"[build] wrote {len(rows)} rows ({n_c} chart / {n_n} natural) -> {out_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-chart", type=int, default=200)
    ap.add_argument("--n-natural", type=int, default=100)
    ap.add_argument("--out-dir", default="calib/multimodal_mixed")
    a = ap.parse_args()
    build(a.n_chart, a.n_natural, a.out_dir)
