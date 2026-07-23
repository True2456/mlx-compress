#!/usr/bin/env python3
"""Build cloud REAP calibration mix: official Cerebras agentic recipe + ChartQA.

Reads the local Cerebras 6-way mix (calib/cerebras_reap_mix.jsonl) when present,
subsamples unique rows into category targets, seasons with ChartQA multimodal
rows paired to local PNGs under calib/multimodal_images/, and writes
calib/cloud_reap_8k.jsonl.

No duplicate padding. Undershooting a bucket is preferred over cycling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

# Allow `python scripts/build_calib_mix.py` to import sibling builders
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CEREBRAS = ROOT / "calib" / "cerebras_reap_mix.jsonl"
DEFAULT_OUT = ROOT / "calib" / "cloud_reap_8k.jsonl"
DEFAULT_IMAGES = ROOT / "calib" / "multimodal_images"
MIN_CHARS = 200

# Target composition (~8k unique; multimodal capped by available local PNGs)
CATEGORY_TARGETS: dict[str, int] = {
    "coding": 2000,  # ~25%
    "tool_use": 1600,  # ~20%
    "agentic": 1600,  # ~20%
    "reasoning_math": 1200,  # ~15%
    "general_instruction": 800,  # ~10% code-reasoning / MoT code
    "multimodal": 800,  # ~10%; will cap at available ChartQA+PNG pairs
}

SOURCE_TO_CATEGORY: dict[str, str] = {
    "theblackcat102/evol-codealpaca-v1": "coding",
    "NobodyExistsOnTheInternet/xlam-function-calling-60k": "tool_use",
    "Salesforce/xlam-function-calling-60k": "tool_use",
    "SWE-bench/SWE-smith-trajectories:tool": "agentic",
    "open-r1/Mixture-of-Thoughts/math": "reasoning_math",
    "open-r1/Mixture-of-Thoughts/science": "reasoning_math",
    "open-r1/Mixture-of-Thoughts/code": "general_instruction",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def categorize_source(source: str) -> Optional[str]:
    if source in SOURCE_TO_CATEGORY:
        return SOURCE_TO_CATEGORY[source]
    # Fuzzy fallbacks for slight naming drift
    s = source.lower()
    if "evol-codealpaca" in s:
        return "coding"
    if "xlam" in s:
        return "tool_use"
    if "swe-smith" in s or "swe_smith" in s:
        return "agentic"
    if "mixture-of-thoughts" in s and "/math" in s:
        return "reasoning_math"
    if "mixture-of-thoughts" in s and "/science" in s:
        return "reasoning_math"
    if "mixture-of-thoughts" in s and "/code" in s:
        return "general_instruction"
    return None


def load_cerebras_pools(
    path: Path, min_chars: int
) -> dict[str, list[dict[str, Any]]]:
    pools: dict[str, list[dict[str, Any]]] = {c: [] for c in CATEGORY_TARGETS if c != "multimodal"}
    seen: set[str] = set()
    skipped_short = 0
    skipped_dup = 0
    skipped_unknown = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = str(row.get("text") or "").strip()
            if len(text) < min_chars:
                skipped_short += 1
                continue
            h = text_hash(text)
            if h in seen:
                skipped_dup += 1
                continue
            cat = categorize_source(str(row.get("source") or ""))
            if cat is None or cat not in pools:
                skipped_unknown += 1
                continue
            seen.add(h)
            pools[cat].append(
                {
                    "text": text,
                    "category": cat,
                    "source": row.get("source"),
                }
            )

    log(
        f"Cerebras pools from {path}: "
        + ", ".join(f"{k}={len(v)}" for k, v in pools.items())
        + f" (skipped short={skipped_short}, dup={skipped_dup}, unknown={skipped_unknown})"
    )
    return pools


def list_local_chartqa_images(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        return []
    imgs = sorted(image_dir.glob("chartqa_*.png"))
    # Prefer numeric order chartqa_0000.png …
    def key(p: Path) -> tuple[int, str]:
        m = re.search(r"chartqa_(\d+)", p.stem)
        return (int(m.group(1)) if m else 10**9, p.name)

    return sorted(imgs, key=key)


def build_chartqa_rows(
    image_dir: Path,
    max_n: int,
    min_chars: int,
    seed: int,
    repo_root: Path,
) -> list[dict[str, Any]]:
    """Pair HuggingFaceM4/ChartQA train Q/A with local chartqa_XXXX.png by index."""
    images = list_local_chartqa_images(image_dir)
    if not images:
        log(f"WARNING: no local ChartQA PNGs in {image_dir}")
        return []

    cap = min(max_n, len(images))
    log(f"Loading ChartQA train from HF (pairing first {cap} with local PNGs)…")
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceM4/ChartQA", split="train")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    # Deterministic subsample of indices if we ever have more images than cap
    rng = random.Random(seed + 99)
    indices = list(range(min(len(ds), len(images))))
    if len(indices) > cap:
        indices = sorted(rng.sample(indices, cap))
    else:
        indices = indices[:cap]

    for i in indices:
        ex = ds[i]
        query = str(ex.get("query") or "").strip()
        label = ex.get("label")
        if isinstance(label, list):
            answer = ", ".join(str(x) for x in label)
        else:
            answer = str(label or "").strip()
        if not query or not answer:
            continue
        text = f"USER:\n{query}\n\nASSISTANT:\n{answer}".strip()
        if len(text) < min_chars:
            # ChartQA answers are often short; keep if still reasonably formed
            # Prefer keeping real multimodal seasoning over dropping for length.
            if len(text) < 40:
                continue
        h = text_hash(text)
        if h in seen:
            continue
        seen.add(h)
        img_path = images[i]
        rel = str(img_path.relative_to(repo_root))
        rows.append(
            {
                "text": text,
                "category": "multimodal",
                "image": rel,
                "source": "HuggingFaceM4/ChartQA",
            }
        )

    log(f"ChartQA multimodal rows: {len(rows)} (local PNGs={len(images)}, cap={cap})")
    return rows


def subsample_unique(
    pools: dict[str, list[dict[str, Any]]],
    multimodal: list[dict[str, Any]],
    targets: dict[str, int],
    seed: int,
    min_chars: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    global_seen: set[str] = set()
    final: list[dict[str, Any]] = []

    def take(cat: str, items: list[dict[str, Any]], n: int) -> int:
        eligible = []
        for item in items:
            text = item["text"].strip()
            # Multimodal may be short; only enforce min_chars for text-only cats
            if cat != "multimodal" and len(text) < min_chars:
                continue
            h = text_hash(text)
            if h in global_seen:
                continue
            eligible.append(item)
        rng.shuffle(eligible)
        picked = eligible[:n]
        for item in picked:
            global_seen.add(text_hash(item["text"].strip()))
            out = {"text": item["text"].strip(), "category": cat}
            if item.get("image"):
                out["image"] = item["image"]
            final.append(out)
        if len(picked) < n:
            log(f"  NOTE: {cat} undershot target {n} -> got {len(picked)} (no padding)")
        else:
            log(f"  OK: {cat} {len(picked)}/{n}")
        return len(picked)

    for cat, n in targets.items():
        if cat == "multimodal":
            take(cat, multimodal, n)
        else:
            take(cat, pools.get(cat, []), n)

    rng.shuffle(final)
    return final


def audit(rows: list[dict[str, Any]], repo_root: Path) -> dict[str, Any]:
    texts = [r["text"] for r in rows]
    unique = len(set(texts))
    cats = Counter(r["category"] for r in rows)
    lengths = [len(t) for t in texts]
    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)
    p50 = lengths_sorted[n // 2] if n else 0
    mean = sum(lengths) / n if n else 0

    mm = [r for r in rows if r.get("category") == "multimodal"]
    mm_with_image = 0
    mm_image_exists = 0
    for r in mm:
        img = r.get("image")
        if img:
            mm_with_image += 1
            p = Path(img)
            if not p.is_absolute():
                p = repo_root / p
            if p.is_file():
                mm_image_exists += 1

    filler_re = re.compile(r"\[CATEGORY TASK #", re.I)
    synthetic = sum(1 for t in texts if filler_re.search(t))

    report = {
        "total": len(rows),
        "unique_texts": unique,
        "dup_count": len(rows) - unique,
        "category_counts": dict(cats),
        "length": {
            "min": min(lengths) if lengths else 0,
            "p50": p50,
            "mean": round(mean, 1),
            "max": max(lengths) if lengths else 0,
        },
        "multimodal": {
            "count": len(mm),
            "with_image_field": mm_with_image,
            "image_path_exists": mm_image_exists,
        },
        "synthetic_fillers": synthetic,
    }
    return report


def print_audit(report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    log("=" * 72)
    log("AUDIT")
    log(f"  total rows     : {report['total']}")
    log(f"  unique texts   : {report['unique_texts']}")
    log(f"  dup count      : {report['dup_count']}")
    log(f"  categories     : {report['category_counts']}")
    ln = report["length"]
    log(f"  length min/p50/mean/max : {ln['min']} / {ln['p50']} / {ln['mean']} / {ln['max']}")
    mm = report["multimodal"]
    log(
        f"  multimodal     : {mm['count']} "
        f"(image field={mm['with_image_field']}, path exists={mm['image_path_exists']})"
    )
    log(f"  synthetic fillers : {report['synthetic_fillers']}")
    log("-" * 72)
    log("Sample (1 per category, truncated):")
    seen_cats: set[str] = set()
    for r in rows:
        cat = r["category"]
        if cat in seen_cats:
            continue
        seen_cats.add(cat)
        snippet = r["text"].replace("\n", "\\n")[:180]
        extra = f" image={r.get('image')}" if r.get("image") else ""
        log(f"  [{cat}]{extra} {snippet}…")
    log("=" * 72)


def build(
    cerebras_path: Path,
    out_path: Path,
    image_dir: Path,
    seed: int,
    min_chars: int,
    rebuild_cerebras: bool,
) -> dict[str, Any]:
    if rebuild_cerebras or not cerebras_path.is_file():
        log("Cerebras mix missing or --rebuild-cerebras set; building via HF…")
        from build_cerebras_calib_mix import build as build_cerebras

        build_cerebras(cerebras_path, seed=seed, per_source=4096, max_chars=16000)

    if not cerebras_path.is_file():
        raise SystemExit(f"Missing Cerebras mix: {cerebras_path}")

    pools = load_cerebras_pools(cerebras_path, min_chars=min_chars)
    multimodal = build_chartqa_rows(
        image_dir=image_dir,
        max_n=CATEGORY_TARGETS["multimodal"],
        min_chars=min_chars,
        seed=seed,
        repo_root=ROOT,
    )

    rows = subsample_unique(
        pools=pools,
        multimodal=multimodal,
        targets=CATEGORY_TARGETS,
        seed=seed,
        min_chars=min_chars,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = audit(rows, ROOT)
    print_audit(report, rows)
    log(f"Wrote {len(rows)} -> {out_path}")

    def rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(ROOT))
        except ValueError:
            return str(p)

    manifest = {
        "output": rel(out_path),
        "recipe": "cerebras_agentic_6way_subsample + ChartQA multimodal",
        "seed": seed,
        "min_chars": min_chars,
        "targets": CATEGORY_TARGETS,
        "cerebras_source": rel(cerebras_path),
        **report,
    }
    manifest_path = out_path.resolve().with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"Wrote manifest -> {manifest_path}")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cerebras", type=Path, default=DEFAULT_CEREBRAS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-chars", type=int, default=MIN_CHARS)
    p.add_argument(
        "--rebuild-cerebras",
        action="store_true",
        help="Force rebuild calib/cerebras_reap_mix.jsonl from HF before subsampling",
    )
    args = p.parse_args()
    build(
        cerebras_path=args.cerebras,
        out_path=args.out,
        image_dir=args.images,
        seed=args.seed,
        min_chars=args.min_chars,
        rebuild_cerebras=args.rebuild_cerebras,
    )


if __name__ == "__main__":
    main()
