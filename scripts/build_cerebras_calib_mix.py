#!/usr/bin/env python3
"""Build the Cerebras REAP agentic 6-way calibration mix (24,576 rows).

Recipe from CerebrasResearch/reap:
  evol-codealpaca-v1:4096
  xlam-function-calling-60k:4096  (ungated mirror)
  Mixture-of-Thoughts[code|math|science]:4096 each
  SWE-smith-trajectories(tool):4096
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "calib" / "cerebras_reap_mix.jsonl"
PER_SOURCE = 4096


def log(msg: str) -> None:
    print(msg, flush=True)


def messages_to_text(messages: Any, max_chars: int) -> str:
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except Exception:
            return messages[:max_chars]
    if not isinstance(messages, list):
        return str(messages)[:max_chars]
    parts: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "unknown")).upper()
        content = m.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            )
        content = str(content).strip()
        if content:
            parts.append(f"{role}:\n{content}")
    return "\n\n".join(parts)[:max_chars]


def sample_stream(
    name: str,
    split: str,
    n: int,
    text_fn,
    seed: int,
    subset: Optional[str] = None,
    max_scan: int = 30_000,
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    if subset:
        ds = load_dataset(name, subset, split=split, streaming=True)
        source = f"{name}/{subset}"
    else:
        ds = load_dataset(name, split=split, streaming=True)
        source = f"{name}:{split}" if split not in ("train",) else name

    rng = random.Random(seed)
    chosen: list[dict[str, Any]] = []
    seen = 0
    kept = 0
    # Reservoir sample of size n
    for ex in ds:
        seen += 1
        try:
            text = text_fn(ex)
        except Exception:
            continue
        if not text or len(str(text).strip()) < 32:
            continue
        kept += 1
        item = {"text": str(text).strip(), "source": source}
        if len(chosen) < n:
            chosen.append(item)
        else:
            j = rng.randint(1, kept)
            if j <= n:
                chosen[j - 1] = item
        if seen >= max_scan and len(chosen) >= n:
            break
        if seen % 5000 == 0:
            log(f"  … {source}: scanned {seen}, kept_pool={kept}, filled={len(chosen)}/{n}")

    log(f"OK {source}: {len(chosen)}/{n} (scanned={seen}, eligible={kept})")
    if len(chosen) < n:
        log(f"WARNING: short on {source}")
    return chosen


def build(out_path: Path, seed: int, per_source: int, max_chars: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    log("1/6 evol-codealpaca")
    rows += sample_stream(
        "theblackcat102/evol-codealpaca-v1",
        "train",
        per_source,
        lambda ex: f"USER:\n{ex.get('instruction','')}\n\nASSISTANT:\n{ex.get('output','')}"[:max_chars],
        seed + 1,
    )

    log("2/6 xlam (ungated mirror)")
    rows += sample_stream(
        "NobodyExistsOnTheInternet/xlam-function-calling-60k",
        "train",
        per_source,
        lambda ex: (
            f"USER:\n{ex.get('query') or ''}\n\n"
            f"TOOLS:\n{ex.get('tools') if isinstance(ex.get('tools'), str) else json.dumps(ex.get('tools') or [], ensure_ascii=False)[:4000]}\n\n"
            f"ASSISTANT:\n{ex.get('answers') if isinstance(ex.get('answers'), str) else json.dumps(ex.get('answers') or '', ensure_ascii=False)[:4000]}"
        )[:max_chars],
        seed + 2,
    )

    for i, subset in enumerate(("code", "math", "science"), start=3):
        log(f"{i}/6 Mixture-of-Thoughts[{subset}]")
        rows += sample_stream(
            "open-r1/Mixture-of-Thoughts",
            "train",
            per_source,
            lambda ex, mc=max_chars: (
                messages_to_text(ex["messages"], mc)
                if ex.get("messages")
                else str(ex.get("text") or "")[:mc]
            ),
            seed + i,
            subset=subset,
        )

    log("6/6 SWE-smith-trajectories(tool)")
    rows += sample_stream(
        "SWE-bench/SWE-smith-trajectories",
        "tool",
        per_source,
        lambda ex: messages_to_text(ex.get("messages"), max_chars),
        seed + 6,
        max_scan=50_000,
    )

    rng = random.Random(seed)
    rng.shuffle(rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "output": str(out_path),
        "recipe": "cerebras_reap_agentic_6way",
        "per_source": per_source,
        "n_total": len(rows),
        "seed": seed,
        "max_chars": max_chars,
        "by_source": dict(Counter(r["source"] for r in rows)),
        "spec": (
            "theblackcat102/evol-codealpaca-v1:4096,"
            "Salesforce/xlam-function-calling-60k:4096 (via ungated mirror),"
            "open-r1/Mixture-of-Thoughts[code]:4096,"
            "open-r1/Mixture-of-Thoughts[math]:4096,"
            "open-r1/Mixture-of-Thoughts[science]:4096,"
            "SWE-bench/SWE-smith-trajectories(tool):4096"
        ),
    }
    out_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    log(json.dumps(manifest, indent=2))
    log(f"Wrote {len(rows)} -> {out_path}")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--per-source", type=int, default=PER_SOURCE)
    p.add_argument("--max-chars", type=int, default=16000)
    args = p.parse_args()
    build(args.out, args.seed, args.per_source, args.max_chars)


if __name__ == "__main__":
    main()
