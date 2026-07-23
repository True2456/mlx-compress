#!/usr/bin/env python3
"""
Mac layerwise REAP on full BF16 Step-3.7-Flash.

Collect saliency only → write nested plans (10/15/20/25).
Does NOT apply/prune weights (that needs disk later).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reap_stream.collect_step3p7 import collect_layerwise
from reap_stream.dataset import load_prompt_texts
from reap_stream.saliency import build_plan

RUNGS = (0.10, 0.15, 0.20, 0.25)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default=str(ROOT / "models" / "Step-3.7-Flash"),
        help="HF/MLX Step-3.7-Flash path (BF16)",
    )
    p.add_argument(
        "--dataset",
        default=str(ROOT / "calib" / "cloud_reap_8k.jsonl"),
    )
    p.add_argument("--output", default=str(ROOT / "artifacts" / "step37-bf16-layerwise"))
    p.add_argument("--max-tokens", type=int, default=384)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--layers-at-once", type=int, default=2)
    p.add_argument("--truncation", choices=["head", "tail", "headtail"],
                   default="head",
                   help="How to fit prompts into max-tokens. 'headtail' keeps "
                        "both task setup and the ASSISTANT answer for long "
                        "agentic/tool prompts (validated best vs full-length).")
    p.add_argument("--min-experts", type=int, default=8)
    p.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Per-layer checkpoint dir (default: <output>/checkpoints)",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing checkpoints and start fresh",
    )
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = (
        Path(args.checkpoint_dir) if args.checkpoint_dir else out / "checkpoints"
    )
    log_path = out / "run.log"

    def log(msg: str) -> None:
        line = msg.rstrip()
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    t0 = time.time()
    prompts = load_prompt_texts(args.dataset, limit=args.max_samples)
    meta = {
        "model": args.model,
        "dataset": args.dataset,
        "n_prompts": len(prompts),
        "max_tokens": args.max_tokens,
        "truncation": args.truncation,
        "layers_at_once": args.layers_at_once,
        "checkpoint_dir": str(checkpoint_dir),
        "resume": not args.no_resume,
        "rungs": list(RUNGS),
        "note": "collect+plans only; apply deferred",
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    log("=" * 70)
    log("MAC BF16 LAYERWISE REAP (collect + nested plans)")
    log("=" * 70)
    log(json.dumps(meta, indent=2))

    stats, trace = collect_layerwise(
        args.model,
        prompts=prompts,
        max_tokens=args.max_tokens,
        layers=None,
        layers_at_once=args.layers_at_once,
        checkpoint_dir=checkpoint_dir,
        resume=not args.no_resume,
        truncation=args.truncation,
    )
    (out / "trace.json").write_text(json.dumps(trace, indent=2))
    sal = {str(k): v.to_dict() for k, v in stats.items()}
    (out / "saliency.json").write_text(json.dumps(sal))
    n_experts = next(iter(stats.values())).num_experts
    log(f"saliency layers={len(stats)} experts={n_experts}")

    prev_prune: set[int] | None = None
    nested_ok = True
    mid = sorted(stats)[len(stats) // 2]
    for ratio in RUNGS:
        plan = build_plan(stats, ratio=ratio, min_experts=args.min_experts)
        plan["arch"] = "step3p7"
        plan["model_path"] = args.model
        plan["mode"] = "layerwise"
        plan["precision"] = "bf16"
        plan["dataset"] = args.dataset
        plan["n_prompts"] = len(prompts)
        plan["max_tokens"] = args.max_tokens
        plan["apply_deferred"] = True
        name = f"plan_p{int(ratio * 100):02d}.json"
        (out / name).write_text(json.dumps(plan, indent=2))
        keep = len(plan["layers"][str(mid)]["keep"])
        prune = set(plan["layers"][str(mid)]["prune"])
        log(f"  {name}: keep={keep} prune={n_experts - keep} (layer {mid})")
        if prev_prune is not None and not prev_prune.issubset(prune):
            nested_ok = False
            log(f"  FAIL nest at {ratio}")
        prev_prune = prune

    elapsed = time.time() - t0
    summary = {
        "ok": True,
        "nested_ok": nested_ok,
        "n_experts": n_experts,
        "n_moe_layers": len(stats),
        "n_prompts": len(prompts),
        "elapsed_sec": elapsed,
        "output": str(out),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"nested={nested_ok} elapsed_h={elapsed/3600:.2f}")
    log(f"DONE → {out}")


if __name__ == "__main__":
    main()
