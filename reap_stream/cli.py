#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_GEMMA = str(
    Path.home()
    / ".lmstudio/models/lmstudio-community/gemma-4-26B-A4B-it-QAT-MLX-4bit"
)
DEFAULT_STEP = str(
    Path.home() / ".lmstudio/models/mlx-community/Step-3.7-Flash-148B-MLX"
)


def main() -> None:
    p = argparse.ArgumentParser(description="Streaming REAP (Gemma4 / Step-3.7 MLX)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument(
            "--arch",
            choices=("gemma4", "step3p7"),
            default="gemma4",
            help="Model family (default gemma4)",
        )

    c = sub.add_parser("collect", help="Collect REAP telemetry + write prune plan")
    add_common(c)
    c.add_argument("--model", default=None)
    c.add_argument("--output", default=None)
    c.add_argument("--ratio", type=float, default=0.25)
    c.add_argument("--max-tokens", type=int, default=256)
    c.add_argument("--layers", default=None, help="e.g. 0-2,5")
    c.add_argument("--min-experts", type=int, default=8)
    c.add_argument(
        "--mode",
        choices=("layerwise", "full"),
        default="layerwise",
        help="layerwise = streaming blocks (default); full = whole model in RAM",
    )
    c.add_argument(
        "--layers-at-once",
        type=int,
        default=4,
        help="How many decoder layers to keep resident before freeing (default 4)",
    )
    c.add_argument(
        "--dataset-file",
        default=None,
        help="JSONL calib file with `text` field (e.g. calib/cerebras_reap_mix.jsonl)",
    )
    c.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on dataset-file rows",
    )

    a = sub.add_parser("apply", help="Apply pruning plan to MLX checkpoint")
    add_common(a)
    a.add_argument("--model", default=None)
    a.add_argument("--plan", required=True)
    a.add_argument("--output", required=True)
    a.add_argument("--dry-run", action="store_true")

    i = sub.add_parser("inspect", help="Lazy-load and print MoE layout")
    add_common(i)
    i.add_argument("--model", default=None)

    n = sub.add_parser(
        "intervene",
        help="Middle-canvas interventions (skip/reverse/repeat) for Step-3.7",
    )
    n.add_argument("--model", default=DEFAULT_STEP)
    n.add_argument("--output", default="artifacts/step37-interventions")
    n.add_argument("--middle-start", type=int, default=12)
    n.add_argument("--middle-end", type=int, default=36)
    n.add_argument("--skip-stride", type=int, default=2)
    n.add_argument("--repeat-times", type=int, default=3)
    n.add_argument("--max-tokens", type=int, default=96)
    n.add_argument("--max-samples", type=int, default=16)
    n.add_argument("--dataset-file", default=None)
    n.add_argument(
        "--variants",
        default="baseline,skip_middle,reverse_middle,middle_repeat",
    )

    d = sub.add_parser(
        "expert-diff",
        help="Contrastive refusal vs benign MoE expert routing (Step-3.7)",
    )
    d.add_argument("--model", default=DEFAULT_STEP)
    d.add_argument("--output", default="artifacts/step37-refusal-diff")
    d.add_argument("--layers", default=None, help="e.g. 12-20,24 (default: 12-35)")
    d.add_argument("--layers-at-once", type=int, default=2)
    d.add_argument("--max-tokens", type=int, default=96)
    d.add_argument("--top-k", type=int, default=10)

    args = p.parse_args()

    def parse_layers(spec: str | None):
        if not spec:
            return None
        out = []
        for part in spec.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-", 1)
                out.extend(range(int(lo), int(hi) + 1))
            else:
                out.append(int(part))
        return out

    arch = getattr(args, "arch", "step3p7")
    if getattr(args, "model", None) is None and args.cmd != "intervene":
        args.model = DEFAULT_STEP if arch == "step3p7" else DEFAULT_GEMMA
    if args.cmd == "collect" and args.output is None:
        args.output = (
            "artifacts/step37" if arch == "step3p7" else "artifacts/gemma4"
        )

    if args.cmd == "intervene":
        from .intervene_step3p7 import run_interventions

        run_interventions(
            model_path=args.model,
            output_dir=args.output,
            middle_start=args.middle_start,
            middle_end=args.middle_end,
            skip_stride=args.skip_stride,
            repeat_times=args.repeat_times,
            max_tokens=args.max_tokens,
            max_samples=args.max_samples,
            dataset_file=args.dataset_file,
            variants=[v.strip() for v in args.variants.split(",") if v.strip()],
        )
        return

    if args.cmd == "expert-diff":
        from .expert_diff_step3p7 import run_expert_diff

        def parse_layers(spec: str | None):
            if not spec:
                return None
            out = []
            for part in spec.split(","):
                part = part.strip()
                if "-" in part:
                    lo, hi = part.split("-", 1)
                    out.extend(range(int(lo), int(hi) + 1))
                else:
                    out.append(int(part))
            return out

        run_expert_diff(
            model_path=args.model,
            output_dir=args.output,
            layers=parse_layers(args.layers),
            layers_at_once=args.layers_at_once,
            max_tokens=args.max_tokens,
            top_k=args.top_k,
        )
        return

    if args.cmd == "inspect":
        if arch == "step3p7":
            from .collect_step3p7 import inspect_model
            import json

            print(json.dumps(inspect_model(args.model), indent=2))
        else:
            raise SystemExit("inspect currently implemented for --arch step3p7")
        return

    if args.cmd == "collect":
        if arch == "step3p7":
            from .collect_step3p7 import collect_and_plan
        else:
            from .collect_gemma4 import collect_and_plan

        path = collect_and_plan(
            model_path=args.model,
            output_dir=args.output,
            ratio=args.ratio,
            max_tokens=args.max_tokens,
            layers=parse_layers(args.layers),
            min_experts=args.min_experts,
            mode=args.mode,
            layers_at_once=args.layers_at_once,
            dataset_file=args.dataset_file,
            max_samples=args.max_samples,
        )
        print(f"Wrote plan: {path}")
    elif args.cmd == "apply":
        if arch == "step3p7":
            from .apply_step3p7 import apply_plan
        else:
            from .apply_gemma4 import apply_plan

        apply_plan(
            model_path=args.model,
            plan_path=args.plan,
            output_dir=args.output,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
