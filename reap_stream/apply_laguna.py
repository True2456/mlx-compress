"""Apply a REAP plan to a Laguna-S-2.1 (poolside) MLX checkpoint.

Port of apply_deepseek_v4.py's pattern (lazy load, in-place lazy slicing on
the module tree, then model.save_weights), simplified because Laguna has no
hash-routed layers (plain REAP deletion everywhere, no REAM needed) and no
Hyper-Connections.

Laguna-specific vs. DeepSeek-V4's layout:
- router weight: layer.mlp.gate.proj.weight (LagunaTopKRouter), under
  `.proj` -- not `layer.mlp.gate.weight` directly (verified against
  mlx_vlm/models/laguna/language.py: `self.proj = nn.Linear(...)` inside
  LagunaTopKRouter, and LanguageModel.sanitize's _remap_router_weights
  renames the on-disk `mlp.gate.weight` key to `mlp.gate.proj.weight` at
  load time).
- router bias: layer.mlp.gate.e_score_correction_bias -- present on every
  MoE layer (no hash-routed exception, unlike DeepSeek-V4).
- experts: layer.mlp.switch_mlp.{gate,up,down}_proj -- same SwitchGLU
  convention as Step-3.7/Gemma-4/DeepSeek-V4, sliceable identically. This
  checkpoint's experts are natively NVFP4 (weight + scales, no biases
  tensor -- confirmed via model.safetensors.index.json: only `.weight` and
  `.scales` keys per switch_mlp projection, no `.biases`), so `_slice_first_dim`
  naturally no-ops on the absent "biases" key.
- layer 0 is dense (`mlp_only_layers: [0]`) -- has no `.switch_mlp`/`.gate`,
  detected the same way collect_laguna.py does (hasattr check) rather than
  a hardcoded index.
- `num_experts` is a single top-level config field (like DeepSeek-V4's
  `n_routed_experts`), so every MoE layer must end up the SAME kept width.

Checkpoint already ships at native 4-bit (NVFP4) -- per the project's
DeepSeek-V4 finding (docs/DEEPSEEK-V4-FINDINGS.md #7), when the source is
already low-bit, prune alone is the best-quality way to hit a size budget:
quant damage generally dominates prune damage, and there's no headroom left
to lose more to a second quantization pass anyway. This script does NOT
offer a --requantize path for that reason (unlike apply_deepseek_v4.py,
which requantizes because its source was still a relatively fresh 4-bit
budget with room to trade more precision for size).

Usage:
    .venv/bin/python -m reap_stream.apply_laguna \
        --model ~/.lmstudio/models/poolside/Laguna-S-2.1-NVFP4-mlx \
        --plan artifacts/laguna-reap/pruning-plan.json \
        --output models/Laguna-S-2.1-REAP \
        --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_vlm import load


def _text_model(model):
    lm = getattr(model, "language_model", None) or model
    return getattr(lm, "model", lm)


def _is_moe_layer(layer) -> bool:
    return hasattr(layer.mlp, "switch_mlp")


def _slice_first_dim(module: Any, keep: list[int], keys: tuple[str, ...] | None = None) -> None:
    idx = mx.array(keep)
    names = keys if keys is not None else ("weight", "scales", "biases", "bias")
    for name in names:
        if name in module and module[name] is not None:
            module[name] = module[name][idx]


def apply_plan(
    model_path: str,
    plan_path: str | Path,
    output_dir: str | Path,
    dry_run: bool = False,
) -> Path:
    plan = json.loads(Path(plan_path).read_text())
    model, processor = load(model_path, lazy=True)
    text = _text_model(model)
    n_layers = len(text.layers)

    keep_counts = {len(v["keep"]) for v in plan["layers"].values()}
    if len(keep_counts) != 1:
        raise ValueError(f"Laguna/MLX expects uniform keep counts; got {keep_counts}")
    new_n = keep_counts.pop()

    all_moe = {i for i, layer in enumerate(text.layers) if _is_moe_layer(layer)}
    planned = {int(k) for k in plan["layers"]}
    if planned - all_moe:
        raise ValueError(f"plan references non-MoE layers: {sorted(planned - all_moe)}")
    if not dry_run and planned != all_moe:
        missing = sorted(all_moe - planned)
        raise ValueError(
            "Non-dry-run apply requires a plan covering every MoE layer "
            f"(missing {len(missing)} e.g. {missing[:8]}). Use --dry-run for partial plans."
        )

    sliced = []
    for layer_key, layer_plan in plan["layers"].items():
        i = int(layer_key)
        layer = text.layers[i]
        keep = layer_plan["keep"]

        gate = layer.mlp.gate
        _slice_first_dim(gate.proj, keep, keys=("weight",))
        if hasattr(gate, "e_score_correction_bias") and gate.e_score_correction_bias is not None:
            gate.e_score_correction_bias = gate.e_score_correction_bias[mx.array(keep)]

        sg = layer.mlp.switch_mlp
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            _slice_first_dim(getattr(sg, proj_name), keep)

        print(f"[apply] layer {i:02d}/{n_layers - 1} pruned -> keep={len(keep)}", flush=True)
        sliced.append({"layer": i, "keep": len(keep), "router": list(gate.proj.weight.shape)})

    output_dir = Path(output_dir)
    if dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "new_experts": new_n,
                    "layers_sliced": len(sliced),
                    "sample": sliced[:3],
                    "output": str(output_dir),
                },
                indent=2,
            )
        )
        return output_dir

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    src = Path(model_path)
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.jinja",
        "config.py",
    ):
        p = src / name
        if p.exists():
            shutil.copy2(p, output_dir / name)

    cfg = json.loads((src / "config.json").read_text())
    cfg["num_experts"] = new_n
    cfg["_reap_note"] = (
        f"REAP-pruned from {plan['layers'][next(iter(plan['layers']))]['num_experts']} "
        f"to {new_n} experts/layer on every MoE layer (layer 0 stays dense, "
        f"mlp_only_layers unchanged); plain deletion, no REAM merge needed "
        f"(no hash-routed layers in this architecture); no additional "
        f"quantization applied (checkpoint already native NVFP4 4-bit -- see "
        f"apply_laguna.py docstring)."
    )

    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (output_dir / "reap-plan.json").write_text(json.dumps(plan, indent=2))

    model.save_weights(str(output_dir / "model.safetensors"))
    print(
        f"Wrote pruned Laguna checkpoint -> {output_dir} "
        f"({new_n} experts/layer, {len(sliced)} MoE layers pruned)"
    )
    return output_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    apply_plan(a.model, a.plan, a.output, dry_run=a.dry_run)


if __name__ == "__main__":
    main()
