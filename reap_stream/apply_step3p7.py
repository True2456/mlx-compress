"""Apply REAP plan to Step-3.7 Flash MLX (mlx_vlm step3p7) checkpoints."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_vlm import load


def _text_model(model):
    lm = getattr(model, "language_model", None) or model
    return getattr(lm, "model", lm)


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

    keep_counts = {len(v["keep"]) for v in plan["layers"].values()}
    if len(keep_counts) != 1:
        raise ValueError(f"Step MLX expects uniform keep counts; got {keep_counts}")
    new_n = keep_counts.pop()

    all_moe = {i for i, layer in enumerate(text.layers) if getattr(layer, "is_moe_layer", False)}
    planned = {int(k) for k in plan["layers"]}
    if planned - all_moe:
        raise ValueError(f"plan references non-MoE layers: {sorted(planned - all_moe)}")
    if not dry_run and planned != all_moe:
        missing = sorted(all_moe - planned)
        raise ValueError(
            "Non-dry-run apply requires a plan covering every MoE layer "
            f"(missing {len(missing)} e.g. {missing[:8]}). "
            "Use --dry-run for partial plans."
        )

    sliced = []
    for layer_key, layer_plan in plan["layers"].items():
        i = int(layer_key)
        layer = text.layers[i]
        keep = layer_plan["keep"]
        moe = layer.mlp

        # Router: Linear weight (E, H) + router_bias (E,)
        _slice_first_dim(moe.gate.gate, keep)
        moe.gate.router_bias = moe.gate.router_bias[mx.array(keep)]

        sg = moe.switch_mlp
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            _slice_first_dim(getattr(sg, proj_name), keep)

        sliced.append(
            {
                "layer": i,
                "keep": len(keep),
                "router": list(moe.gate.gate.weight.shape),
                "router_bias": list(moe.gate.router_bias.shape),
                "switch_gate": list(sg.gate_proj.weight.shape),
            }
        )

    def bump_cfg(cfg: Any) -> None:
        if cfg is None:
            return
        if hasattr(cfg, "moe_num_experts"):
            cfg.moe_num_experts = new_n
        if isinstance(cfg, dict) and "moe_num_experts" in cfg:
            cfg["moe_num_experts"] = new_n
        tc = getattr(cfg, "text_config", None)
        if tc is not None and hasattr(tc, "moe_num_experts"):
            tc.moe_num_experts = new_n
        if isinstance(cfg, dict) and isinstance(cfg.get("text_config"), dict):
            cfg["text_config"]["moe_num_experts"] = new_n

    output_dir = Path(output_dir)
    if dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "new_experts": new_n,
                    "layers_sliced": len(sliced),
                    "moe_layers_total": len(all_moe),
                    "partial_plan": planned != all_moe,
                    "sample": sliced[:3],
                    "output": str(output_dir),
                },
                indent=2,
            )
        )
        return output_dir

    bump_cfg(getattr(model, "args", None) or getattr(model, "config", None))
    lm = getattr(model, "language_model", None)
    if lm is not None:
        bump_cfg(getattr(lm, "args", None))
    bump_cfg(getattr(text, "args", None))

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    src = Path(model_path)
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "generation_config.json",
        "processor_config.json",
        "config.json",
    ):
        p = src / name
        if p.exists():
            shutil.copy2(p, output_dir / name)

    cfg_path = output_dir / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        if "text_config" in cfg and isinstance(cfg["text_config"], dict):
            cfg["text_config"]["moe_num_experts"] = new_n
        cfg["moe_num_experts"] = new_n
        cfg_path.write_text(json.dumps(cfg, indent=2))

    model.save_weights(str(output_dir / "model.safetensors"))
    print(f"Wrote pruned Step checkpoint -> {output_dir} ({new_n} experts/layer)")
    return output_dir
