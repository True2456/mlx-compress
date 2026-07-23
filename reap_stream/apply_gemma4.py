from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import load


def _text_model(model):
    lm = getattr(model, "language_model", None) or model
    return getattr(lm, "model", lm)


def _slice_first_dim(module: Any, keep: list[int], keys: tuple[str, ...] | None = None) -> None:
    """Slice expert/axis-0 tensors on an MLX module (dense or quantized)."""
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
    model, tokenizer = load(model_path)
    text = _text_model(model)

    keep_counts = {len(v["keep"]) for v in plan["layers"].values()}
    if len(keep_counts) != 1:
        raise ValueError(
            f"Gemma4/MLX requires uniform experts/layer; got keep counts {keep_counts}"
        )
    new_n = keep_counts.pop()

    for layer_key, layer_plan in plan["layers"].items():
        i = int(layer_key)
        layer = text.layers[i]
        if not getattr(layer, "enable_moe", False):
            continue
        keep = layer_plan["keep"]

        # router linear (often quantized): weight/scales/biases are (E, ...)
        _slice_first_dim(layer.router.proj, keep)
        if hasattr(layer.router, "per_expert_scale"):
            layer.router.per_expert_scale = layer.router.per_expert_scale[mx.array(keep)]

        sg = layer.experts.switch_glu
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            _slice_first_dim(getattr(sg, proj_name), keep)

    def bump_cfg(cfg: Any) -> None:
        if cfg is None:
            return
        if hasattr(cfg, "num_experts"):
            cfg.num_experts = new_n
        if isinstance(cfg, dict) and "num_experts" in cfg:
            cfg["num_experts"] = new_n
        tc = getattr(cfg, "text_config", None)
        if tc is not None and hasattr(tc, "num_experts"):
            tc.num_experts = new_n
        if isinstance(cfg, dict) and isinstance(cfg.get("text_config"), dict):
            cfg["text_config"]["num_experts"] = new_n

    bump_cfg(getattr(model, "args", None) or getattr(model, "config", None))
    bump_cfg(getattr(getattr(model, "language_model", None), "args", None))
    for layer in text.layers:
        if hasattr(layer, "config") and hasattr(layer.config, "num_experts"):
            layer.config.num_experts = new_n
        if hasattr(layer, "router") and hasattr(layer.router, "config"):
            layer.router.config.num_experts = new_n

    output_dir = Path(output_dir)
    if dry_run:
        print(f"dry-run OK: would keep {new_n} experts/layer -> {output_dir}")
        return output_dir

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    src = Path(model_path)
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "processor_config.json",
    ):
        p = src / name
        if p.exists():
            shutil.copy2(p, output_dir / name)

    model.save_weights(str(output_dir / "model.safetensors"))

    cfg = json.loads((src / "config.json").read_text())
    if "text_config" in cfg and isinstance(cfg["text_config"], dict):
        cfg["text_config"]["num_experts"] = new_n
    cfg["num_experts"] = new_n
    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (output_dir / "reap-plan.json").write_text(json.dumps(plan, indent=2))
    print(f"Wrote pruned model ({new_n} experts/layer) to {output_dir}")
    return output_dir
