"""Fused REAP-apply + quantize for Step-3.7 (mlx_vlm).

Applies a REAP plan (expert slicing) AND quantizes in a single pass, so we
only ever write the small quantized student (~89 GB at 4-bit) and never the
~316 GB reaped BF16 intermediate.

Vision tower is left unquantized (skip_multimodal_module), matching how
mlx_vlm.convert treats multimodal modules.

Usage:
    .venv/bin/python scripts/build_student.py \
        --model models/Step-3.7-Flash \
        --plan  artifacts/step37-bf16-layerwise-5k/plan_p15.json \
        --out   models/Step-3.7-p15-4bit \
        --bits 4 --group-size 64 --mode affine
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx
from mlx_vlm.utils import (
    fetch_from_hub,
    get_model_path,
    save_config,
    save_weights,
    skip_multimodal_module,
)
from mlx_vlm.quant_utils import quantize_model


def _text_model(model):
    lm = getattr(model, "language_model", None) or model
    return getattr(lm, "model", lm)


def _slice_first_dim(module, keep_idx, names=("weight", "scales", "biases", "bias")):
    for name in names:
        if name in module and module[name] is not None:
            module[name] = module[name][keep_idx]






def apply_and_quantize(model_path, plan_path, out_dir, bits, group_size, mode):
    plan = json.loads(Path(plan_path).read_text())
    keep_counts = {len(v["keep"]) for v in plan["layers"].values()}
    if len(keep_counts) != 1:
        raise ValueError(f"expected uniform keep counts, got {keep_counts}")
    new_n = keep_counts.pop()

    src = get_model_path(model_path)
    print("[INFO] Loading (lazy)")
    model, config, processor = fetch_from_hub(src, lazy=True, trust_remote_code=True)
    text = _text_model(model)

    all_moe = {i for i, l in enumerate(text.layers) if getattr(l, "is_moe_layer", False)}
    planned = {int(k) for k in plan["layers"]}
    if planned != all_moe:
        raise ValueError(
            f"plan must cover every MoE layer; missing {sorted(all_moe - planned)[:8]}"
        )

    print(f"[INFO] Applying REAP plan: 288 -> {new_n} experts across {len(planned)} layers")
    for layer_key, layer_plan in plan["layers"].items():
        i = int(layer_key)
        keep = mx.array(layer_plan["keep"])
        moe = text.layers[i].mlp
        _slice_first_dim(moe.gate.gate, keep)
        moe.gate.router_bias = moe.gate.router_bias[keep]
        for proj in ("gate_proj", "up_proj", "down_proj"):
            _slice_first_dim(getattr(moe.switch_mlp, proj), keep)

    # bump expert count in every config view
    for cfg in (config, config.get("text_config")):
        if isinstance(cfg, dict) and "moe_num_experts" in cfg:
            cfg["moe_num_experts"] = new_n
    for obj in (getattr(model, "config", None), getattr(text, "args", None),
                getattr(getattr(model, "language_model", None), "args", None)):
        if obj is not None and hasattr(obj, "moe_num_experts"):
            obj.moe_num_experts = new_n
        tc = getattr(obj, "text_config", None)
        if tc is not None and hasattr(tc, "moe_num_experts"):
            tc.moe_num_experts = new_n

    def predicate(path, module):
        if skip_multimodal_module(path):   # keep vision tower unquantized
            return False
        return True

    print(f"[INFO] Quantizing text path: {bits}-bit {mode} gs={group_size} (vision left BF16)")
    config.setdefault("vision_config", {})
    target, config = quantize_model(
        model, config, group_size, bits, mode=mode, quant_predicate=predicate
    )

    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    print(f"[INFO] Saving -> {out_dir}")
    save_weights(out_dir, target, donate_weights=True)

    for pattern in ("*.py", "*.json"):
        for f in glob.glob(str(src / pattern)):
            if Path(f).name == "model.safetensors.index.json":
                continue
            shutil.copy(f, out_dir)
    for item in src.iterdir():
        if item.is_dir():
            dest = out_dir / item.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)

    # CORRECTED 2026-07-26 (see docs/TOKENIZER-INVESTIGATION.md's correction
    # section): the earlier comment here was wrong. save_pretrained() below
    # persists whatever tokenizer_class transformers resolved this tokenizer
    # to, which is "LlamaTokenizerFast" -- a KNOWN architecture name that
    # makes AutoTokenizer.from_pretrained() (what mlx_lm/LM Studio actually
    # calls) apply Llama's own hardcoded SentencePiece/Metaspace conversion
    # recipe instead of reading this model's real, correct, custom
    # pretokenizer from tokenizer.json. fix_tokenizer_class() removes that
    # misleading class declaration and verifies the fix live.
    processor.save_pretrained(out_dir)
    sys.path.insert(0, str(Path(__file__).parent))
    from fix_tokenizer_class import fix_tokenizer_class
    fix_tokenizer_class(out_dir)
    save_config(config, config_path=out_dir / "config.json")
    print(f"[OK] student written: {new_n} experts/layer, {bits}-bit {mode}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--mode", default="affine")
    a = ap.parse_args()
    apply_and_quantize(a.model, a.plan, a.out, a.bits, a.group_size, a.mode)
