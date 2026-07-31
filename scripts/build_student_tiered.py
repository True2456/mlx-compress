"""Fused REAP-apply + saliency-tiered mixed-precision quantize for Step-3.7.

Like build_student.py, but instead of one uniform bit-width, each layer's kept
experts are split into banks by their (blended) saliency rank and quantized at
different widths — default 61 hot @ 6-bit / 123 base @ 4-bit / 61 cold @ 3-bit,
which averages the same ~4.5 effective bpw as the uniform 4-bit student.
Routing/selection is untouched; only storage precision differs per expert.

Output loads ONLY via reap_stream.tiered.maybe_patch_tiered (local evals);
stock LM Studio cannot run it. This is a headroom measurement, not a deploy
artifact.

Usage:
    .venv/bin/python scripts/build_student_tiered.py \
        --model models/Step-3.7-Flash \
        --plan  artifacts/plans/plan_p15_blend03.json \
        --out   models/Step-3.7-p15-tiered \
        --tier-sizes 61 123 61 --tier-bits 6 4 3
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_vlm.utils import (
    fetch_from_hub,
    get_model_path,
    save_config,
    save_weights,
    skip_multimodal_module,
)
from mlx_vlm.quant_utils import quantize_model

from reap_stream.tiered import TieredSwitchGLU


def _text_model(model):
    lm = getattr(model, "language_model", None) or model
    return getattr(lm, "model", lm)


def _slice_first_dim(module, keep_idx, names=("weight", "scales", "biases", "bias")):
    for name in names:
        if name in module and module[name] is not None:
            module[name] = module[name][keep_idx]






def build(model_path, plan_path, out_dir, tier_sizes, tier_bits, group_size):
    plan = json.loads(Path(plan_path).read_text())
    keep_counts = {len(v["keep"]) for v in plan["layers"].values()}
    if len(keep_counts) != 1:
        raise ValueError(f"expected uniform keep counts, got {keep_counts}")
    new_n = keep_counts.pop()
    if sum(tier_sizes) != new_n:
        raise ValueError(f"tier sizes {tier_sizes} must sum to keep count {new_n}")

    src = get_model_path(model_path)
    print("[INFO] Loading (lazy)")
    model, config, processor = fetch_from_hub(src, lazy=True, trust_remote_code=True)
    text = _text_model(model)

    all_moe = {i for i, l in enumerate(text.layers) if getattr(l, "is_moe_layer", False)}
    planned = {int(k) for k in plan["layers"]}
    if planned != all_moe:
        raise ValueError(f"plan must cover every MoE layer; missing {sorted(all_moe - planned)[:8]}")

    print(f"[INFO] REAP 288 -> {new_n}, tiers {tier_sizes} @ {tier_bits}-bit")
    for layer_key, layer_plan in plan["layers"].items():
        i = int(layer_key)
        keep = layer_plan["keep"]                      # ascending global ids
        moe = text.layers[i].mlp
        _slice_first_dim(moe.gate.gate, mx.array(keep))
        moe.gate.router_bias = moe.gate.router_bias[mx.array(keep)]

        # rank kept experts by blended score, descending -> tier membership
        scores = np.asarray(layer_plan["scores"], dtype=np.float64)[keep]
        order = np.argsort(-scores)                    # positions within keep list
        bounds = np.cumsum([0] + list(tier_sizes))
        map_bank = np.zeros(new_n, dtype=np.int32)
        map_slot = np.zeros(new_n, dtype=np.int32)
        bank_positions = []
        for b in range(len(tier_sizes)):
            pos = np.sort(order[bounds[b]:bounds[b + 1]])   # keep-list positions
            bank_positions.append(pos)
            map_bank[pos] = b
            map_slot[pos] = np.arange(len(pos), dtype=np.int32)

        old = moe.switch_mlp
        tiered = TieredSwitchGLU(
            old.gate_proj.weight.shape[-1],
            old.down_proj.weight.shape[-1],
            tier_sizes,
            old.activation,
        )
        for b, pos in enumerate(bank_positions):
            gpos = mx.array(np.asarray(keep)[pos])          # global expert ids
            for proj in ("gate_proj", "up_proj", "down_proj"):
                getattr(tiered.banks[b], proj).weight = getattr(old, proj).weight[gpos]
        tiered.map_bank = mx.array(map_bank)
        tiered.map_slot = mx.array(map_slot)
        moe.switch_mlp = tiered

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

    bank_bits = {f"banks.{b}": bits for b, bits in enumerate(tier_bits)}

    def predicate(path, module):
        if skip_multimodal_module(path):
            return False
        for tag, bits in bank_bits.items():
            if f".{tag}." in path or path.endswith(tag):
                return {"group_size": group_size, "bits": bits, "mode": "affine"}
        return True

    print(f"[INFO] Quantizing: banks at {tier_bits}-bit, rest 4-bit gs={group_size} (vision BF16)")
    config.setdefault("vision_config", {})
    target, config = quantize_model(model, config, group_size, 4, mode="affine",
                                    quant_predicate=predicate)
    config["tiered_expert_banks"] = {"sizes": list(tier_sizes), "bits": list(tier_bits)}

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
    # section): save_pretrained() persists tokenizer_class="LlamaTokenizerFast",
    # a known name that makes AutoTokenizer (what mlx_lm/LM Studio calls)
    # apply Llama's own SentencePiece/Metaspace conversion instead of this
    # model's real pretokenizer. fix_tokenizer_class() removes that
    # declaration and verifies live.
    processor.save_pretrained(out_dir)
    sys.path.insert(0, str(Path(__file__).parent))
    from fix_tokenizer_class import fix_tokenizer_class
    fix_tokenizer_class(out_dir)
    save_config(config, config_path=out_dir / "config.json")
    print(f"[OK] tiered student written: {new_n} experts/layer as {tier_sizes} @ {tier_bits}-bit")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tier-sizes", type=int, nargs="+", default=[61, 123, 61])
    ap.add_argument("--tier-bits", type=int, nargs="+", default=[6, 4, 3])
    ap.add_argument("--group-size", type=int, default=64)
    a = ap.parse_args()
    build(a.model, a.plan, a.out, a.tier_sizes, a.tier_bits, a.group_size)
