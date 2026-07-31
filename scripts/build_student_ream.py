"""Build a REAM student: merge low-saliency experts instead of pruning them,
then quantize with the current shared8+head8 policy.

NEW and additive. Mirrors build_student_shared8.py's structure but swaps the
`_slice_first_dim(..., keep)` prune step for reap_stream.ream.merge_experts.
build_student_shared8.py is imported (for write_generation_config) and NOT
modified.

STATUS: unit-tested on synthetic tensors (reap_stream/test_ream.py); this
end-to-end build is UNRUN -- it loads the 375 GB BF16 base and was written
under a no-large-launch constraint. Validate against
artifacts/ppl-p15-vblend-shared8-500.json (+ the 250-image NLL) before trusting
it, exactly as shared8 and head8 were. Do not promote on faith: SwiGLU merging
is approximate, and this model's flat saliency may mean the merge partners are
near-orthogonal (see docs/FINDINGS.md "cheapest decisive experiment" -- the
co-occurrence gate that would predict this is not yet measured).

Usage:
    .venv/bin/python scripts/build_student_ream.py \
        --model models/Step-3.7-Flash \
        --plan  artifacts/plans/plan_p15_blend03.json \
        --out   models/Step-3.7-p15-ream-shared8-head8
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx
from mlx_vlm.utils import (fetch_from_hub, get_model_path, save_config,
                           save_weights, skip_multimodal_module)
from mlx_vlm.quant_utils import quantize_model

from reap_stream.ream import assign_merges, merge_experts, router_similarity
# reuse, do not duplicate: the sampler-defaults writer is unchanged
from scripts.build_student_shared8 import write_generation_config, _text_model, _layer_of


def apply_ream_and_quantize(model_path, plan_path, out_dir, group_size):
    plan = json.loads(Path(plan_path).read_text())
    keep_counts = {len(v["keep"]) for v in plan["layers"].values()}
    if len(keep_counts) != 1:
        raise ValueError(f"expected uniform keep counts, got {keep_counts}")
    new_n = keep_counts.pop()

    src = get_model_path(model_path)
    print("[INFO] Loading (lazy BF16)")
    model, config, processor = fetch_from_hub(src, lazy=True, trust_remote_code=True)
    text = _text_model(model)

    all_moe = {i for i, l in enumerate(text.layers) if getattr(l, "is_moe_layer", False)}
    planned = {int(k) for k in plan["layers"]}
    if planned != all_moe:
        raise ValueError(f"plan must cover every MoE layer; missing {sorted(all_moe - planned)[:8]}")

    print(f"[INFO] Applying REAM: 288 -> {new_n} experts/layer via saliency-weighted merge")
    merge_report = {}
    for layer_key, lp in plan["layers"].items():
        i = int(layer_key)
        keep, prune, scores = lp["keep"], lp["prune"], lp["scores"]
        moe = text.layers[i].mlp

        # similarity from BF16 router rows (pre-quant), assign merges
        router_rows = moe.gate.gate["weight"]
        sim = router_similarity(router_rows, mx)
        groups = assign_merges(keep, prune, sim)
        merge_report[layer_key] = sum(len(v) for v in groups.values())

        # merge router matrix + bias, then the three expert projections
        moe.gate.gate["weight"] = merge_experts(moe.gate.gate["weight"], keep, groups, scores, mx)
        moe.gate.router_bias = merge_experts(moe.gate.router_bias, keep, groups, scores, mx)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            m = getattr(moe.switch_mlp, proj)
            m["weight"] = merge_experts(m["weight"], keep, groups, scores, mx)

    merged_total = sum(merge_report.values())
    print(f"[INFO] merged {merged_total} pruned experts across {len(merge_report)} layers")

    # config: expert count -> new_n (same as shared8)
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

    # identical quant policy to build_student_shared8 (shared8 + head8)
    def predicate(path, module):
        if skip_multimodal_module(path):
            return False
        li = _layer_of(path)
        if ".share_expert." in path or ".gate.gate" in path:
            return {"group_size": group_size, "bits": 8, "mode": "affine"}
        if li is not None and li <= 2 and ".mlp." in path:
            return {"group_size": group_size, "bits": 8, "mode": "affine"}
        if "lm_head" in path or "embed_tokens" in path:
            return {"group_size": group_size, "bits": 8, "mode": "affine"}
        return True  # routed experts @ 4-bit

    print("[INFO] Quantizing (shared8+head8 policy)")
    config.setdefault("vision_config", {})
    target, config = quantize_model(model, config, group_size, 4, mode="affine",
                                    quant_predicate=predicate)

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

    processor.save_pretrained(out_dir)
    sys.path.insert(0, str(Path(__file__).parent))
    from fix_tokenizer_class import fix_tokenizer_class
    fix_tokenizer_class(out_dir)                 # see TOKENIZER-INVESTIGATION.md's correction
    save_config(config, config_path=out_dir / "config.json")
    write_generation_config(out_dir, config)
    (out_dir / "ream_merge_report.json").write_text(json.dumps(merge_report, indent=2))
    print(f"[OK] REAM student written: {new_n} experts/layer")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group-size", type=int, default=64)
    a = ap.parse_args()
    apply_ream_and_quantize(a.model, a.plan, a.out, a.group_size)
