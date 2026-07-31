"""Fused REAP-apply + mixed-precision quantize: vblend plan + shared8 policy.

Per docs/TOMOGRAPHY-RESULT.md: share_expert + dense-layer (0-2) MLPs + all
router gates + lm_head/embed_tokens at 8-bit, everything else (routed MoE
experts) at 4-bit gs64, vision tower BF16. Same REAP-apply surgery as
build_student.py; only the quant_predicate differs. Uses only stock per-path
quant config -- no custom classes, loads in stock LM Studio / mlx_vlm like any
quantized checkpoint.

The head was added 2026-07-24 after measuring -0.0086 NLL across all five
categories for +0.53 GB; it is the always-on weight class the tomography sweep
never covered. ~4.68 bpw, ~93.5 GB.

Usage:
    .venv/bin/python scripts/build_student_shared8.py \
        --model models/Step-3.7-Flash \
        --plan  artifacts/plans/plan_p15_blend03.json \
        --out   models/Step-3.7-p15-4bit-vblend-shared8
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


def _layer_of(path: str) -> int | None:
    parts = path.split(".")
    for j, p in enumerate(parts):
        if p == "layers" and j + 1 < len(parts) and parts[j + 1].isdigit():
            return int(parts[j + 1])
    return None






def write_generation_config(out_dir: Path, config: dict):
    """Ship sampling defaults with the checkpoint.

    The upstream checkpoint has no generation_config.json, so every runtime
    falls back to its own defaults -- LM Studio's are tuned for prose. That is
    a bad fit here: this tokenizer emits one token per digit (ids 19-28) and a
    standalone space token (223), so numbers carry no token-level redundancy
    and a single resampled token corrupts them. Prose is unaffected, which is
    why the failure reads as "numbers are wrong, everything else is perfect".

    repetition_penalty is pinned to 1.0 deliberately. Its rolling window
    (context_size=20 in mlx_lm) is nearly all digit tokens inside any numeric
    run, so it suppresses every digit while leaving the letters and
    punctuation competing with them untouched -- it is the one sampler setting
    that can reorder tokens rather than just filter them. min_p carries loop
    suppression instead.
    """
    text_cfg = config.get("text_config") or {}
    gen = {
        "bos_token_id": text_cfg.get("bos_token_id", config.get("bos_token_id")),
        "eos_token_id": text_cfg.get("eos_token_id", config.get("eos_token_id")),
        "do_sample": True,
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 100,
        "min_p": 0.05,
        "repetition_penalty": 1.0,
    }
    (out_dir / "generation_config.json").write_text(json.dumps(gen, indent=2) + "\n")
    print("[INFO] wrote generation_config.json (temp 0.5, min_p 0.05, no repeat penalty)")


def apply_and_quantize(model_path, plan_path, out_dir, group_size):
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
        raise ValueError(f"plan must cover every MoE layer; missing {sorted(all_moe - planned)[:8]}")

    print(f"[INFO] Applying REAP plan: 288 -> {new_n} experts across {len(planned)} layers")
    for layer_key, layer_plan in plan["layers"].items():
        i = int(layer_key)
        keep = mx.array(layer_plan["keep"])
        moe = text.layers[i].mlp
        _slice_first_dim(moe.gate.gate, keep)
        moe.gate.router_bias = moe.gate.router_bias[keep]
        for proj in ("gate_proj", "up_proj", "down_proj"):
            _slice_first_dim(getattr(moe.switch_mlp, proj), keep)

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
        if skip_multimodal_module(path):
            return False
        li = _layer_of(path)
        if ".share_expert." in path or ".gate.gate" in path:
            return {"group_size": group_size, "bits": 8, "mode": "affine"}
        if li is not None and li <= 2 and ".mlp." in path:
            return {"group_size": group_size, "bits": 8, "mode": "affine"}
        # lm_head / embed_tokens: the most extreme members of the always-on
        # weight class -- every token, and every vocabulary row participates in
        # every argmax, so no sparsity smooths the quantization noise. Measured
        # -0.0086 NLL across all five categories for +0.53 GB
        # (TOMOGRAPHY-RESULT.md follow-up). Stated explicitly rather than left
        # to the default below: they were 4-bit only by accident, so lowering
        # the global bit-width would otherwise drag them down silently. Digit
        # argmax margins say never go below 4-bit here (diag_head_digits.py).
        if "lm_head" in path or "embed_tokens" in path:
            return {"group_size": group_size, "bits": 8, "mode": "affine"}
        return True  # default 4-bit: routed experts

    print(f"[INFO] Quantizing: shared_expert/dense-mlp/router @8-bit, "
          f"routed experts @4-bit gs={group_size} (vision BF16)")
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
    write_generation_config(out_dir, config)
    print(f"[OK] student written: {new_n} experts/layer, shared8 policy")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group-size", type=int, default=64)
    a = ap.parse_args()
    apply_and_quantize(a.model, a.plan, a.out, a.group_size)
