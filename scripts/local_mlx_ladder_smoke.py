#!/usr/bin/env python3
"""
Local MLX ladder smoke on already-REAPed Step-3.7-Flash-148B-MLX.

Validates: collect → nested plans (10/15/20/25) → dry-run apply → in-memory
NLL gate on p10 vs baseline. Does NOT write into LM Studio model dir.
"""
from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm import load
from mlx_vlm.models.base import create_attention_mask

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path.home() / ".lmstudio/models/mlx-community/Step-3.7-Flash-148B-MLX"
OUT = ROOT / "artifacts" / "local-mlx-ladder-smoke"
CALIB = ROOT / "calib" / "cloud_reap_8k.jsonl"
MAX_SAMPLES = 48
MAX_TOKENS = 128
LAYERS_AT_ONCE = 2
RUNGS = (0.10, 0.15, 0.20, 0.25)


def _text_model(model):
    lm = getattr(model, "language_model", None) or model
    return getattr(lm, "model", lm)


def _slice_first_dim(module, keep):
    idx = mx.array(keep)
    for name in ("weight", "scales", "biases", "bias"):
        if name in module and module[name] is not None:
            module[name] = module[name][idx]


def apply_in_memory(text, plan: dict) -> int:
    new_n = None
    for layer_key, layer_plan in plan["layers"].items():
        i = int(layer_key)
        keep = layer_plan["keep"]
        new_n = len(keep)
        moe = text.layers[i].mlp
        _slice_first_dim(moe.gate.gate, keep)
        moe.gate.router_bias = moe.gate.router_bias[mx.array(keep)]
        for proj in ("gate_proj", "up_proj", "down_proj"):
            _slice_first_dim(getattr(moe.switch_mlp, proj), keep)
    return int(new_n or 0)


def mean_nll(text, lm_head, batches: list[list[int]], sliding_window) -> float:
    total = 0.0
    ntok = 0
    for tokens in batches:
        if len(tokens) < 2:
            continue
        ids = mx.array(tokens)[None]
        h = text.embed_tokens(ids)
        for layer in text.layers:
            if getattr(layer, "is_sliding", False):
                mask = create_attention_mask(h, None, window_size=sliding_window)
            else:
                mask = create_attention_mask(h, None)
            h = layer(h, mask=mask, cache=None)
            mx.eval(h)
        logits = lm_head(h).astype(mx.float32)
        mx.eval(logits)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        # numpy gather for robustness
        logp_np = np.array(logp[0])
        tok = tokens
        for i in range(len(tok) - 1):
            total += float(-logp_np[i, tok[i + 1]])
            ntok += 1
        mx.clear_cache()
    return total / max(ntok, 1)


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT))
    from reap_stream.collect_step3p7 import collect_layerwise
    from reap_stream.dataset import load_prompt_texts
    from reap_stream.saliency import build_plan
    from reap_stream.apply_step3p7 import apply_plan

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("=" * 70)
    print("LOCAL MLX LADDER SMOKE (148B already-REAPed, no LM Studio overwrite)")
    print("=" * 70)
    print(f"model: {MODEL}")
    print(f"calib: {CALIB} (max {MAX_SAMPLES} x {MAX_TOKENS} tok)")

    prompts = load_prompt_texts(CALIB, limit=MAX_SAMPLES)
    print(f"loaded {len(prompts)} prompts")

    print("\n--- COLLECT ---")
    stats, trace = collect_layerwise(
        str(MODEL),
        prompts=prompts,
        max_tokens=MAX_TOKENS,
        layers=None,
        layers_at_once=LAYERS_AT_ONCE,
    )
    (OUT / "trace.json").write_text(json.dumps(trace, indent=2))
    sal = {str(k): v.to_dict() for k, v in stats.items()}
    (OUT / "saliency.json").write_text(json.dumps(sal, indent=2))
    n_experts = next(iter(stats.values())).num_experts
    print(f"saliency layers={len(stats)} experts={n_experts}")

    print("\n--- NESTED PLANS ---")
    plans = {}
    for ratio in RUNGS:
        plan = build_plan(stats, ratio=ratio, min_experts=8)
        plan["arch"] = "step3p7"
        plan["smoke"] = True
        plan["note"] = "local ladder smoke on already-REAPed 148B — not a product prune"
        name = f"plan_p{int(ratio * 100):02d}.json"
        path = OUT / name
        path.write_text(json.dumps(plan, indent=2))
        plans[ratio] = plan
        keep = len(next(iter(plan["layers"].values()))["keep"])
        print(f"  {name}: keep={keep} prune={n_experts - keep}")

    # Nesting check: prune sets must grow
    def prune_set(plan):
        # use one mid MoE layer
        key = sorted(plan["layers"], key=int)[len(plan["layers"]) // 2]
        return set(plan["layers"][key]["prune"])

    nested_ok = True
    prev = set()
    for ratio in RUNGS:
        cur = prune_set(plans[ratio])
        if not prev.issubset(cur):
            nested_ok = False
            print(f"  FAIL nest at {ratio}: prev not subset")
        prev = cur
    print(f"nested prune sets: {'OK' if nested_ok else 'FAIL'}")

    print("\n--- DRY-RUN APPLY ---")
    for ratio in RUNGS:
        plan_path = OUT / f"plan_p{int(ratio * 100):02d}.json"
        apply_plan(str(MODEL), plan_path, OUT / f"dry_p{int(ratio*100):02d}", dry_run=True)

    print("\n--- IN-MEMORY NLL GATE (baseline vs p10) ---")
    # tokenize a few val prompts
    from reap_stream.collect_step3p7 import _tokenize_prompts, _text_config

    val_prompts = prompts[:8]
    model, processor = load(str(MODEL), lazy=True)
    text = _text_model(model)
    cfg = _text_config(model)
    sliding = getattr(cfg, "sliding_window", None)
    lm = getattr(model, "language_model", model)
    lm_head = getattr(lm, "lm_head", None) or getattr(model, "lm_head", None)
    batches = _tokenize_prompts(processor, val_prompts, MAX_TOKENS)

    base_nll = mean_nll(text, lm_head, batches, sliding)
    print(f"  baseline mean NLL: {base_nll:.4f}  PPL~{math.exp(base_nll):.3f}")

    # reload fresh for p10 slice
    del model, text, lm_head
    gc.collect()
    mx.clear_cache()

    model, processor = load(str(MODEL), lazy=True)
    text = _text_model(model)
    lm = getattr(model, "language_model", model)
    lm_head = getattr(lm, "lm_head", None) or getattr(model, "lm_head", None)
    new_n = apply_in_memory(text, plans[0.10])
    # force materialize first moe shapes
    sample_layer = int(sorted(plans[0.10]["layers"], key=int)[0])
    print(f"  p10 in-memory experts={new_n} router={tuple(text.layers[sample_layer].mlp.gate.gate.weight.shape)}")
    p10_nll = mean_nll(text, lm_head, batches, sliding)
    delta = (math.exp(p10_nll) - math.exp(base_nll)) / math.exp(base_nll) * 100
    print(f"  p10 mean NLL: {p10_nll:.4f}  PPL~{math.exp(p10_nll):.3f}  ΔPPL={delta:+.2f}%")

    gate_pass = delta <= 25.0 and math.isfinite(p10_nll)
    summary = {
        "model": str(MODEL),
        "n_experts": n_experts,
        "samples": len(prompts),
        "max_tokens": MAX_TOKENS,
        "nested_ok": nested_ok,
        "baseline_nll": base_nll,
        "p10_nll": p10_nll,
        "delta_ppl_pct": delta,
        "gate_pass_p10": gate_pass,
        "elapsed_s": time.time() - t0,
        "lm_studio_untouched": True,
        "output": str(OUT),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 70)
    print(json.dumps(summary, indent=2))
    print("=" * 70)
    if not nested_ok or not gate_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
