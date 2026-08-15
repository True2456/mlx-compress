"""Apply a REAP plan to a DeepSeek-V4-Flash MLX checkpoint.

Port of apply_step3p7.py's pattern (lazy load, in-place lazy slicing on the
module tree, then model.save_weights) -- proven at 375GB (Step-3.7 bf16),
well past this machine's RAM, so it already streams correctly rather than
needing the shard-level safetensors surgery an earlier draft of this file
assumed was necessary. MLX's lazy arrays mean nothing is materialized until
save_weights evaluates it.

DeepSeek-V4-specific vs. apply_step3p7.py's Step-3.7 layout:
- router weight: layer.ffn.gate.weight (MoEGate), not layer.mlp.gate.gate
- router bias: layer.ffn.gate.e_score_correction_bias -- ONLY present on
  non-hash-routed layers (MoEGate.__init__ sets tid2eid instead, for the
  first `num_hash_layers` layers)
- experts: layer.ffn.switch_mlp.{gate,up,down}_proj -- same SwitchGLU
  convention as Step-3.7/Gemma-4, sliceable identically
- every layer has .ffn unconditionally (no is_moe_layer/enable_moe flag to
  check, unlike Step-3.7/Gemma-4)

Hash-routed layers (layer_idx < num_hash_layers, 3 by default) use REAM
(reap_stream/ream.py) instead of plain deletion. Two reasons this is the
right tool here specifically, not a general re-endorsement of REAM as a
compression strategy -- REAM was built, measured, and REJECTED as a deploy
strategy for Step-3.7 (docs/REAM-RESULT.md: real accuracy flat-to-worse
despite a large PPL "gain" that turned out to be smoothing; independently
confirmed on math/factual AND tool-call accuracy, not just PPL). That
finding was checked against the tokenizer bug in
docs/TOKENIZER-INVESTIGATION.md and is explicitly unaffected by it ("PPL/NLL
evaluations were never confounded... eval numbers in
FINDINGS.md/HEAD8-RESULT.md/REAM-RESULT.md are unaffected" -- the tokenizer
bug only broke AutoTokenizer-dispatch serving, a different code path from
eval_ppl_streamed.py). So REAM is NOT used here for a hoped-for quality win.

It's used because plain deletion is structurally broken for hash layers:
(1) DeepSeek-V4's config has one global n_routed_experts field with no
per-layer override, so every layer must end up the SAME width or the model
fails to load (ValueError: shape mismatch on MoEGate.weight) -- confirmed by
running the plain-deletion version of this script and hitting exactly that.
(2) MoEGate.tid2eid maps each vocabulary token to FIXED expert ids for these
layers; deleting an expert leaves its hash entries pointing at nothing, with
no principled destination for plain pruning to fall back on.

REAM's assign_merges (router-row cosine similarity) gives every deleted
expert a principled destination -- the kept expert it's most similar to --
which solves both problems in one step: merging brings hash layers to the
same width as everywhere else, and the SAME assignment remaps tid2eid
entries to their merged destination's new index. Router weight
(MoEGate.weight) is plain bf16, mergeable directly. switch_mlp's three
projections are mxfp4-quantized (QuantizedSwitchLinear); REAM's merge needs
real values, so those go through dequantize -> merge_experts -> requantize,
verified round-trip-clean on this checkpoint's actual quantization scheme
before use (0.06 max abs diff, 0.7% mean relative -- expected quantization-
grid noise, not corruption).

Non-hash layers keep plain REAP deletion, matching the already-validated
choice for learned routing -- this file does not extend REAM's weight-
merging to them.

## Mixed-precision requantization (--requantize)

Optional second pass, after pruning/merging, that pushes expert weights
below the checkpoint's native mxfp4 (4-bit). Not real DWQ: mlx_lm has no
deepseek_v4 support, so there is no calibrated/distilled path available for
this architecture -- this is round-to-nearest requantization via the same
dequantize -> operate -> requantize machinery already validated for REAM.

A first version of this pass targeted group_size=32 and landed at 121GB
against a 92-98GB target -- traced to mx.quantize's affine mode storing
scales AND biases in the INPUT array's dtype, and that version upcast to
float32 before quantizing, so every 32-weight group paid 8 bytes of float32
overhead (as much as the 2-bit weights themselves: effective ~4 bits/weight,
not 2). Fixed by quantizing in the array's native bf16 (halves the
overhead) and using group_size=128 (needed on top of that to actually reach
target -- group_size=32 with the bf16 fix still lands at ~110GB). The real
cost of that: max abs error at bits=2 goes from 0.063 (group_size=32) to
0.188 (group_size=128) on a real expert tensor from this checkpoint,
measured before committing to the run -- a genuine, non-trivial fidelity
cost paid specifically to hit the requested size, not a free win.

Policy, chosen by measuring actual per-category byte counts on this
checkpoint rather than guessing (experts are 93% of total weight bytes;
gate_proj/up_proj/down_proj are equal-sized, ~41.8GB each at the current
125GB/4-bit state):
- down_proj stays at native 4-bit on every layer -- generally the more
  quantization-sensitive of the three SwiGLU projections.
- gate_proj/up_proj on the `num_hash_layers` hash-routed layers ALSO stay at
  4-bit -- fixed routing can't dynamically compensate for a degraded expert
  the way learned top-k routing might.
- gate_proj/up_proj on every other layer drop to 2-bit, group_size=128.
Computed (bf16-scale-aware math, not the original naive estimate) to land
at ~95GB (target was 92-98GB), verified by summing this checkpoint's real
per-category tensor bytes before running the actual pass.

Usage:
    .venv/bin/python -m reap_stream.apply_deepseek_v4 \
        --model models/DeepSeek-V4-Flash-fp8 \
        --plan artifacts/deepseek-v4-reap/pruning-plan.json \
        --output models/DeepSeek-V4-Flash-fp8-reap \
        --requantize \
        --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_vlm import load

from .ream import router_similarity, assign_merges, merge_experts


def _text_model(model):
    lm = getattr(model, "language_model", None) or model
    return getattr(lm, "model", lm)


def _slice_first_dim(module: Any, keep: list[int], keys: tuple[str, ...] | None = None) -> None:
    idx = mx.array(keep)
    names = keys if keys is not None else ("weight", "scales", "biases", "bias")
    for name in names:
        if name in module and module[name] is not None:
            module[name] = module[name][idx]


def _dequant(module) -> mx.array:
    return mx.dequantize(
        module.weight,
        module.scales,
        module.get("biases"),
        group_size=module.group_size,
        bits=module.bits,
        mode=module.mode,
    )


def _requant_inplace(
    module,
    merged_float: mx.array,
    group_size: int | None = None,
    bits: int | None = None,
    mode: str | None = None,
) -> None:
    """Requantize into `module`. Defaults to the module's OWN current
    group_size/bits/mode (used by the REAM merge path, same scheme in and
    out); pass explicit values to change the scheme (used by requantize_expert_bits
    to push experts to a lower bit depth).

    Deliberately does NOT upcast to float32 first: mx.quantize's affine mode
    stores scales/biases in the INPUT array's dtype (verified: bf16 in ->
    bf16 scales+biases, float32 in -> float32 scales+biases, double the
    bytes for identical precision). An earlier version of this function
    forced float32 here and it silently doubled the per-group overhead --
    caught only because the resulting checkpoint came out ~121GB against a
    92-98GB target instead of the ~95GB computed from (wrongly) assuming
    zero/negligible overhead. merged_float is already bf16 from _dequant."""
    out = mx.quantize(
        merged_float,
        group_size=group_size if group_size is not None else module.group_size,
        bits=bits if bits is not None else module.bits,
        mode=mode if mode is not None else module.mode,
    )
    module.weight, module.scales = out[0], out[1]
    module.group_size = group_size if group_size is not None else module.group_size
    module.bits = bits if bits is not None else module.bits
    module.mode = mode if mode is not None else module.mode
    if len(out) > 2 and out[2] is not None:
        module.biases = out[2]
    elif hasattr(module, "biases"):
        module.biases = None


def _ream_hash_layer(layer, layer_plan: dict) -> dict:
    """REAM (merge) a hash-routed layer to layer_plan's keep width, and remap
    its tid2eid table to match. See module docstring for why this layer type
    needs merging rather than plain deletion."""
    keep = layer_plan["keep"]
    prune = layer_plan["prune"]
    scores = layer_plan["scores"]

    gate = layer.ffn.gate
    router_rows = gate.weight  # (E, H) bf16, not quantized
    mx.eval(router_rows)
    sim = router_similarity(router_rows, mx)
    groups = assign_merges(keep, prune, sim)  # {kept_id: [absorbed pruned ids]}

    gate.weight = merge_experts(router_rows, keep, groups, scores, mx).astype(router_rows.dtype)

    sg = layer.ffn.switch_mlp
    for proj_name in ("gate_proj", "up_proj", "down_proj"):
        proj = getattr(sg, proj_name)
        full = _dequant(proj)
        mx.eval(full)
        merged = merge_experts(full, keep, groups, scores, mx)
        _requant_inplace(proj, merged)

    # Remap tid2eid: every old expert id (kept or pruned) -> its new
    # post-merge index. Kept ids renumber by position in the sorted `keep`
    # list (matching merge_experts' own output order); pruned ids inherit
    # the new index of whichever kept expert absorbed them.
    new_id_of = {old: i for i, old in enumerate(keep)}
    pruned_to_kept = {p: k for k, ps in groups.items() for p in ps}
    n_experts_total = len(keep) + len(prune)  # keep/prune partition the full expert set
    remap = np.zeros(n_experts_total, dtype=np.int64)
    for old in range(n_experts_total):
        remap[old] = new_id_of[old] if old in new_id_of else new_id_of[pruned_to_kept[old]]
    old_tid2eid = np.array(gate.tid2eid)
    gate.tid2eid = mx.array(remap[old_tid2eid], dtype=mx.int32)

    return {"kept": len(keep), "merge_groups": len(groups), "absorbed": len(prune)}


def requantize_expert_bits(
    text,
    num_hash_layers: int,
    low_bits: int = 2,
    low_group_size: int = 128,
) -> dict:
    """Second pass: push expert weights below the checkpoint's native 4-bit,
    at a fixed mixed policy (see module docstring for the byte-count math
    behind it and why down_proj / hash-layer gate+up stay protected).

    Runs after apply_plan's main REAP/REAM loop, so it operates on the
    already-pruned (218-wide) switch_mlp modules, not the original 256.

    Returns touched_paths: {full_model_relative_path: {group_size,bits,mode}}
    for every module actually changed -- apply_plan needs this to write an
    explicit config["quantization"] override. Reload auto-detects a SINGLE
    uniform scheme from the raw-fp8 quantization_config field (verified via
    make_quantization_config's own path keys matching nn.quantize's real
    traversal exactly: "language_model.model.layers.{i}.ffn.switch_mlp.{proj}"
    for this model, since _is_text_model is unset here so quantized_model IS
    the full model, not a narrower .language_model._model). Leaving that
    auto-detected uniform scheme in place after this pass would reconstruct
    every switch_mlp module as native 4-bit on reload and then fail to load
    our actually-mixed-bit saved tensors into it."""
    n_layers = len(text.layers)
    touched = {"low_bits": low_bits, "low_group_size": low_group_size, "layers": []}
    touched_paths: dict[str, dict] = {}
    for i in range(n_layers):
        sg = text.layers[i].ffn.switch_mlp
        protect_gate_up = i < num_hash_layers
        layer_bits = {}
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            proj = getattr(sg, proj_name)
            if proj_name == "down_proj" or protect_gate_up:
                layer_bits[proj_name] = proj.bits  # unchanged, native (4-bit)
                continue
            full = _dequant(proj)
            mx.eval(full)
            _requant_inplace(proj, full, group_size=low_group_size, bits=low_bits, mode="affine")
            mx.eval(proj.weight, proj.scales)
            # This project's own documented highest-leverage memory fix
            # (docs/FINDINGS.md): without this, MLX's allocator cache from
            # each dequantized float tensor (a full 218x2048x4096-class array,
            # ~3.6GB, per projection) accumulates across all 80 dequant/
            # requant cycles in this loop instead of being released between
            # them -- confirmed the hard way: the first version of this pass
            # (no clear_cache here) was silently OOM-killed partway through,
            # empty log, no traceback, matching this project's own earlier
            # OOM signature exactly.
            del full
            mx.clear_cache()
            layer_bits[proj_name] = low_bits
            path = f"language_model.model.layers.{i}.ffn.switch_mlp.{proj_name}"
            touched_paths[path] = {"group_size": low_group_size, "bits": low_bits, "mode": "affine"}
        touched["layers"].append({"layer": i, **layer_bits})
        print(f"[requantize] layer {i:02d}/{n_layers - 1} -> {layer_bits}", flush=True)
    return touched, touched_paths


def apply_plan(
    model_path: str,
    plan_path: str | Path,
    output_dir: str | Path,
    dry_run: bool = False,
    requantize: bool = False,
    low_bits: int = 2,
    low_group_size: int = 128,
) -> Path:
    plan = json.loads(Path(plan_path).read_text())
    model, processor = load(model_path, lazy=True)
    text = _text_model(model)
    num_hash_layers = text.args.num_hash_layers
    n_layers = len(text.layers)

    # Global (not just non-hash) uniform keep-count: DeepSeek-V4's config has
    # one n_routed_experts field with no per-layer override, so EVERY layer
    # -- hash-routed ones too, now REAM-merged rather than deleted -- must
    # end up the same width or the model fails to load.
    keep_counts = {len(v["keep"]) for v in plan["layers"].values()}
    if len(keep_counts) != 1:
        raise ValueError(f"DeepSeek-V4/MLX expects uniform keep counts; got {keep_counts}")
    new_n = keep_counts.pop()

    planned = {int(k) for k in plan["layers"]}
    all_layers = set(range(n_layers))
    if planned - all_layers:
        raise ValueError(f"plan references out-of-range layers: {sorted(planned - all_layers)}")
    if not dry_run and (all_layers - planned):
        missing = sorted(all_layers - planned)
        raise ValueError(
            "Non-dry-run apply requires a plan covering every layer "
            f"(missing {len(missing)} e.g. {missing[:8]}). Use --dry-run for partial plans."
        )

    sliced = []
    reamed = []
    for layer_key, layer_plan in plan["layers"].items():
        i = int(layer_key)
        layer = text.layers[i]
        keep = layer_plan["keep"]

        if i < num_hash_layers:
            result = _ream_hash_layer(layer, layer_plan)
            reamed.append({"layer": i, **result})
            print(f"[apply] layer {i:02d}/{n_layers - 1} REAM-merged -> {result}", flush=True)
            continue

        ffn = layer.ffn
        _slice_first_dim(ffn.gate, keep, keys=("weight",))
        if hasattr(ffn.gate, "e_score_correction_bias"):
            ffn.gate.e_score_correction_bias = ffn.gate.e_score_correction_bias[mx.array(keep)]

        sg = ffn.switch_mlp
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            _slice_first_dim(getattr(sg, proj_name), keep)
        print(f"[apply] layer {i:02d}/{n_layers - 1} pruned -> keep={len(keep)}", flush=True)

        sliced.append({"layer": i, "keep": len(keep), "router": list(ffn.gate.weight.shape)})

    output_dir = Path(output_dir)
    if dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "new_experts": new_n,
                    "layers_sliced": len(sliced),
                    "layers_reamed_hash_routed": reamed,
                    "sample": sliced[:3],
                    "requantize_requested": requantize,
                    "note": "requantize pass is skipped in dry-run (touches every "
                            "layer's gate/up, not just the 3 hash layers -- would "
                            "defeat dry-run's purpose as a fast structural check; "
                            "its bit-depth math and path-key mechanism are already "
                            "verified separately, see module docstring)" if requantize else None,
                    "output": str(output_dir),
                },
                indent=2,
            )
        )
        return output_dir

    requant_summary = None
    touched_paths: dict[str, dict] = {}
    if requantize:
        requant_summary, touched_paths = requantize_expert_bits(
            text, num_hash_layers, low_bits=low_bits, low_group_size=low_group_size
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    src = Path(model_path)
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ):
        p = src / name
        if p.exists():
            shutil.copy2(p, output_dir / name)

    # The raw upstream release ships no chat template anywhere -- not in
    # tokenizer_config.json, not in tokenizer.json, no chat_template.jinja
    # file. DeepSeek uses a custom Python encoding script instead
    # (encoding/encoding_dsv4.py) that never reaches an MLX-format checkpoint.
    # mlx_vlm's own processing_deepseek_v4.py patches templating in at
    # runtime, so generation works fine without this -- but LM Studio's
    # lightweight file-based classifier (not a full model load) looks for a
    # static template as part of building its metadata summary, and appears
    # to fail its ENTIRE config parse (blanking arch and quantization too,
    # both otherwise valid) rather than degrading gracefully when it's
    # missing. Confirmed by direct comparison: mlx-community's own
    # DeepSeek-V4-Flash-2bit-DQ conversion ships exactly this file and
    # displays correctly; ours didn't until this was added. Verified against
    # our own tokenizer.json: all four special tokens the template uses
    # (<begin/end of sentence>, <User>, <Assistant>) are present in
    # added_tokens.
    template_src = Path(__file__).parent / "assets" / "deepseek_v4_chat_template.jinja"
    if template_src.exists():
        shutil.copy2(template_src, output_dir / "chat_template.jinja")

    cfg = json.loads((src / "config.json").read_text())
    # n_routed_experts is top-level in DeepSeek-V4's config.json, no
    # text_config nesting (unlike Gemma-4/Step-3.7) -- verified against the
    # actual downloaded checkpoint's config.json.
    cfg["n_routed_experts"] = new_n
    # LM Studio's loader (mlx_engine/generate.py) picks between two internal
    # code paths on a single check: `"vision_config" in config_json`. Models
    # without it go through mlx_lm, which has no deepseek_v4 entry (confirmed:
    # same gap as this project's own mlx_lm install, stops at deepseek_v32).
    # Models WITH it go through mlx_vlm, which has full deepseek_v4 support,
    # including MTP speculative decoding -- verified present in LM Studio's
    # own bundled environment, just never reached for this model. DeepSeek-V4-
    # Flash is genuinely text-only (no vision_config in the real upstream
    # config.json, no vision fields in mlx_vlm's own ModelConfig for this
    # model_type) -- this key is a pure dispatch trick, not a real capability.
    # mlx_vlm.utils.load_image_processor degrades gracefully for a model class
    # with no ImageProcessor attribute, so an empty dict is sufficient and
    # doesn't require faking any real vision fields. Verified end-to-end: loads
    # and generates correctly via LM Studio's own local server with this set.
    cfg["vision_config"] = {}
    cfg["_reap_note"] = (
        f"pruned to {new_n} experts/layer on layers >= num_hash_layers "
        f"({num_hash_layers}) via plain deletion; hash-routed layers "
        f"{[r['layer'] for r in reamed]} REAM-merged to the same width "
        "instead, with tid2eid remapped to match (see apply_deepseek_v4.py "
        "docstring for why)."
    )

    if requantize:
        # Reload auto-detects a single uniform scheme from the raw-fp8
        # quantization_config field (checked, see module docstring / the
        # empirical path-key verification in requantize_expert_bits). Once
        # this pass makes bit-depth vary per module, that auto-detected
        # uniform scheme is wrong and must be replaced with an explicit
        # config["quantization"] -- baseline from the SAME function
        # load_model would have used (proven correct twice already), with
        # our actually-changed paths overridden on top.
        from mlx_vlm.models.deepseek_v4.language import make_quantization_config

        full_quant_cfg = make_quantization_config(model)
        full_quant_cfg.update(touched_paths)
        cfg["quantization"] = full_quant_cfg
        cfg["quantization_config"] = full_quant_cfg
        cfg["_requantize_note"] = (
            f"expert gate_proj/up_proj requantized to {low_bits}-bit affine "
            f"(group_size={low_group_size}) except down_proj (native, all "
            f"layers) and hash-routed layers {list(range(num_hash_layers))} "
            f"(native, fixed routing can't compensate for a degraded expert). "
            f"{len(touched_paths)} modules changed."
        )

    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (output_dir / "reap-plan.json").write_text(json.dumps(plan, indent=2))

    model.save_weights(str(output_dir / "model.safetensors"))
    print(
        f"Wrote pruned DeepSeek-V4 checkpoint -> {output_dir} "
        f"({new_n} experts/layer; hash layers REAM-merged: {reamed}; "
        f"requantized: {requant_summary is not None})"
    )
    return output_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--requantize", action="store_true",
                     help="also push expert gate/up projections to a lower bit "
                          "depth after pruning/merging -- see module docstring")
    ap.add_argument("--low-bits", type=int, default=2)
    ap.add_argument("--low-group-size", type=int, default=128)
    a = ap.parse_args()
    apply_plan(a.model, a.plan, a.output, dry_run=a.dry_run,
               requantize=a.requantize, low_bits=a.low_bits, low_group_size=a.low_group_size)


if __name__ == "__main__":
    main()
