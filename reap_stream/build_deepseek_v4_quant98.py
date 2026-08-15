"""Requantize DeepSeek-V4-Flash-0731 experts below their native mxfp4
precision to hit a 90-100GB target -- no REAP pruning, full 256 experts/layer
kept everywhere.

Real per-category byte counts, measured against the live-loaded model (not
the raw HF file -- an earlier pass mis-measured element counts from the raw
per-expert tensor layout and landed at 156GB native, ~1.9x over budget):

    expert gate_proj   49.06 GB  (native mxfp4, bits=4, group_size=32)
    expert up_proj     49.06 GB
    expert down_proj   49.06 GB
    attention           5.55 GB  (native mxfp8 -- left untouched)
    shared_experts       1.12 GB  (native mxfp8 -- left untouched)
    lm_head/embed/other  2.35 GB  (left untouched)

Policy (validated against a real 2-layer requant+measure test, matched
predicted bytes to 4 decimal places):
    - down_proj: 3-bit affine, group_size=128 (kept a notch higher --
      REAP's own findings flagged down_proj as the most quantization-
      sensitive of the three SwiGLU projections)
    - gate_proj / up_proj: 2-bit affine, group_size=128
    - everything else: untouched, native precision

Measured: 2.0804 GB/layer x 43 layers = 89.46GB experts + 9.02GB non-expert
= 98.48GB total (~91.7 GiB).

This has NOT yet been validated for actual quality (PPL sanity check,
chunked-prefill KL vs the raw checkpoint as teacher, real task benchmarks --
see docs/LING3-QUANTIZATION-SESSION-SUMMARY.md for why none of those steps
are optional). 2-bit is the bit width Ling-3.0's own sweep flagged as the
first to show real degradation, though that was a different architecture
(KDA recurrent attention) with a different failure mode than DeepSeek-V4's
MLA -- worth confirming rather than assuming it transfers either way.

Same memory-safety pattern as apply_deepseek_v4.py's requantize_expert_bits:
mx.clear_cache() after each dequant/requant cycle, or MLX's allocator cache
accumulates across all 129 (43 layers x 3 projections) cycles and OOMs
silently.

Source directory is only ever read, never modified.

Usage:
    .venv/bin/python -m reap_stream.build_deepseek_v4_quant98 \
        --model ~/Desktop/models/DeepSeek-V4-Flash-0731 \
        --out models/DeepSeek-V4-Flash-0731-2bit3bit-gs128
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.utils import save_weights


def _dequant(module) -> mx.array:
    return mx.dequantize(
        module.weight,
        module.scales,
        module.get("biases"),
        group_size=module.group_size,
        bits=module.bits,
        mode=module.mode,
    )


def _requant_inplace(module, full: mx.array, bits: int, group_size: int, mode: str = "affine") -> None:
    out = mx.quantize(full, group_size=group_size, bits=bits, mode=mode)
    module.weight, module.scales = out[0], out[1]
    module.group_size, module.bits, module.mode = group_size, bits, mode
    if len(out) > 2 and out[2] is not None:
        module.biases = out[2]
    elif hasattr(module, "biases"):
        module.biases = None


POLICY = {
    "gate_proj": {"bits": 2, "group_size": 128, "mode": "affine"},
    "up_proj": {"bits": 2, "group_size": 128, "mode": "affine"},
    "down_proj": {"bits": 3, "group_size": 128, "mode": "affine"},
}


def build(model_path: str, out_dir: str) -> Path:
    src = Path(model_path).expanduser()
    out = Path(out_dir)

    print(f"[build-dsv4-98] loading (lazy): {src}", flush=True)
    model, processor = load(str(src), lazy=True)
    lm = getattr(model, "language_model", None) or model
    text = getattr(lm, "model", lm)
    n_layers = len(text.layers)

    from mlx_vlm.models.deepseek_v4.language import make_quantization_config
    quant_cfg = make_quantization_config(model)  # baseline: native mxfp4/mxfp8 everywhere

    prefix = "language_model.model" if hasattr(model, "language_model") else "model"
    for i in range(n_layers):
        sg = text.layers[i].ffn.switch_mlp
        for proj_name, spec in POLICY.items():
            proj = getattr(sg, proj_name)
            full = _dequant(proj)
            mx.eval(full)
            _requant_inplace(proj, full, spec["bits"], spec["group_size"], spec["mode"])
            mx.eval(proj.weight, proj.scales)
            del full
            mx.clear_cache()
            quant_cfg[f"{prefix}.layers.{i}.ffn.switch_mlp.{proj_name}"] = dict(spec)
        print(f"[build-dsv4-98] layer {i:02d}/{n_layers - 1} requantized "
              f"(gate/up=2bit up/gs128, down=3bit/gs128)", flush=True)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    print(f"[build-dsv4-98] saving (streams shard-by-shard) -> {out}", flush=True)
    save_weights(out, model, donate_weights=True)

    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ):
        p = src / name
        if p.exists():
            shutil.copy2(p, out / name)

    template_src = Path(__file__).parent / "assets" / "deepseek_v4_chat_template.jinja"
    if template_src.exists():
        shutil.copy2(template_src, out / "chat_template.jinja")

    cfg = json.loads((src / "config.json").read_text())
    cfg["quantization"] = quant_cfg
    cfg["quantization_config"] = quant_cfg
    cfg["vision_config"] = {}
    cfg["_build_note"] = (
        "No REAP pruning -- full 256 experts/layer. Requantized below native "
        "mxfp4: gate_proj/up_proj to 2-bit affine (group_size=128), down_proj "
        "to 3-bit affine (group_size=128). Attention/shared-experts/embed/"
        "lm_head left at native precision. Measured target ~98.5GB "
        "(~91.7 GiB). NOT yet quality-validated -- see docs/ for the pending "
        "PPL/KL-divergence/real-benchmark validation plan."
    )
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    print(f"[build-dsv4-98] done -> {out}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a.model, a.out)


if __name__ == "__main__":
    main()
