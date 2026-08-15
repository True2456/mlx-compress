"""Repack the raw DeepSeek-V4-Flash-0731 HF release into a properly packed
MLX checkpoint -- no REAP pruning, no additional requantization, no extra
precision loss.

Key finding (verified directly against the downloaded checkpoint, not
assumed): the raw HF release already ships routed experts at native mxfp4
(4-bit) and attention/shared-expert projections at native mxfp8 (8-bit).
mlx_vlm's deepseek_v4 loader auto-detects this straight from the raw
safetensors -- confirmed by lazy-loading the checkpoint and inspecting a
live module: `switch_mlp.gate_proj` comes back as a `QuantizedSwitchLinear`
with bits=4, group_size=32, mode="mxfp4"; `attn.wq_b` comes back as a
`QuantizedLinear` with bits=8, mode="mxfp8". This matches
`mlx_vlm.models.deepseek_v4.language.make_quantization_config`'s hardcoded
scheme exactly, which is the architecture's own reference recipe, not a
guess.

The raw safetensors files on disk are ~167GB despite this, because the
per-expert tensors are stored as one byte per 4-bit value (I8 dtype
container) instead of MLX's packed representation (8 values per uint32) --
confirmed by comparing the raw file's per-category byte count
(reap_stream/ tooling: 148.176B expert-weight elements, all stored as I8)
against the loaded module's actual packed shape
((256, 2048, 512) uint32 for a (256, 2048, 4096)-logical gate_proj, i.e.
4096 input dims packed 8-per-uint32 = 512).

So this script does NOT dequantize/reslice/requantize anything. It just
lazy-loads (which parses the raw bytes into MLX's packed quantized-array
form) and re-saves through mlx_vlm.utils.save_weights, which is the fix for
that on-disk storage waste. Computed target from real per-category byte
counts: ~78.7GB experts (mxfp4, 4.25 eff bits/weight) + ~9.4GB everything
else (already near-native precision) =~ 88GB, inside the 90-100GB target
band -- see docs/DEEPSEEK-V4-QUANT-PLAN.md for the full byte-count
derivation.

Source directory is only ever read, never modified.

Usage:
    .venv/bin/python -m reap_stream.build_deepseek_v4_native_quant \
        --model ~/Desktop/models/DeepSeek-V4-Flash-0731 \
        --out models/DeepSeek-V4-Flash-0731-native-repack
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from mlx_vlm import load
from mlx_vlm.utils import save_weights


def build(model_path: str, out_dir: str) -> Path:
    src = Path(model_path).expanduser()
    out = Path(out_dir)

    print(f"[build-dsv4] loading (lazy, auto-detects native mxfp4/mxfp8): {src}", flush=True)
    model, processor = load(str(src), lazy=True)

    from mlx_vlm.models.deepseek_v4.language import make_quantization_config
    quant_cfg = make_quantization_config(model)
    print(f"[build-dsv4] detected quantization scheme: "
          f"{len(quant_cfg) - 3} per-path overrides "
          f"(default group_size={quant_cfg['group_size']}, bits={quant_cfg['bits']}, "
          f"mode={quant_cfg['mode']})", flush=True)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    print(f"[build-dsv4] saving (streams shard-by-shard, no dequant) -> {out}", flush=True)
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

    # Same rationale as apply_deepseek_v4.py: no upstream Jinja template
    # exists, but LM Studio's lightweight classifier needs a static one to
    # avoid blanking its whole config summary.
    template_src = Path(__file__).parent / "assets" / "deepseek_v4_chat_template.jinja"
    if template_src.exists():
        shutil.copy2(template_src, out / "chat_template.jinja")

    cfg = json.loads((src / "config.json").read_text())
    cfg["quantization"] = quant_cfg
    cfg["quantization_config"] = quant_cfg
    cfg["vision_config"] = {}  # same LM Studio dispatch trick as apply_deepseek_v4.py
    cfg["_build_note"] = (
        "Repacked from raw deepseek-ai/DeepSeek-V4-Flash-0731 -- no REAP pruning, "
        "no additional requantization beyond the checkpoint's own native "
        "mxfp4 (experts) / mxfp8 (attention, shared experts) precision. "
        "Fixes the raw release's on-disk storage waste (I8-per-4-bit-value) "
        "via mlx_vlm's packed quantized-array representation."
    )
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    print(f"[build-dsv4] done -> {out}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a.model, a.out)


if __name__ == "__main__":
    main()
