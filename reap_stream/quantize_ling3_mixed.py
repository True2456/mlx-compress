"""Build a Ling-3.0-flash quantized checkpoint with a custom mixed-precision
policy, matching the existing 8fixed-5routed layout but with a configurable
routed-expert bit width.

Tests the hypothesis in docs/LING3-LONG-CONTEXT-QUANT-FINDINGS.md: that
5-bit routed-expert RTN quantization noise compounds through KDA's
recurrent state over long context and destabilizes by ~14K tokens. If
raising routed-expert precision delays or removes that onset
(reap_stream/divergence_rate_by_length.py against this checkpoint), that
confirms the noise-into-recurrence hypothesis and identifies the fix.

Lazy-loads the BF16 teacher and quantizes+saves shard-by-shard (same
memory-safe pattern as scripts/build_student.py / mlx_vlm's quantize_model),
so the 237GB teacher is never resident.

Usage:
    .venv/bin/python -m reap_stream.quantize_ling3_mixed \
        --model models/Ling-3.0-flash \
        --out artifacts/ling3-8fixed-6routed \
        --routed-bits 6 --fixed-bits 8 --group-size 64
"""
from __future__ import annotations

import argparse
import glob
import shutil
from pathlib import Path

from mlx_lm.utils import (
    load,
    quantize_model,
    save_config,
    save_model,
)


# layer_group_size=6 in config.json -> global (MLA) layers are every 6th,
# 0-indexed: (i+1) % 6 == 0 or i >= (42 // 6) * 6. Established empirically
# via reap_stream.collect_ling3.inspect_model on models/Ling-3.0-flash.
GLOBAL_ATTENTION_LAYERS = {5, 11, 17, 23, 29, 35, 41}


def build(model_path: str, out_dir: str, routed_bits: int, fixed_bits: int,
          group_size: int, exclude_kda_attention: bool = False,
          exclude_router: bool = False):
    src = Path(model_path)
    print(f"[quantize-ling3] loading (lazy): {src}", flush=True)
    model, tokenizer, config = load(str(src), lazy=True, return_config=True)

    def quant_predicate(path: str, module):
        # mlx_lm.utils.quantize_model's wrapped_predicate already restricts
        # calls to this to modules with to_quantized() and a group_size-
        # divisible weight dim, so no need to re-check quantizability here.
        if exclude_kda_attention and ".attention." in path:
            parts = path.split(".")
            layer_idx = int(parts[2])
            if layer_idx not in GLOBAL_ATTENTION_LAYERS:
                return False  # KDA layer's attention weights: leave BF16
        if exclude_router and ".mlp.gate.gate_proj" in path:
            return False  # MoE router weight: leave BF16
        bits = routed_bits if "switch_mlp" in path else fixed_bits
        return {"group_size": group_size, "bits": bits}

    print(f"[quantize-ling3] quantizing: routed(switch_mlp)={routed_bits}-bit, "
          f"everything-else={fixed_bits}-bit, group_size={group_size}, "
          f"exclude_kda_attention={exclude_kda_attention}, "
          f"exclude_router={exclude_router}", flush=True)
    q_model, q_config = quantize_model(
        model, config, group_size=group_size, bits=fixed_bits,
        quant_predicate=quant_predicate,
    )

    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    print(f"[quantize-ling3] saving -> {out}", flush=True)
    save_model(out, q_model, donate_model=True)
    save_config(q_config, config_path=out / "config.json")
    tokenizer.save_pretrained(out)
    # NOTE: deliberately NOT copying *.py (configuration_bailing_moe_v3.py /
    # modeling_bailing_moe_v3.py) into the output. mlx_lm/oMLX never uses
    # them -- it has its own native bailing_hybrid.py -- but config.json's
    # auto_map references them, and their mere presence makes transformers
    # try to actually load custom code via that auto_map, which broke
    # tokenizer loading in oMLX (TokenizersBackend has no attribute
    # get_added_tokens_decoder) even though the .py files themselves were
    # untouched. The working reference checkpoint (ling3-8fixed-5routed)
    # doesn't have them either. Confirmed fix: removing them from a built
    # checkpoint made it load correctly.
    for pattern in ("generation_config.json",):
        for f in glob.glob(str(src / pattern)):
            shutil.copy(f, out)
    print(f"[quantize-ling3] done: {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--routed-bits", type=int, default=6)
    ap.add_argument("--fixed-bits", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--exclude-kda-attention", action="store_true",
                    help="Leave KDA (non-global) layers' attention weights unquantized (BF16)")
    ap.add_argument("--exclude-router", action="store_true",
                    help="Leave the MoE router (gate.gate_proj) unquantized (BF16)")
    a = ap.parse_args()
    build(a.model, a.out, a.routed_bits, a.fixed_bits, a.group_size,
          a.exclude_kda_attention, a.exclude_router)
