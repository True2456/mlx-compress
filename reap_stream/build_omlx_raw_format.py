"""Convert our AWQ-calibrated DeepSeek-V4 checkpoint into the raw per-expert
HF key format oMLX's custom deepseek_v4 loader expects, since it does its
OWN sanitize()/fusion from raw HF-style names rather than accepting an
already-mlx_vlm-format checkpoint directly (verified: it loads fine via
plain mlx_vlm.load(), but that isn't the code path oMLX's app actually uses
for this model_type -- see docs discussion; oq.py dispatches to the custom
patch unconditionally whenever config.json's model_type starts with
"deepseek_v4").

Two real things this relies on, both verified against oMLX's own source
(omlx/patches/deepseek_v4/deepseek_v4_model.py's sanitize()):
1. The per-expert fusion loop stacks weight/scales/biases (PLURAL naming)
   under layers.N.ffn.experts.E.{w1,w2,w3}.{weight,scales,biases} into the
   final switch_mlp.{gate,down,up}_proj.{...} tensors it needs. Requires
   the biases-stacking patch already applied to that file (adds "biases" to
   the suffix tuple, which originally only had "weight","scales" -- native
   mxfp4/mxfp8 never has biases, so this was never needed until AWQ's
   affine quantization, which always produces a bias term).
2. Slicing our EXISTING fused (256, out, in) AWQ tensors along axis 0 gives
   bit-exact per-expert tensors -- no dequant/reslice/requant needed, unlike
   apply_deepseek_v4.py's REAP work (which prunes the expert COUNT and
   therefore has to actually recompute). We're keeping all 256 experts,
   just changing their on-disk shape from batched to per-expert, so this is
   a pure reshape/slice, zero precision cost.

Non-expert tensors (attention, shared_experts, embed, lm_head/head,
hc_head, norms, router, mtp) were never touched by AWQ -- copied
byte-for-byte from the ORIGINAL raw checkpoint rather than re-derived from
our mlx_vlm-format AWQ checkpoint, to avoid any risk of getting the tricky
native mxfp4/mxfp8 packing conventions subtly wrong. This also
transparently restores the MTP tensors that got silently dropped when we
first loaded the raw checkpoint via mlx_vlm with strict=False (mlx_vlm's
model class has no MTP submodule) -- MTP was never touched, so it's just as
present in the original download as everything else non-expert.

Reads only -- never opens the AWQ checkpoint or the original raw checkpoint
in write mode. Writes only to --out.

Usage:
    .venv/bin/python -m reap_stream.build_omlx_raw_format \
        --awq-checkpoint ~/.lmstudio/models/truemod/DeepSeek-V4-Flash-0731-awq2bit3bit \
        --raw-checkpoint ~/Desktop/models/DeepSeek-V4-Flash-0731 \
        --out models/DeepSeek-V4-Flash-0731-awq-omlx-raw
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import mlx.core as mx
from mlx_vlm import load


N_ROUTED_EXPERTS = 256
PROJ_MAP = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}


def _unfuse_and_write_experts(awq_path: str, out_dir: Path, shard_index: dict, t0_layers: int):
    """Load the AWQ checkpoint (lazy), slice each layer's fused switch_mlp
    tensors into 256 per-expert tensors, write one shard per layer."""
    model, _ = load(awq_path, lazy=True)
    lm = model.language_model
    text = lm.model
    n_layers = text.args.num_hidden_layers

    for i in range(n_layers):
        sg = text.layers[i].ffn.switch_mlp
        shard_name = f"experts-layer-{i:03d}.safetensors"
        tensors = {}
        for proj_name, raw_name in PROJ_MAP.items():
            proj = getattr(sg, proj_name)
            weight = proj.weight
            scales = proj.scales
            biases = proj.get("biases")
            mx.eval(weight, scales, biases if biases is not None else weight)
            for e in range(N_ROUTED_EXPERTS):
                prefix = f"layers.{i}.ffn.experts.{e}.{raw_name}"
                tensors[f"{prefix}.weight"] = weight[e]
                tensors[f"{prefix}.scales"] = scales[e]
                if biases is not None:
                    tensors[f"{prefix}.biases"] = biases[e]
        mx.eval(list(tensors.values()))
        mx.save_safetensors(str(out_dir / shard_name), tensors)
        for k in tensors:
            shard_index[k] = shard_name
        n_tensors = len(tensors)
        del tensors
        mx.clear_cache()
        print(f"[omlx-raw] layer {i:02d}/{n_layers - 1} experts unfused "
              f"({n_tensors} tensors) -> {shard_name}", flush=True)


def _copy_non_expert_tensors(raw_path: Path, out_dir: Path, shard_index: dict):
    """Copy every tensor NOT under layers.N.ffn.experts.* byte-for-byte from
    the original raw checkpoint into fresh shards."""
    idx = json.loads((raw_path / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    expert_re = re.compile(r"^layers\.\d+\.ffn\.experts\.\d+\.")
    non_expert_keys = [k for k in wm if not expert_re.match(k)]
    print(f"[omlx-raw] {len(non_expert_keys)} non-expert tensors to copy raw "
          f"(attn/shared_experts/embed/head/hc_head/norms/router/mtp)", flush=True)

    by_shard: dict[str, list[str]] = {}
    for k in non_expert_keys:
        by_shard.setdefault(wm[k], []).append(k)

    out_shard_num = 0
    for src_shard, keys in by_shard.items():
        out_shard_num += 1
        out_name = f"non-expert-{out_shard_num:03d}.safetensors"
        arrs = mx.load(str(raw_path / src_shard))
        tensors = {k: arrs[k] for k in keys}
        mx.save_safetensors(str(out_dir / out_name), tensors)
        for k in keys:
            shard_index[k] = out_name
        del arrs, tensors
        mx.clear_cache()
        print(f"[omlx-raw] copied {len(keys)} tensors from {src_shard} -> {out_name}", flush=True)


def build(awq_checkpoint: str, raw_checkpoint: str, out_dir: str):
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    raw_path = Path(raw_checkpoint).expanduser()
    awq_path = str(Path(awq_checkpoint).expanduser())

    shard_index: dict[str, str] = {}

    print("[omlx-raw] step 1/2: copying non-expert tensors raw from original checkpoint", flush=True)
    _copy_non_expert_tensors(raw_path, out, shard_index)

    print("[omlx-raw] step 2/2: unfusing AWQ expert tensors per-layer", flush=True)
    _unfuse_and_write_experts(awq_path, out, shard_index, 0)

    total_size = sum((out / f).stat().st_size for f in set(shard_index.values()))
    index = {"metadata": {"total_size": total_size}, "weight_map": dict(sorted(shard_index.items()))}
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    # config.json: the ORIGINAL raw HF config, not our mlx_vlm-modified one --
    # oMLX's custom loader parses the raw HF config format (model_type,
    # compress_ratios, etc.), not our added "quantization"/"vision_config"
    # dispatch-trick fields.
    shutil.copy2(raw_path / "config.json", out / "config.json")
    for name in ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "LICENSE", "README.md"):
        p = raw_path / name
        if p.exists():
            shutil.copy2(p, out / name)
    for folder in ("encoding", "inference"):
        p = raw_path / folder
        if p.exists():
            shutil.copytree(p, out / folder)

    print(f"[omlx-raw] done -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--awq-checkpoint", required=True)
    ap.add_argument("--raw-checkpoint", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a.awq_checkpoint, a.raw_checkpoint, a.out)


if __name__ == "__main__":
    main()
