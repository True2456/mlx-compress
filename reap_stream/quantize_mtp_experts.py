"""Quantize the DSpark/MTP drafter's routed experts from their native mxfp4
down to the backbone's AWQ recipe (2-bit gate/up gs128, 3-bit down gs64).

Why this is safe to do aggressively: DSpark is speculative decoding with
*exact rejection sampling* (see omlx/patches/mlx_lm_mtp/deepseek_v4_dspark.py's
module docstring). Every drafted token is verified against the main model, so
quantization error in the drafter costs ACCEPT RATE (speed), never output
correctness. The failure mode of over-quantizing here is "MTP stops helping",
not "model gets dumber".

This is plain RTN (no calibration). The reconstruction error at 2-bit is
substantial (~45% relative on gate/up, ~19% on 3-bit down, measured), which is
exactly the gap AWQ's scale search exists to close -- so treat the resulting
accept rate as the thing to measure, and escalate to a real AWQ calibration
pass over dspark_forward() if it degrades too far.

Format notes (verified against oMLX's own sanitize, deepseek_v4_model.py
~line 2210):
  * On disk the native experts are int8 ``.weight`` + uint8 ``.scale``
    (singular). oMLX reinterprets the weight via ``.view(mx.uint32)`` and
    renames ``.scale`` -> ``.scales``; that pair is mxfp4 (bits=4,
    group_size=32, e8m0 exponent scale, no bias).
  * Affine output carries ``.weight`` (uint32) + ``.scales`` + ``.biases``
    (both bf16). The biases half is only loadable because of the MTP
    expert-stacking fix in omlx/patches/mlx_lm_mtp/deepseek_v4_model.py
    (upstreamed as jundot/omlx#2598) -- stock oMLX drops mtp ``.biases``.

Unchanged shards are hardlinked (same filesystem => zero extra disk), so only
the three MTP-bearing shards are rewritten.

Usage:
    .venv/bin/python -m reap_stream.quantize_mtp_experts \
        --src ~/.lmstudio/models/truemod/DeepSeek-V4-Flash-0731-awq-omlx-raw-v2 \
        --out ~/.lmstudio/models/truemod/DeepSeek-V4-Flash-0731-awq-omlx-raw-v2-mtpq
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path

import mlx.core as mx

# Native source format for the untouched DSpark experts.
SRC_BITS, SRC_GROUP, SRC_MODE = 4, 32, "mxfp4"

# Target recipe, mirroring the backbone v2 build.
# Default: mirror the backbone's AWQ recipe. Overridable from the CLI --
# the drafter is verified by exact rejection sampling, so quantizing it harder
# costs ACCEPT RATE (speed) only, never output correctness. That makes the
# drafter the one place where aggressive bit reduction is a pure size/speed
# trade with no quality risk.
RECIPE = {
    "w1": (2, 128),   # gate_proj
    "w3": (2, 128),   # up_proj
    "w2": (3, 64),    # down_proj
}
PROJ_TO_NAME = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}

EXPERT_RE = re.compile(r"^mtp\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$")


def _requantize_shard(src_path: Path, dst_path: Path, recipe: dict | None = None) -> tuple[int, int, int]:
    """Rewrite one shard, requantizing its mtp expert tensors. Returns
    (n_requantized, src_bytes, dst_bytes)."""
    arrs = mx.load(str(src_path))
    out: dict[str, mx.array] = {}
    n_req = 0

    for key in sorted(arrs):
        m = EXPERT_RE.match(key)
        if m is None:
            out[key] = arrs[key]
            continue
        if m.group(4) == "scale":
            continue  # consumed alongside its .weight

        proj = m.group(3)
        bits, group_size = (recipe or RECIPE)[proj]
        stem = key[: -len(".weight")]
        scale = arrs[f"{stem}.scale"]

        # native mxfp4 -> float
        full = mx.dequantize(
            arrs[key].view(mx.uint32), scale, None,
            group_size=SRC_GROUP, bits=SRC_BITS, mode=SRC_MODE,
        )
        # float -> affine at the target recipe
        qw, qs, qb = mx.quantize(full, group_size=group_size, bits=bits, mode="affine")
        mx.eval(qw, qs, qb)
        out[f"{stem}.weight"] = qw
        out[f"{stem}.scales"] = qs
        out[f"{stem}.biases"] = qb
        del full
        n_req += 1
        if n_req % 128 == 0:
            mx.clear_cache()

    mx.eval(list(out.values()))
    mx.save_safetensors(str(dst_path), out)
    del arrs, out
    mx.clear_cache()
    return n_req, src_path.stat().st_size, dst_path.stat().st_size


def build(src: str, out: str, recipe: dict | None = None) -> None:
    src_p, out_p = Path(src).expanduser(), Path(out).expanduser()
    if out_p.exists():
        raise SystemExit(f"refusing to overwrite existing {out_p}")
    out_p.mkdir(parents=True)

    index = json.loads((src_p / "model.safetensors.index.json").read_text())
    wm = index["weight_map"]

    mtp_shards = {s for k, s in wm.items() if EXPERT_RE.match(k)}
    all_shards = set(wm.values())
    print(f"[mtpq] {len(all_shards)} shards total, {len(mtp_shards)} carry mtp experts", flush=True)

    # Unchanged shards: hardlink (same volume => free).
    for shard in sorted(all_shards - mtp_shards):
        os.link(src_p / shard, out_p / shard)
    print(f"[mtpq] hardlinked {len(all_shards) - len(mtp_shards)} unchanged shards", flush=True)

    t0 = time.time()
    saved = 0
    for shard in sorted(mtp_shards):
        n, sb, db = _requantize_shard(src_p / shard, out_p / shard, recipe)
        saved += sb - db
        print(f"[mtpq] {shard}: {n} tensors requantized, "
              f"{sb/1e9:.2f}GB -> {db/1e9:.2f}GB ({time.time()-t0:.0f}s)", flush=True)

    # weight_map: every mtp expert .scale becomes .scales, plus a new .biases
    new_wm: dict[str, str] = {}
    for k, shard in wm.items():
        m = EXPERT_RE.match(k)
        if m is None:
            new_wm[k] = shard
            continue
        stem = k.rsplit(".", 1)[0]
        new_wm[f"{stem}.weight"] = shard
        new_wm[f"{stem}.scales"] = shard
        new_wm[f"{stem}.biases"] = shard
    total_size = sum((out_p / f).stat().st_size for f in all_shards)
    (out_p / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total_size}, "weight_map": dict(sorted(new_wm.items()))},
        indent=2))

    # config.json: replace the mtp switch_mlp entries (native mxfp4) with ours.
    config = json.loads((src_p / "config.json").read_text())
    quant = dict(config["quantization"])
    n_stages = 1 + max(int(EXPERT_RE.match(k).group(1)) for k in wm if EXPERT_RE.match(k))
    for stage in range(n_stages):
        for proj, (bits, group_size) in (recipe or RECIPE).items():
            path = f"mtp.{stage}.ffn.switch_mlp.{PROJ_TO_NAME[proj]}"
            quant[path] = {"bits": bits, "group_size": group_size, "mode": "affine"}
    config["quantization"] = quant
    config["quantization_config"] = quant
    (out_p / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[mtpq] rewrote {n_stages * 3} mtp quantization entries in config.json", flush=True)

    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
                 "special_tokens_map.json", "chat_template.jinja", "LICENSE", "README.md"):
        p = src_p / name
        if p.exists():
            shutil.copy2(p, out_p / name)
    for folder in ("encoding", "inference"):
        p = src_p / folder
        if p.is_dir():
            shutil.copytree(p, out_p / folder)

    print(f"[mtpq] done in {time.time()-t0:.0f}s, saved {saved/1e9:.2f}GB -> {out_p}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=2,
                    help="gate_proj/up_proj bits (default 2)")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--down-bits", type=int, default=3,
                    help="down_proj bits (default 3). Set 2 for a uniform 2-bit drafter.")
    ap.add_argument("--down-group-size", type=int, default=64)
    a = ap.parse_args()
    recipe = {"w1": (a.bits, a.group_size),
              "w3": (a.bits, a.group_size),
              "w2": (a.down_bits, a.down_group_size)}
    print(f"[mtpq] recipe: gate/up {a.bits}-bit gs{a.group_size}, "
          f"down {a.down_bits}-bit gs{a.down_group_size}", flush=True)
    build(a.src, a.out, recipe)


if __name__ == "__main__":
    main()
