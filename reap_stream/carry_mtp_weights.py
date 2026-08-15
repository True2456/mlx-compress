"""Carry a Qwen3.5 checkpoint's mtp.* weights alongside a quantized build.

mlx_vlm's Model.sanitize DROPS every `mtp.*` key at load, so a build saved from
a loaded model loses the MTP head entirely (measured: 15 tensors, 0.849GB).
Without them the checkpoint can never yield a speculative drafter without
keeping the 55.6GB original around.

They CANNOT simply be added to the main shards. sanitize does:

    def should_shift_norm_weights(weights):
        has_mtp_weights = any("mtp." in key for key in weights)
        ...
        return has_mtp_weights or has_unsanitized_conv1d

and a True there re-applies a `+1.0` offset to every layernorm -- values that
already have it baked in would be silently corrupted. So the weights go to a
SUBDIRECTORY: mlx_vlm loads only shards named in model.safetensors.index.json,
and its fallback `glob(model_path/"*.safetensors")` is non-recursive, so a
nested file is invisible to both paths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="original bf16 checkpoint")
    ap.add_argument("--target", required=True, help="quantized build to augment")
    a = ap.parse_args()

    src, dst = Path(a.source).expanduser(), Path(a.target).expanduser()
    mtp = {}
    for f in sorted(src.glob("*.safetensors")):
        w = mx.load(str(f))
        mtp.update({k: v for k, v in w.items() if k.startswith("mtp.")})
    if not mtp:
        raise SystemExit(f"no mtp.* tensors in {src}")

    out_dir = dst / "mtp"
    out_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out_dir / "mtp_weights.safetensors"), mtp,
                        metadata={"format": "mlx", "source": str(src)})
    nbytes = sum(v.nbytes for v in mtp.values())
    (out_dir / "README.md").write_text(
        "# MTP head weights (carried, not active)\n\n"
        f"{len(mtp)} `mtp.*` tensors ({nbytes/1e9:.3f} GB, bf16) from `{src.name}`.\n\n"
        "These are NOT loaded by mlx_vlm: `Model.sanitize` strips `mtp.*`, and "
        "this file sits outside `model.safetensors.index.json` so the loader "
        "never reads it. They are kept so a `qwen3_5_mtp` speculative drafter "
        "can be built later without re-downloading the bf16 original.\n\n"
        "**Do not move this file into the model directory root.** Its presence "
        "in the loaded weight set flips `should_shift_norm_weights()` to True, "
        "which re-applies a `+1.0` offset to layernorms that already have it.\n"
    )
    print(f"[mtp] wrote {len(mtp)} tensors ({nbytes/1e9:.3f} GB) -> {out_dir}")
    for k in sorted(mtp)[:4]:
        print(f"   {k}  {tuple(mtp[k].shape)}")


if __name__ == "__main__":
    main()
