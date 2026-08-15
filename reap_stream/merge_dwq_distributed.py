"""Merge both ranks' DWQ checkpoint shards into a full, loadable model dir.

The distributed trainer saves only each rank's own trainable scales:
    <ckpt-dir>/rank0/trainable_scales.safetensors   (its layer range)
    <ckpt-dir>/rank1/trainable_scales.safetensors   (its layer range)
Together these cover every layer exactly once. This script clones the original
quantized model directory and overwrites just the trained scale tensors inside
the affected shards, leaving everything else (weights, biases, configs,
tokenizer, index) untouched.

Two details that matter:

* Key prefix. Checkpoint keys are relative to `model.language_model`, e.g.
  "model.layers.10.ffn.switch_mlp.gate_proj.scales", while the on-disk shards
  use the full VLM-shaped name "language_model.model.layers.10....". The
  prefix is added here.

* The clone uses APFS copy-on-write (cp -c), so duplicating the ~94GB model is
  instant and consumes almost no extra disk until a shard is actually rewritten
  -- only the handful of shards holding switch_mlp scales get new blocks.

Usage:
    python -m reap_stream.merge_dwq_distributed \
        --source models/DeepSeek-V4-Flash-0731-awq2bit3bit-v2 \
        --ckpt-dir artifacts/dwq-pilot-ckpt \
        --out models/DeepSeek-V4-Flash-0731-awq2bit3bit-v2-dwq
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import mlx.core as mx

PREFIX = "language_model."


def _load_rank_checkpoints(ckpt_dir: Path) -> dict[str, mx.array]:
    merged: dict[str, mx.array] = {}
    found = []
    for rank_dir in sorted(ckpt_dir.glob("rank*")):
        wp = rank_dir / "trainable_scales.safetensors"
        if not wp.exists():
            continue
        d = mx.load(str(wp))
        overlap = set(d) & set(merged)
        if overlap:
            raise ValueError(
                f"{rank_dir.name} overlaps an earlier rank on {len(overlap)} keys "
                f"(e.g. {sorted(overlap)[:3]}) -- ranks must own disjoint layers")
        merged.update(d)
        sj = rank_dir / "state.json"
        step = json.loads(sj.read_text()).get("step") if sj.exists() else "?"
        found.append(f"{rank_dir.name}: {len(d)} tensors @ step {step}")
    if not found:
        raise FileNotFoundError(f"no rank*/trainable_scales.safetensors under {ckpt_dir}")
    for f in found:
        print(f"  {f}")
    return merged


# Raw-byte view for each dtype we may patch. safetensors stores tensors as a
# flat little-endian buffer, and MLX's .view() to a same-width unsigned int
# reproduces those bytes exactly (verified against on-disk output for bfloat16,
# which numpy cannot represent natively).
_BYTE_VIEW = {
    mx.bfloat16: mx.uint16, mx.float16: mx.uint16,
    mx.float32: mx.uint32, mx.uint32: mx.uint32,
    mx.uint8: mx.uint8, mx.int8: mx.uint8,
}


def _raw_bytes(arr: mx.array) -> bytes:
    import numpy as np
    view_dtype = _BYTE_VIEW.get(arr.dtype)
    if view_dtype is None:
        raise SystemExit(f"no raw-byte view registered for dtype {arr.dtype}")
    return np.array(arr.view(view_dtype)).tobytes()


def _patch_shard_inplace(path: Path, keys: dict[str, str],
                         trained: dict[str, mx.array]) -> int:
    """Overwrite specific tensors' bytes inside a safetensors file.

    Only valid because each trained tensor has the same shape and dtype as the
    one it replaces, so its byte length is identical and the file layout,
    header, and every other tensor stay untouched. Rewriting whole shards would
    instead break the APFS clone and cost ~101GB of new disk for this model.
    Length is asserted per tensor -- a mismatch aborts before writing anything.
    """
    import struct
    with open(path, "r+b") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hdr_len))
        data_start = 8 + hdr_len

        payloads = []
        for full, ck in keys.items():
            meta = header[full]
            start, end = meta["data_offsets"]
            want = end - start
            new = trained[ck]
            dtype = getattr(mx, str(meta["dtype"]).lower().replace("bf16", "bfloat16")
                            .replace("f32", "float32").replace("f16", "float16"), None)
            if dtype is not None and new.dtype != dtype:
                new = new.astype(dtype)
            raw = _raw_bytes(new)
            if len(raw) != want:
                raise SystemExit(
                    f"{path.name}:{full} byte length {len(raw)} != on-disk {want} "
                    "-- refusing to patch (shape/dtype changed?)")
            payloads.append((start, raw))

        for start, raw in payloads:  # only write once every tensor validated
            f.seek(data_start + start)
            f.write(raw)
        f.flush()
    return len(keys)


def merge(source: Path, ckpt_dir: Path, out: Path, force: bool = False) -> None:
    if out.exists():
        if not force:
            raise SystemExit(f"{out} already exists (use --force to replace)")
        shutil.rmtree(out)

    print(f"[merge] loading rank checkpoints from {ckpt_dir}")
    trained = _load_rank_checkpoints(ckpt_dir)
    print(f"[merge] {len(trained)} trained tensors total")

    index_path = source / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]

    # Map trained tensors -> the shard that holds them, verifying each exists.
    by_shard: dict[str, dict[str, str]] = defaultdict(dict)
    missing = []
    for k in trained:
        full = PREFIX + k
        shard = weight_map.get(full)
        if shard is None:
            missing.append(full)
        else:
            by_shard[shard][full] = k
    if missing:
        raise SystemExit(
            f"{len(missing)} trained tensors have no home in the index, e.g. "
            f"{missing[:3]} -- wrong --source model for these checkpoints?")

    print(f"[merge] cloning {source} -> {out} (APFS copy-on-write)")
    r = subprocess.run(["cp", "-Rc", str(source), str(out)], capture_output=True, text=True)
    if r.returncode != 0:  # non-APFS or cross-volume: fall back to a real copy
        print(f"[merge] clone unavailable ({r.stderr.strip()[:80]}); copying instead")
        shutil.copytree(source, out)

    print(f"[merge] patching {len(by_shard)} of {len(set(weight_map.values()))} shards "
          f"in place")
    total = 0
    for shard, keys in sorted(by_shard.items()):
        n = _patch_shard_inplace(out / shard, keys, trained)
        total += n
        print(f"  {shard}: patched {n} tensors")
    print(f"[merge] {total} tensors patched")

    (out / "dwq_merge_info.json").write_text(json.dumps({
        "source": str(source), "ckpt_dir": str(ckpt_dir),
        "trained_tensors": len(trained), "shards_rewritten": sorted(by_shard),
    }, indent=2))
    print(f"[merge] done -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="original quantized model dir")
    ap.add_argument("--ckpt-dir", required=True, help="dir containing rank0/ rank1/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    merge(Path(a.source), Path(a.ckpt_dir), Path(a.out), a.force)


if __name__ == "__main__":
    main()
