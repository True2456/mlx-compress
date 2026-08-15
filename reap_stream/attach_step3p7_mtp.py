"""Attach Step-3.7's MTP draft layers (45/46/47) to our pruned/quantized checkpoint.

Our converted checkpoint only carries layers 0-44 (num_hidden_layers). The 3
DeepSeek-V3-style nextn/MTP layers upstream never made it through the original
raw-checkpoint conversion, even though config.json's text_config still
declares num_nextn_predict_layers=3 -- oMLX detects that mismatch and
correctly (silently) skips MTP attachment rather than crashing.

The upstream MTP weights live entirely in one shard, model-00024.safetensors
(51 keys for layers 45-47, plus 3 incidental duplicate keys -- lm_head,
embed_tokens, model.norm -- already present in our checkpoint at 8-bit, so
skipped here). This script extracts just the 51 MTP keys, quantizes the
linear projections to the checkpoint's base policy (4-bit, group_size=64,
affine -- matching every ordinary layer's default), leaves norm vectors
native per the existing precedent (input_layernorm/k_norm/etc are plain
.weight with no .scales/.biases anywhere in the checkpoint), adds the
language_model. prefix to match this checkpoint's VLM-wrapper key naming,
and writes them as a new shard appended to the existing sharded checkpoint --
no need to touch or reload the other 93GB/2482 keys already on disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx

NORM_SUFFIXES = ("enorm.weight", "hnorm.weight", "input_layernorm.weight",
                  "post_attention_layernorm.weight", "self_attn.k_norm.weight",
                  "self_attn.q_norm.weight")
SKIP_KEYS = {"lm_head.weight", "model.embed_tokens.weight", "model.norm.weight"}


def is_norm_key(key: str) -> bool:
    return any(key.endswith(s) for s in NORM_SUFFIXES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True, help="path to model-00024.safetensors (raw bf16, from upstream)")
    ap.add_argument("--checkpoint", required=True, help="our existing sharded checkpoint dir")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--mode", default="affine")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    ckpt = Path(a.checkpoint)
    index_path = ckpt / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]

    existing_shards = sorted(set(weight_map.values()))
    shard_nums = [int(s.split("-")[1]) for s in existing_shards]
    total_shards_old = max(shard_nums)
    new_shard_name = f"model-{total_shards_old + 1:05d}-of-{total_shards_old + 1:05d}.safetensors"
    print(f"existing shards: {len(existing_shards)}, new shard will be {new_shard_name}")

    raw = mx.load(a.shard)
    mtp_keys = [k for k in raw if k not in SKIP_KEYS and (".layers.45." in k or ".layers.46." in k or ".layers.47." in k)]
    print(f"found {len(mtp_keys)} MTP tensors in shard")

    out: dict[str, mx.array] = {}
    quantized, native = 0, 0
    for key in sorted(mtp_keys):
        arr = raw[key]
        new_key = "language_model." + key
        if is_norm_key(key) or arr.ndim != 2:
            if is_norm_key(key):
                # ZeroCenteredRMSNorm stores weight=ones(dims) and applies it
                # directly as the RMSNorm scale. Raw HF checkpoints ship the
                # Gemma-style zero-centered convention (0 == identity), which
                # step3p5.Model.sanitize() corrects with `v + 1` for every
                # ordinary layer -- confirmed independently by the reference
                # C++ implementation's own comment ("Gemma-style, weights
                # already contain +1 offset"). That sanitize pass only fires
                # when it detects raw HF-style keys anywhere in the full
                # weight dict; our checkpoint's other ~2482 tensors already
                # went through it in a prior conversion and use the
                # post-shift convention, so the whole-dict is_vanilla check
                # will not re-apply it here. Apply it now, once, at
                # extraction time, so these norms match everything else.
                arr = arr + 1
            out[new_key] = arr
            native += 1
        else:
            w, scales, biases = mx.quantize(arr, group_size=a.group_size, bits=a.bits, mode=a.mode)
            base = new_key[: -len(".weight")] if new_key.endswith(".weight") else new_key
            out[base + ".weight"] = w
            out[base + ".scales"] = scales
            if biases is not None:
                out[base + ".biases"] = biases
            quantized += 1

    print(f"quantized {quantized} projection(s) to {a.bits}-bit/gs{a.group_size}, "
          f"kept {native} norm/other tensor(s) native")
    print(f"output tensors: {len(out)}, keys: {sorted(out.keys())[:6]} ...")

    if a.dry_run:
        print("[dry-run] not writing anything")
        return

    mx.eval(*out.values())
    mx.save_safetensors(str(ckpt / new_shard_name), out, metadata={"format": "mlx"})

    for k in out:
        weight_map[k] = new_shard_name
    index["metadata"]["total_size"] = index["metadata"].get("total_size", 0) + sum(
        v.nbytes for v in out.values()
    )
    index_path.write_text(json.dumps(index, indent=2))

    # Existing shard filenames keep their original "of-{total_shards_old}"
    # suffix -- renumbering all 20 of them to reflect the new count isn't
    # required (weight_map is the source of truth for which file holds which
    # key) and isn't worth the risk of touching files that already work.
    print(f"wrote {new_shard_name} ({sum(v.nbytes for v in out.values())/1024**3:.2f} GiB), "
          f"updated index with {len(out)} new keys")


if __name__ == "__main__":
    main()
