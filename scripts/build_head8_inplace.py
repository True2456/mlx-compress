#!/usr/bin/env python3
"""Build shared8-head8 from the existing deploy checkpoint, without rebuilding it.

`head8` differs from the deployed shared8 student in exactly two tensors --
`lm_head` and `embed_tokens`, requantized 4-bit -> 8-bit. A full tomography
build regenerates all 93 GB, which this machine no longer has room for (and
which risks differing elsewhere). Instead: hardlink the 18 untouched shards,
rewrite only the two that hold the head tensors, and add the two per-path
entries to config.json.

Costs ~6 GB and a couple of minutes instead of 93 GB and a full requantize,
and gives a cleaner experiment -- provably the deployed weights with two
tensors changed, rather than a fresh quantization run.

Rationale for testing it at all (SHARED8-RESULT.md): components that fire on
every token carry the quantization damage, because they get no implicit
smoothing from top-k partial activation. lm_head/embed_tokens are the extreme
case and were never a sweep variant -- the build predicate returns bare `True`
for them, so they sit at 4-bit by default. Costs +0.53 GB; digit-row
perturbation drops from 12% of the tightest argmax margin to 0.7%.

Usage:
    .venv/bin/python scripts/build_head8_inplace.py \
        --src models/Step-3.7-p15-4bit-vblend-shared8 \
        --out models/Step-3.7-p15-vblend-shared8-head8
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import mlx.core as mx

HEAD_PATHS = ("language_model.lm_head", "language_model.model.embed_tokens")
GROUP_SIZE, NEW_BITS, OLD_BITS = 64, 8, 4


def requantize_shard(src_shard: Path, out_shard: Path, targets: list[str],
                     bf16_source: dict[str, mx.array]):
    """Copy a shard through, replacing the named tensors with 8-bit quants of
    the ORIGINAL BF16 weights.

    Quantizing the student's existing 4-bit head to 8-bit recovers nothing --
    it stores 4-bit-quality values in 8-bit containers and adds a second
    rounding pass, which measured +0.013 NLL *worse* than leaving it at 4-bit.
    The 8-bit weights must come from the BF16 base.
    """
    w = mx.load(str(src_shard))
    out = {}
    done = []
    for name, val in w.items():
        stem = name.rsplit(".", 1)[0]
        leaf = name.rsplit(".", 1)[-1]
        if stem in targets and leaf in ("weight", "scales", "biases"):
            if leaf != "weight":
                continue                      # emitted alongside the weight
            ref = bf16_source[stem]
            q, s, b = mx.quantize(ref, group_size=GROUP_SIZE, bits=NEW_BITS)
            out[f"{stem}.weight"], out[f"{stem}.scales"], out[f"{stem}.biases"] = q, s, b
            done.append(stem)
        else:
            out[name] = val
    mx.save_safetensors(str(out_shard), out, metadata={"format": "mlx"})
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="models/Step-3.7-Flash",
                    help="source of the authoritative tokenizer")
    a = ap.parse_args()
    src, out = Path(a.src).resolve(), Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    index = json.loads((src / "model.safetensors.index.json").read_text())
    wmap = index["weight_map"]
    dirty = {wmap[f"{p}.weight"] for p in HEAD_PATHS}
    print(f"[head8] shards to rewrite: {sorted(dirty)}")

    # Everything except the dirty shards: hardlink (free) or copy (small files).
    for item in src.iterdir():
        if item.name in dirty:
            continue
        dest = out / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        elif item.suffix == ".safetensors":
            os.link(item, dest)               # hardlink: no extra disk
        else:
            shutil.copy(item, dest)

    # BF16 originals -- the student's 4-bit head cannot be upgraded in place.
    base = Path(a.base)
    bidx = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    bf16 = {}
    for p in HEAD_PATHS:
        key = p.replace("language_model.", "") + ".weight"
        if key not in bidx:
            raise KeyError(f"{key} not in base index; cannot source BF16 head")
        bf16[p] = mx.load(str(base / bidx[key]))[key]
        print(f"[head8] BF16 {key}: {bf16[p].shape} {bf16[p].dtype}")

    done = []
    for shard in sorted(dirty):
        print(f"[head8] quantizing {shard} from BF16 ...", flush=True)
        done += requantize_shard(src / shard, out / shard, list(HEAD_PATHS), bf16)
    if sorted(done) != sorted(HEAD_PATHS):
        raise RuntimeError(f"expected to requantize {HEAD_PATHS}, did {done}")

    # Per-path quant entries so the loader reads these two back at 8-bit.
    cfg = json.loads((out / "config.json").read_text())
    entry = {"group_size": GROUP_SIZE, "bits": NEW_BITS, "mode": "affine"}
    for key in ("quantization", "quantization_config"):
        if key in cfg:
            for p in HEAD_PATHS:
                cfg[key][p] = dict(entry)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    # Take the tokenizer from the authoritative base so this build cannot
    # inherit a stale one -- but base's OWN tokenizer_config.json (StepFun's
    # upstream file, unmodified) declares tokenizer_class="LlamaTokenizerFast",
    # which makes AutoTokenizer (mlx_lm/LM Studio's actual loader) discard the
    # real pretokenizer for Llama's SentencePiece/Metaspace one. See
    # docs/TOKENIZER-INVESTIGATION.md's correction section. fix_tokenizer_class
    # strips that declaration and verifies live.
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        if (base / name).exists():
            shutil.copy(base / name, out)
    sys.path.insert(0, str(Path(__file__).parent))
    from fix_tokenizer_class import fix_tokenizer_class
    fix_tokenizer_class(out)
    stages = len(json.loads((out / "tokenizer.json").read_text())
                 ["pre_tokenizer"]["pretokenizers"])
    print(f"[head8] tokenizer from base ({stages} pretokenizer stages)")

    du = sum(f.stat().st_size for f in out.rglob("*")
             if f.is_file() and f.stat().st_nlink == 1) / 1e9
    print(f"[OK] {out}  (new bytes on disk: {du:.2f} GB)")


if __name__ == "__main__":
    main()
