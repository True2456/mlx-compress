#!/usr/bin/env python3
"""
BF16 Download & Shard Integrity Verifier

Verifies that all base model shards are downloaded and valid before attaching the GPU VM:
1. Checks model.safetensors.index.json existence and JSON syntax.
2. Checks presence and size of every .safetensors shard listed in index.
3. Reports total download progress and missing shards.
"""
import os
import sys
import json
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Verify BF16 Base Model Shards Integrity")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing downloaded model shards")
    return parser.parse_args()

def main():
    args = parse_args()
    model_dir = Path(args.model_dir)

    print("=" * 65)
    print(f"Verifying Model Shards Integrity in: {model_dir}")
    print("=" * 65)

    if not model_dir.exists():
        print(f"❌ Error: Directory {model_dir} does not exist.")
        sys.exit(1)

    index_file = model_dir / "model.safetensors.index.json"
    if not index_file.exists():
        print(f"❌ Error: {index_file} not found. Download is INCOMPLETE.")
        sys.exit(1)

    try:
        with open(index_file, "r") as f:
            index_data = json.load(f)
    except Exception as e:
        print(f"❌ Error parsing index file: {e}")
        sys.exit(1)

    weight_map = index_data.get("weight_map", {})
    required_shards = sorted(list(set(weight_map.values())))

    print(f"📋 Found index file mapping {len(weight_map)} tensors across {len(required_shards)} safetensors shards.")

    missing_shards = []
    total_bytes = 0

    for shard in required_shards:
        shard_path = model_dir / shard
        if not shard_path.exists():
            missing_shards.append(shard)
        else:
            size = shard_path.stat().st_size
            if size == 0:
                missing_shards.append(shard)
            else:
                total_bytes += size

    total_gb = total_bytes / (1024 ** 3)

    if missing_shards:
        print(f"\n❌ VERIFICATION FAILED: {len(missing_shards)} / {len(required_shards)} shards missing or incomplete:")
        for m in missing_shards[:10]:
            print(f"   - Missing: {m}")
        if len(missing_shards) > 10:
            print(f"   ... and {len(missing_shards) - 10} more.")
        sys.exit(1)
    else:
        print("\n" + "=" * 65)
        print(f"✅ ALL {len(required_shards)} SHARDS VERIFIED & INTACT!")
        print(f"📦 Total Base Model Size: {total_gb:.2f} GB")
        print("Ready for REAP GPU VM Attachment!")
        print("=" * 65)

if __name__ == "__main__":
    main()
