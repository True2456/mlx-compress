#!/usr/bin/env python3
"""
Hugging Face Upload Handler for REAP Winner & Artifacts

Uploads:
1. Winning Pruned Weights -> True2456/Step-3.7-Flash-REAP-pXX
2. Saliency & Plans -> True2456/Step-3.7-Flash-REAP-artifacts
"""
import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo

def parse_args():
    parser = argparse.ArgumentParser(description="Upload REAP Winner & Artifacts to Hugging Face")
    parser.add_argument("--winner-dir", type=str, required=True, help="Path to the winning pruned model directory")
    parser.add_argument("--repo-name", type=str, required=True, help="Target Hugging Face repo name (e.g. True2456/Step-3.7-Flash-REAP-p15)")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/reap_run", help="Path to saliency and plan JSONs")
    parser.add_argument("--private", action="store_true", default=True, help="Set repo to private (default: True)")
    return parser.parse_args()

def upload_model_variant(api: HfApi, folder_path: Path, repo_name: str, token: str, private: bool = True):
    """Uploads a single model variant folder to Hugging Face Hub."""
    if not folder_path.exists():
        print(f"⚠️ Warning: Directory {folder_path} does not exist. Skipping.")
        return

    print(f"🚀 Creating & uploading repo: {repo_name}...")
    try:
        create_repo(repo_id=repo_name, private=private, exist_ok=True, token=token)
        api.upload_folder(
            folder_path=str(folder_path),
            repo_id=repo_name,
            repo_type="model",
            token=token
        )
        print(f"🎉 Successfully uploaded variant to https://huggingface.co/{repo_name}")
    except Exception as e:
        print(f"❌ Error uploading {repo_name}: {e}")

def main():
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("❌ Error: HF_TOKEN environment variable not found.")
        sys.exit(1)

    api = HfApi(token=token)

    print("=" * 65)
    print(f"Uploading REAP Model Weights & Artifacts to Hugging Face")
    print("=" * 65)

    winner_path = Path(args.winner_dir)
    upload_model_variant(api, winner_path, args.repo_name, token, args.private)

    # Upload Artifacts (Saliency & Plans)
    artifacts_repo = f"{args.repo_name.split('/')[0]}/Step-3.7-Flash-REAP-artifacts"
    try:
        create_repo(repo_id=artifacts_repo, private=args.private, exist_ok=True, token=token)
        print(f"🚀 Uploading saliency profiles and plan JSONs to {artifacts_repo}...")
        api.upload_folder(
            folder_path=args.artifacts_dir,
            repo_id=artifacts_repo,
            repo_type="model",
            token=token
        )
        print(f"🎉 Artifacts successfully uploaded to https://huggingface.co/{artifacts_repo}")
    except Exception as e:
        print(f"⚠️ Warning uploading artifacts: {e}")

    print("\n" + "=" * 65)
    print("✅ Hugging Face Upload Complete! Ready to shut down GPU VM.")
    print("=" * 65)

if __name__ == "__main__":
    main()
