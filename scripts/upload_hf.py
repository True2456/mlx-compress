#!/usr/bin/env python3
"""
Hugging Face Upload Handler for REAP Winner & Artifacts

Uploads:
1. Winning Pruned Weights -> True2456/Step-3.7-Flash-REAP-pXX
2. Saliency & Plans -> True2456/Step-3.7-Flash-REAP-artifacts
"""
import os
import sys
import time
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo, get_token

def parse_args():
    parser = argparse.ArgumentParser(description="Upload REAP Winner & Artifacts to Hugging Face")
    parser.add_argument("--winner-dir", type=str, required=True, help="Path to the winning pruned model directory")
    parser.add_argument("--repo-name", type=str, required=True, help="Target Hugging Face repo name (e.g. True2456/Step-3.7-Flash-REAP-p15)")
    parser.add_argument("--artifacts-dir", type=str, default=None,
                         help="Optional: path to saliency/plan JSONs to upload as a companion repo. "
                              "Skipped if not given.")
    parser.add_argument("--public", action="store_true",
                         help="Make the repo public. Default is private.")
    return parser.parse_args()

def upload_model_variant(api: HfApi, folder_path: Path, repo_name: str, token: str,
                          private: bool = True, max_retries: int = 5) -> bool:
    """Uploads a single model variant folder to Hugging Face Hub. Returns
    True on confirmed success, False otherwise -- caller must check this,
    the exception is swallowed here so one bad file doesn't kill retries."""
    if not folder_path.exists():
        print(f"⚠️ Warning: Directory {folder_path} does not exist. Skipping.")
        return False

    print(f"🚀 Creating & uploading repo: {repo_name}...")
    create_repo(repo_id=repo_name, private=private, exist_ok=True, token=token)

    for attempt in range(1, max_retries + 1):
        try:
            api.upload_folder(
                folder_path=str(folder_path),
                repo_id=repo_name,
                repo_type="model",
                token=token,
            )
            print(f"🎉 Successfully uploaded variant to https://huggingface.co/{repo_name}")
            return True
        except Exception as e:
            wait = min(60, 5 * attempt)
            print(f"⚠️ Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                print(f"   Retrying in {wait}s (already-uploaded content is content-addressed "
                      f"and won't be re-sent)...")
                time.sleep(wait)
            else:
                print(f"❌ Giving up after {max_retries} attempts uploading {repo_name}.")
                return False

def main():
    args = parse_args()
    # HF_TOKEN env var if set, else fall back to `hf auth login`'s cached token.
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        print("❌ Error: not authenticated. Run `hf auth login` or set HF_TOKEN.")
        sys.exit(1)

    api = HfApi(token=token)
    private = not args.public

    print("=" * 65)
    print(f"Uploading REAP Model Weights to Hugging Face (private={private})")
    print("=" * 65)

    winner_path = Path(args.winner_dir)
    ok = upload_model_variant(api, winner_path, args.repo_name, token, private)

    artifacts_ok = True
    if args.artifacts_dir:
        artifacts_repo = f"{args.repo_name.split('/')[0]}/Step-3.7-Flash-REAP-artifacts"
        artifacts_ok = upload_model_variant(api, Path(args.artifacts_dir), artifacts_repo, token, private)

    print("\n" + "=" * 65)
    if ok and artifacts_ok:
        print(f"✅ Upload confirmed complete: https://huggingface.co/{args.repo_name}")
        print("=" * 65)
        sys.exit(0)
    else:
        print("❌ Upload did NOT complete successfully -- see errors above.")
        print("=" * 65)
        sys.exit(1)

if __name__ == "__main__":
    main()
