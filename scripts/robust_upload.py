#!/usr/bin/env python3
import os
import sys
import time
import httpx
from pathlib import Path
from huggingface_hub import HfApi, create_repo, get_token, set_client_factory

# Set timeouts and performance optimizations via environment variables before other imports or clients are initialized
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def custom_client_factory():
    # Set a very generous timeout for connection, read, write, and pool timeouts to prevent drops
    return httpx.Client(
        timeout=httpx.Timeout(600.0, connect=60.0, read=600.0, write=600.0, pool=600.0),
        follow_redirects=True
    )

def main():
    # Configure custom HTTP client factory globally
    set_client_factory(custom_client_factory)

    repo_id = "True2456/DeepSeek-V4-Flash-0731-AWQ"
    local_path = "/Users/true/.lmstudio/models/True2456/DeepSeek-V4-Flash-0731-AWQ"
    repo_type = "model"

    # Authenticate
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        print("❌ Error: not authenticated. Please set HF_TOKEN env var or run `hf auth login` first.")
        sys.exit(1)

    api = HfApi(token=token)

    print("=" * 70)
    print("Starting Robust Hugging Face Upload")
    print(f"Target Repository: https://huggingface.co/{repo_id}")
    print(f"Source Folder:     {local_path}")
    print(f"Repository Type:   {repo_type}")
    print("=" * 70)

    # Ensure repository exists
    try:
        create_repo(repo_id=repo_id, repo_type=repo_type, private=True, exist_ok=True, token=token)
        print("✓ Repository checked/created successfully.")
    except Exception as e:
        print(f"⚠️ Warning during create_repo: {e}")

    # Start upload with retry logic
    max_retries = 10
    retry_delay = 15

    for attempt in range(1, max_retries + 1):
        print(f"\n🚀 Upload attempt {attempt}/{max_retries}...")
        try:
            # We use upload_folder directly, ignoring the local cache folder.
            api.upload_folder(
                folder_path=local_path,
                repo_id=repo_id,
                repo_type=repo_type,
                token=token,
                ignore_patterns=[".cache/**", "**/.DS_Store"],
            )
            print("\n🎉 Upload completed successfully!")
            print(f"Verify files at: https://huggingface.co/{repo_id}/tree/main")
            print("=" * 70)
            sys.exit(0)
        except Exception as e:
            print(f"⚠️ Upload attempt {attempt} failed with error: {e}")
            if attempt < max_retries:
                sleep_time = retry_delay * attempt
                print(f"Retrying in {sleep_time} seconds... (Resume will skip already uploaded files)")
                time.sleep(sleep_time)
            else:
                print("❌ Max retries reached. Upload failed.")
                sys.exit(1)

if __name__ == "__main__":
    main()
