"""
Deploy Rinklytics Streamlit Dashboard to Hugging Face Spaces.
"""
import argparse
import os
import sys
import time
from pathlib import Path

try:
    from huggingface_hub import HfApi
except ImportError:
    print("Error: huggingface_hub is required. Run: uv pip install huggingface_hub")
    sys.exit(1)


def get_token():
    # 1. Environment variable
    token = os.environ.get("HF_TOKEN") or os.environ.get("rinklytics")
    if token:
        return token.strip().strip('"').strip("'")

    # 2. Check local .env file
    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("rinklytics=") or line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def deploy(repo_id="jjerlee/rinklytics", token=None, wait=True):
    token = token or get_token()
    if not token:
        print("Error: Hugging Face access token not found.")
        print("Please set HF_TOKEN environment variable or add it to .env.")
        sys.exit(1)

    api = HfApi(token=token)
    user_info = api.whoami()
    print(f"Authenticated as: {user_info.get('name')}")

    repo_root = Path(__file__).resolve().parent

    print(f"Deploying Rinklytics to Hugging Face Space: {repo_id}...")

    # Upload folder excluding non-dashboard assets and large raw play-by-play files
    api.upload_folder(
        folder_path=str(repo_root),
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=[
            ".git/**",
            ".venv/**",
            "venv/**",
            "__pycache__/**",
            "*.pyc",
            ".DS_Store",
            ".env",
            "credentials.json",
            "latest/play/**",
            "src/**",
            "databricks/**",
            ".databricks/**",
            ".devcontainer/**",
            ".vscode/**",
            "databricks.yml",
            "container/**",
            "dev/**/*.png",
            "dev/**/*.jpg",
            "dev/**/*.jpeg",
            "dev/**/*.gif",
            "docs/**",
            ".github/**",
            "tests/**",
            "scratch/**",
            "deploy_hf.py",
        ],
        commit_message="Deploy Rinklytics Streamlit dashboard and datasets",
    )
    print("Files uploaded successfully!")

    if wait:
        print("Waiting for Space build and startup...")
        for attempt in range(30):
            time.sleep(5)
            runtime = api.get_space_runtime(repo_id)
            print(f"[{attempt + 1}] Stage: {runtime.stage}")
            if runtime.stage == "RUNNING":
                print(f"\nSUCCESS! Rinklytics is live at: https://huggingface.co/spaces/{repo_id}")
                return
            if runtime.stage in ["BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR"]:
                error = getattr(runtime, "error_message", None) or runtime.raw.get("errorMessage")
                print(f"\nERROR: Space encountered {runtime.stage}: {error}")
                print("\nRecent Space logs:")
                try:
                    for line in api.fetch_space_logs(repo_id):
                        print(line, end="")
                except Exception:
                    pass
                sys.exit(1)
        print("Timeout waiting for RUNNING state. Check https://huggingface.co/spaces/" + repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy to Hugging Face Space")
    parser.add_argument("--repo", default="jjerlee/rinklytics", help="Target HF Space repo ID")
    parser.add_argument("--token", default=None, help="Hugging Face access token")
    parser.add_argument("--no-wait", action="store_true", help="Do not wait for build completion")
    args = parser.parse_args()

    deploy(repo_id=args.repo, token=args.token, wait=not args.no_wait)
