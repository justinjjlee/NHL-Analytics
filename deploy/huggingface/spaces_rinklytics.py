"""
spaces_rinklytics.py — Deploy Rinklytics Streamlit Dashboard to Hugging Face Spaces.

Synchronizes application code, layouts, and views to the Hugging Face Space:
    https://huggingface.co/spaces/jjerlee/rinklytics
"""
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import HfApi
except ImportError:
    print("Error: huggingface_hub is required. Run: uv pip install huggingface_hub")
    sys.exit(1)


HF_SPACE_FRONTMATTER = """---
title: Rinklytics - NHL Analytics Dashboard
emoji: 🏒
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 6.29.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
---
"""


def get_repo_root() -> Path:
    """Return the absolute path to the repository root directory."""
    # This script lives in deploy/huggingface/spaces_rinklytics.py -> repo root is parent.parent.parent
    return Path(__file__).resolve().parent.parent.parent


def get_token():
    # 1. Environment variable
    token = os.environ.get("HF_TOKEN") or os.environ.get("rinklytics")
    if token:
        return token.strip().strip('"').strip("'")

    # 2. Check local .env file in repo root
    env_file = get_repo_root() / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("rinklytics=") or line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def prune_space_lfs(repo_id: str = "jjerlee/rinklytics", token: Optional[str] = None) -> int:
    """Permanently purges obsolete data/CSV/parquet LFS files from Hugging Face Space history using rewrite_history=True.
    
    This ensures that historical Git LFS pointers do not consume repository storage quota (1 GB limit).
    """
    token = token or get_token()
    if not token:
        print("Error: Hugging Face access token required to prune LFS files.")
        return 0
    api = HfApi(token=token)
    try:
        lfs_files = list(api.list_lfs_files(repo_id, repo_type="space"))
        to_delete = [
            f for f in lfs_files
            if f.filename.startswith(("latest/", "app/data/hit_eda/", "dev/", "data/"))
            or f.filename.endswith((".csv", ".parquet", ".zip", ".tar.gz"))
        ]
        if not to_delete:
            print("✓ No obsolete data LFS files found in Space history.")
            return 0
        total_mb = sum(f.size for f in to_delete) / (1024 * 1024)
        print(f"🧹 Pruning {len(to_delete)} obsolete LFS data files from Space history ({total_mb:.2f} MB)...")
        api.permanently_delete_lfs_files(
            repo_id=repo_id,
            lfs_files=to_delete,
            repo_type="space",
            rewrite_history=True,
        )
        print("✅ Space LFS storage permanently pruned and Git history rewritten.")
        return len(to_delete)
    except Exception as e:
        print(f"⚠️ Note during Space LFS prune: {e}")
        return 0


def deploy(repo_id="jjerlee/rinklytics", token=None, wait=True, auto_prune_lfs=True):
    token = token or get_token()
    if not token:
        print("Error: Hugging Face access token not found.")
        print("Please set HF_TOKEN environment variable or add it to .env.")
        sys.exit(1)

    api = HfApi(token=token)
    user_info = api.whoami()
    print(f"Authenticated as: {user_info.get('name')}")

    repo_root = get_repo_root()

    print(f"Deploying Rinklytics to Hugging Face Space: {repo_id}...")

    # Optional automatic safeguard: prune obsolete LFS history if storage exceeds 200 MB
    if auto_prune_lfs:
        try:
            space_info = api.space_info(repo_id)
            if space_info.used_storage and space_info.used_storage > 200 * 1024 * 1024:
                print(f"⚠️ Space storage is currently {space_info.used_storage / (1024*1024):.1f} MB (> 200 MB threshold).")
                prune_space_lfs(repo_id=repo_id, token=token)
        except Exception:
            pass

    # Upload pure source code only (data is sourced from Hugging Face dataset jjerlee/nhl-analytics)
    # Exclude all raw datasets, local caches, and heavy research files to keep Space size well under 1GB
    ignore_patterns = [
        "README.md",
        ".git/**",
        ".venv/**",
        "venv/**",
        "__pycache__/**",
        "*.pyc",
        ".DS_Store",
        "**/.DS_Store",
        ".env",
        "credentials.json",
        "latest/**",
        "databricks/**",
        ".databricks/**",
        ".vscode/**",
        "databricks.yml",
        "dev/decision_science/**",
        "dev/player/**",
        "dev/player_college/**",
        "dev/player_draft/**",
        "dev/player_team_success/**",
        "dev/team_success/**",
        "dev/team_success_benchmarkPredictor/**",
        "dev/team_success_measurements_vol1/**",
        "dev/team_success_RegularStanely_vol1/**",
        "dev/team_success_RegularStanely_vol2/**",
        "dev/homeice_advantage/**",
        "dev/**/*.png",
        "dev/**/*.jpg",
        "dev/**/*.jpeg",
        "dev/**/*.gif",
        "dev/**/*.csv",
        "dev/**/*.parquet",
        "app/data/hit_eda/**",
        ".github/**",
        "tests/**",
        "scratch/**",
        "deploy/**",
        "doc/**",
        "deploy_hf.py",
        "*.parquet",
        "*.lock",
        ".cache/**",
        "cache/**",
        "reports/**",
        "*.log",
    ]

    delete_patterns = [
        "latest/**",
        "dev/decision_science/**",
        "dev/player/**",
        "dev/player_college/**",
        "dev/player_draft/**",
        "dev/player_team_success/**",
        "dev/team_success/**",
        "dev/team_success_benchmarkPredictor/**",
        "dev/team_success_measurements_vol1/**",
        "dev/team_success_RegularStanely_vol1/**",
        "dev/team_success_RegularStanely_vol2/**",
        "dev/homeice_advantage/**",
        "app/data/hit_eda/**",
        "streamlit_test.py",
        "main.py",
        "*.parquet",
        "*.lock",
        "*.log",
    ]

    api.upload_folder(
        folder_path=str(repo_root),
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=ignore_patterns,
        delete_patterns=delete_patterns,
        commit_message="Deploy Rinklytics Streamlit dashboard (source code only)",
    )

    # Upload Space README.md with required YAML frontmatter metadata
    readme_path = repo_root / "README.md"
    if readme_path.exists():
        readme_content = readme_path.read_text(encoding="utf-8")
        if not readme_content.startswith("---"):
            readme_to_upload = f"{HF_SPACE_FRONTMATTER}\n{readme_content}"
        else:
            readme_to_upload = readme_content
        api.upload_file(
            path_or_fileobj=readme_to_upload.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="space",
            commit_message="Update Space README with configuration metadata",
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
    parser.add_argument("--prune-lfs", action="store_true", help="Permanently prune obsolete data LFS files and rewrite history")
    parser.add_argument("--prune-only", action="store_true", help="Only run LFS pruning without deploying")
    args = parser.parse_args()

    if args.prune_only:
        prune_space_lfs(repo_id=args.repo, token=args.token)
        sys.exit(0)

    if args.prune_lfs:
        prune_space_lfs(repo_id=args.repo, token=args.token)

    deploy(repo_id=args.repo, token=args.token, wait=not args.no_wait)
