#!/usr/bin/env bash
# =============================================================================
# NHL Analytics - Hugging Face Dataset Bootstrap Runner
# =============================================================================
# Usage:
#   ./deploy/huggingface/bootstrap_dataset.sh
#   ./deploy/huggingface/bootstrap_dataset.sh --skip-upload
#   ./deploy/huggingface/bootstrap_dataset.sh --repo "your-username/nhl-analytics"
# =============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "🏒 Initializing NHL Analytics Dataset Bootstrap..."

# Run using uv to automatically inject huggingface_hub without altering global env
uv run --with huggingface_hub python deploy/huggingface/bootstrap_dataset.py "$@"
