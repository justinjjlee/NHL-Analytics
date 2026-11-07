"""env_utils.py — Unified Environment & Deployment Configuration Utilities.

Re-exports the canonical environment detection, Hugging Face token discovery,
model resolution, and LM Studio health check utilities from `dev.zamboni.env_utils`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dev.zamboni.env_utils import (  # noqa: E402,F401
    check_lm_studio_health,
    discover_hf_token,
    discover_model_name,
    get_project_root,
    is_local_streamlit,
    resolve_lm_studio_endpoint,
)

__all__ = [
    "is_local_streamlit",
    "discover_hf_token",
    "discover_model_name",
    "check_lm_studio_health",
    "resolve_lm_studio_endpoint",
    "get_project_root",
]
