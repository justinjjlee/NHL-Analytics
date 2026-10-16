"""env_utils.py — Environment and Deployment Configuration Utilities.

Determines whether the application is running in a local development environment
or in a deployed Hugging Face Space container.
"""

from __future__ import annotations

import os
from pathlib import Path


def is_local_streamlit() -> bool:
    """Returns True if running in a local development environment, or False in Hugging Face Spaces / CI.
    
    Priority order:
    1. Explicit System Environment Variable (`local_streamlit` / `LOCAL_STREAMLIT`)
    2. Automatic Cloud Runtime Detection (`SPACE_ID`, `SPACE_REPO_NAME`, `SPACE_HOST`, `GITHUB_ACTIONS`) -> False
    3. Streamlit Secrets (`st.secrets["local_streamlit"]`)
    4. Repo `.env` file (`local_streamlit=...`)
    5. Defaults to True for local developer machines.
    """
    # 1. Explicit System Environment Variable override
    for key in ("local_streamlit", "LOCAL_STREAMLIT"):
        val = os.environ.get(key)
        if val is not None:
            return str(val).strip().lower() in ("true", "1", "yes")

    # 2. Hugging Face Spaces / GitHub Actions automatic cloud detection
    if (
        os.environ.get("SPACE_ID")
        or os.environ.get("SPACE_REPO_NAME")
        or os.environ.get("SPACE_HOST")
        or os.environ.get("GITHUB_ACTIONS") == "true"
    ):
        return False

    # 3. Check Streamlit Secrets if available
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for key in ("local_streamlit", "LOCAL_STREAMLIT"):
                if key in st.secrets:
                    val = str(st.secrets[key]).strip().lower()
                    return val in ("true", "1", "yes")
    except Exception:
        pass

    # 4. Check repo root .env file
    try:
        root = Path(__file__).resolve().parent.parent.parent.parent
        env_file = root / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k in ("local_streamlit", "LOCAL_STREAMLIT"):
                        return v.lower() in ("true", "1", "yes")
    except Exception:
        pass

    # 5. Default to True for local developer machines
    return True


def discover_hf_token(explicit_token: str | None = None) -> str | None:
    """Discovers Hugging Face token across explicit arg, env vars, Streamlit secrets, huggingface_hub, and .env file."""
    if explicit_token and str(explicit_token).strip():
        return str(explicit_token).strip().strip('"').strip("'")

    # 1. Environment variables
    for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "rinklytics"):
        val = os.environ.get(key)
        if val and val.strip():
            return val.strip().strip('"').strip("'")

    # 2. Streamlit Secrets (Hugging Face Spaces / Streamlit Cloud)
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "rinklytics"):
                if key in st.secrets and st.secrets[key]:
                    return str(st.secrets[key]).strip().strip('"').strip("'")
    except Exception:
        pass

    # 3. huggingface_hub cached token
    try:
        from huggingface_hub import get_token
        tok = get_token()
        if tok:
            return tok.strip()
    except Exception:
        pass

    # 4. Local repo .env file
    try:
        root = Path(__file__).resolve().parent.parent.parent.parent
        env_path = root / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("#") or not line or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "rinklytics"):
                    cleaned = v.strip().strip('"').strip("'")
                    if cleaned:
                        return cleaned
    except Exception:
        pass

    return None
