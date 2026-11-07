"""
env_utils.py — Environment, Runtime Detection, and LM Studio Discovery Utilities for Agent Zamboni.

Determines whether Agent Zamboni is running in a local development environment
or in a deployed Hugging Face Space container, discovers Hugging Face tokens,
and checks LM Studio local server reachability (`http://127.0.0.1:1234/v1`).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, List
import requests

from .config import DEFAULT_LM_STUDIO_URL, DEFAULT_ZEROGPU_MODEL


def get_project_root() -> Path:
    """Resolve absolute path to the NHL-Analytics repository root."""
    return Path(__file__).resolve().parent.parent.parent


def is_local_streamlit() -> bool:
    """Returns True if running in a local development environment, or False in Hugging Face Spaces / CI.

    Priority order:
    1. Explicit System Environment Variable (`local_streamlit` / `LOCAL_STREAMLIT`)
    2. Automatic Cloud Runtime Detection (`SPACE_ID`, `SPACE_REPO_NAME`, `SPACE_HOST`, `GITHUB_ACTIONS`) -> False
    3. Streamlit Secrets (`st.secrets["local_streamlit"]`)
    4. Repo `.env` file (`local_streamlit=...`)
    5. Defaults to True for local developer machines.
    """
    for key in ("local_streamlit", "LOCAL_STREAMLIT"):
        val = os.environ.get(key)
        if val is not None:
            return str(val).strip().lower() in ("true", "1", "yes")

    if (
        os.environ.get("SPACE_ID")
        or os.environ.get("SPACE_REPO_NAME")
        or os.environ.get("SPACE_HOST")
        or os.environ.get("GITHUB_ACTIONS") == "true"
    ):
        return False

    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for key in ("local_streamlit", "LOCAL_STREAMLIT"):
                if key in st.secrets:
                    val = str(st.secrets[key]).strip().lower()
                    return val in ("true", "1", "yes")
    except Exception:
        pass

    try:
        root = get_project_root()
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

    return True


def discover_hf_token(explicit_token: Optional[str] = None) -> Optional[str]:
    """Discovers Hugging Face token across explicit arg, env vars, Streamlit secrets, huggingface_hub, and .env file."""
    if explicit_token and str(explicit_token).strip():
        return str(explicit_token).strip().strip('"').strip("'")

    for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "rinklytics"):
        val = os.environ.get(key)
        if val and val.strip():
            return val.strip().strip('"').strip("'")

    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "rinklytics"):
                if key in st.secrets and st.secrets[key]:
                    return str(st.secrets[key]).strip().strip('"').strip("'")
    except Exception:
        pass

    try:
        from huggingface_hub import get_token
        tok = get_token()
        if tok:
            return tok.strip()
    except Exception:
        pass

    try:
        root = get_project_root()
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


def discover_model_name(default_model: str = DEFAULT_ZEROGPU_MODEL) -> str:
    """Discovers the configured model name across environment variables, Streamlit secrets, and .env file."""
    keys = ("llm_model_name", "LLM_MODEL_NAME", "ZEROGPU_MODEL")
    for k in keys:
        val = os.environ.get(k)
        if val and str(val).strip():
            return str(val).strip().strip('"').strip("'")

    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for k in keys:
                if k in st.secrets and st.secrets[k]:
                    return str(st.secrets[k]).strip().strip('"').strip("'")
    except Exception:
        pass

    try:
        root = get_project_root()
        env_path = root / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("#") or not line or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() in keys:
                    cleaned = v.strip().strip('"').strip("'")
                    if cleaned:
                        return cleaned
    except Exception:
        pass

    return default_model


def check_lm_studio_health(base_url: str = DEFAULT_LM_STUDIO_URL, timeout: float = 2.5) -> Tuple[bool, Optional[str], List[str]]:
    """
    Probes the local LM Studio OpenAI-compatible server (`GET http://127.0.0.1:1234/v1/models`).
    Also supports Docker host bridge (`http://host.docker.internal:1234/v1`) when running inside
    the local Docker container at port 1901.

    Returns:
        (is_reachable: bool, active_model_id: Optional[str], available_models: List[str])
    """
    candidate_urls = [base_url.rstrip("/")]
    if "127.0.0.1" in base_url or "localhost" in base_url:
        docker_url = base_url.replace("127.0.0.1", "host.docker.internal").replace("localhost", "host.docker.internal").rstrip("/")
        if docker_url not in candidate_urls:
            candidate_urls.append(docker_url)

    preferred_keywords = ("nemotron", "phi-4", "gemma", "qwen", "llama", "mistral")

    for url in candidate_urls:
        try:
            resp = requests.get(f"{url}/models", timeout=timeout)
            if resp.status_code == 200:
                payload = resp.json()
                data = payload.get("data", [])
                model_ids = [
                    m.get("id") for m in data
                    if isinstance(m, dict) and m.get("id") and "embed" not in str(m.get("id")).lower()
                ]
                if not model_ids and data:
                    model_ids = [m.get("id") for m in data if isinstance(m, dict) and m.get("id")]

                active = None
                for kw in preferred_keywords:
                    for mid in model_ids:
                        if kw in mid.lower():
                            active = mid
                            break
                    if active:
                        break
                if not active and model_ids:
                    active = model_ids[0]

                return True, active or "local-model", model_ids
        except Exception:
            continue

    return False, None, []


def resolve_lm_studio_endpoint(base_url: str = DEFAULT_LM_STUDIO_URL, timeout: float = 2.0) -> Tuple[str, bool, Optional[str]]:
    """
    Returns (resolved_base_url, is_alive, active_model_id) for either localhost or host.docker.internal.
    """
    candidate_urls = [base_url.rstrip("/")]
    if "127.0.0.1" in base_url or "localhost" in base_url:
        docker_url = base_url.replace("127.0.0.1", "host.docker.internal").replace("localhost", "host.docker.internal").rstrip("/")
        if docker_url not in candidate_urls:
            candidate_urls.append(docker_url)

    for url in candidate_urls:
        try:
            resp = requests.get(f"{url}/models", timeout=timeout)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                model_ids = [
                    m.get("id") for m in data
                    if isinstance(m, dict) and m.get("id") and "embed" not in str(m.get("id")).lower()
                ]
                active = model_ids[0] if model_ids else "local-model"
                return url, True, active
        except Exception:
            continue

    return base_url.rstrip("/"), False, None
