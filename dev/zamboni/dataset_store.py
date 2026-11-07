"""
dataset_store.py — Lakehouse Persistence, Trajectory Logging & Hugging Face Dataset Sync for Agent Zamboni.

Stores curated Golden Q&A pairs (`golden_qa.parquet`) and interaction trajectory logs
(`interaction_logs.parquet`) locally and synchronizes them to `hf://datasets/jjerlee/nhl-analytics/insights/zamboni/`.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from .config import HF_DATASET_BASE, HF_DATASET_REPO
from .env_utils import discover_hf_token
from .models import ZamboniResponse


class HFZamboniStore:
    """
    Manages local Parquet caching and Hugging Face Dataset synchronization
    for Agent Zamboni's Golden Q&A cache and Trajectory Analysis logs.
    """

    def __init__(self, cache_dir: Optional[str] = None, token: Optional[str] = None):
        default_cache = os.path.expanduser("~/.cache/rinklytics/insights/zamboni")
        self.cache_dir = Path(cache_dir or default_cache)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logs_path = self.cache_dir / "interaction_logs.parquet"
        self.golden_path = self.cache_dir / "golden_qa.parquet"
        self.token = discover_hf_token(token)

    def record_interaction(self, response: ZamboniResponse) -> None:
        """Appends an interaction + Trajectory Analysis record to the local Parquet log."""
        traj = response.trajectory
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "query": response.query,
            "allowed": response.allowed,
            "intent": response.intent,
            "persona": response.persona,
            "target_domains": json.dumps(response.target_domains),
            "pipeline_ids": json.dumps(traj.pipeline_ids),
            "chunk_ids": json.dumps([c.chunk_id for c in response.retrieved_chunks]),
            "skill_names": json.dumps([s.skill_name for s in response.computed_results]),
            "projected_tokens_est": int(traj.projected_tokens_est),
            "reflection_attempts": int(traj.reflection_attempts),
            "critic_passed": bool(traj.critic_passed),
            "critic_feedback": str(traj.critic_feedback or ""),
            "answer_markdown": response.answer_markdown,
            "generator_model": response.generator_model,
            "error_code": response.error_code or "",
            "latency_ms": float(response.latency_ms),
        }
        new_df = pd.DataFrame([row])
        try:
            if self.logs_path.exists():
                old_df = pd.read_parquet(self.logs_path)
                combined = pd.concat([old_df, new_df], ignore_index=True).tail(2000)
            else:
                combined = new_df
            combined.to_parquet(self.logs_path, index=False)
        except Exception:
            pass

    def load_logs(self) -> pd.DataFrame:
        if self.logs_path.exists():
            try:
                return pd.read_parquet(self.logs_path)
            except Exception:
                pass
        return pd.DataFrame()

    def load_golden_qa(self) -> pd.DataFrame:
        if self.golden_path.exists():
            try:
                return pd.read_parquet(self.golden_path)
            except Exception:
                pass
        try:
            return pd.read_parquet(f"{HF_DATASET_BASE}/insights/zamboni/golden_qa.parquet")
        except Exception:
            return pd.DataFrame()

    def lookup_golden_qa(self, query: str) -> Optional[Dict[str, Any]]:
        """Checks if an exact or normalized question exists in the verified Golden Q&A table."""
        df = self.load_golden_qa()
        if df.empty or "query" not in df.columns:
            return None
        q_norm = " ".join(str(query).strip().lower().split())
        for _, r in df.iterrows():
            cand = " ".join(str(r.get("query", "")).strip().lower().split())
            if cand == q_norm and r.get("golden_answer"):
                return r.to_dict()
        return None

    def upsert_golden_qa(self, records: List[Dict[str, Any]]) -> int:
        """Upserts Label Studio verified Q&A records into `golden_qa.parquet`."""
        if not records:
            return 0
        new_df = pd.DataFrame(records)
        old_df = self.load_golden_qa()
        if not old_df.empty:
            combined = pd.concat([old_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["query"], keep="last")
        else:
            combined = new_df.drop_duplicates(subset=["query"], keep="last")
        combined.to_parquet(self.golden_path, index=False)
        return len(combined)

    def sync_to_huggingface(self, commit_message: str = "Sync Agent Zamboni trajectory logs and golden Q&A") -> bool:
        """Uploads local `golden_qa.parquet` and `interaction_logs.parquet` to `jjerlee/nhl-analytics`."""
        if not self.token:
            return False
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=self.token)
            for local_p, remote_name in [
                (self.golden_path, "insights/zamboni/golden_qa.parquet"),
                (self.logs_path, "insights/zamboni/interaction_logs.parquet"),
            ]:
                if local_p.exists():
                    api.upload_file(
                        path_or_fileobj=str(local_p),
                        path_in_repo=remote_name,
                        repo_id=HF_DATASET_REPO,
                        repo_type="dataset",
                        commit_message=commit_message,
                    )
            return True
        except Exception:
            return False
