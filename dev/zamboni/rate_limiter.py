"""
rate_limiter.py — Global Cross-User Rate Limiter for Agent Zamboni.

Mandate:
- Locally (`is_local_streamlit() == True`): Unlimited throughput (no rate limitation).
- Hugging Face Production (`is_local_streamlit() == False`): Strictly limited to
  1 response per 1.0 second across ALL connected internet users globally.
"""

from __future__ import annotations

import os
import time
import threading
from typing import Tuple

try:
    import fcntl
except ImportError:
    fcntl = None  # Windows fallback

from .config import HF_GLOBAL_MIN_INTERVAL_SEC, GLOBAL_RATE_LOCK_FILE


class GlobalRateLimiter:
    """
    Thread-safe and multi-process-safe global rate limiter.
    Enforces a strict minimum interval (`1.0s` default) between pings across all users
    when deployed in Hugging Face Spaces.
    """

    def __init__(
        self,
        min_interval_sec: float = HF_GLOBAL_MIN_INTERVAL_SEC,
        lock_file_path: str = GLOBAL_RATE_LOCK_FILE,
    ):
        self.min_interval_sec = float(min_interval_sec)
        self.lock_file_path = lock_file_path
        self._thread_lock = threading.Lock()
        self._last_ping_ts: float = 0.0
        self._total_allowed: int = 0
        self._total_throttled: int = 0

    def try_acquire(self, is_local: bool = False) -> Tuple[bool, float]:
        """
        Attempt to acquire permission to process a user request.

        Args:
            is_local: True if running locally (bypasses rate limit),
                      False if running on Hugging Face Spaces (enforces 1 req/sec globally).

        Returns:
            (allowed: bool, retry_after_sec: float)
        """
        if is_local:
            with self._thread_lock:
                self._total_allowed += 1
            return True, 0.0

        with self._thread_lock:
            now = time.monotonic()

            # 1. Check in-process monotonic timestamp
            elapsed_mem = now - self._last_ping_ts
            if elapsed_mem < self.min_interval_sec:
                self._total_throttled += 1
                return False, round(self.min_interval_sec - elapsed_mem, 3)

            # 2. Check cross-process file lock (for multi-worker containers)
            wall_now = time.time()
            try:
                os.makedirs(os.path.dirname(self.lock_file_path) or "/tmp", exist_ok=True)
                with open(self.lock_file_path, "a+", encoding="utf-8") as f:
                    if fcntl is not None:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        f.seek(0)
                        raw = f.read().strip()
                        last_wall = float(raw) if raw else 0.0
                        elapsed_wall = wall_now - last_wall
                        if 0 <= elapsed_wall < self.min_interval_sec:
                            self._total_throttled += 1
                            return False, round(self.min_interval_sec - elapsed_wall, 3)

                        f.seek(0)
                        f.truncate()
                        f.write(f"{wall_now:.6f}")
                        f.flush()
                    finally:
                        if fcntl is not None:
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception:
                # If file lock fails due to permissions, in-memory lock still protects process
                pass

            self._last_ping_ts = now
            self._total_allowed += 1
            return True, 0.0

    def get_telemetry(self, is_local: bool = False) -> dict:
        """Returns diagnostic rate-limiter metrics for UI display."""
        return {
            "mode": "Local Unlimited" if is_local else f"HF Global Guardrail (1 req / {self.min_interval_sec:.1f}s)",
            "min_interval_sec": 0.0 if is_local else self.min_interval_sec,
            "total_allowed": self._total_allowed,
            "total_throttled": self._total_throttled,
        }


# Process-wide singleton instance
_GLOBAL_RATE_LIMITER = GlobalRateLimiter()


def get_global_rate_limiter() -> GlobalRateLimiter:
    """Returns the process-wide GlobalRateLimiter singleton (cached across all Streamlit sessions)."""
    try:
        import streamlit as st

        @st.cache_resource
        def _cached_limiter() -> GlobalRateLimiter:
            return _GLOBAL_RATE_LIMITER

        return _cached_limiter()
    except Exception:
        return _GLOBAL_RATE_LIMITER
