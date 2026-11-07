"""
Configuration and Environment Abstraction for Hit EDA.

Supports dual-execution runtime:
  1. Local Environment: Reads from ./latest/play and writes to ./dev/decision_science/hit_eda/data
  2. Databricks Runtime: Reads from Unity Catalog Volumes (/Volumes/nhl-databricks/data/play)
     and writes to Unity Catalog Volumes (/Volumes/nhl-databricks/data/hit_eda) or Delta Tables.
"""

import os
import argparse
from pathlib import Path


def get_repo_root() -> Path:
    """Return the absolute path to the repository root directory."""
    # This file lives in dev/decision_science/hit_eda/config.py -> repo root is parents[3]
    return Path(__file__).resolve().parents[3]


def detect_environment(override_env: str = None) -> str:
    """
    Detect whether the script is running in Databricks or a local development environment.
    
    Trigger precedence:
      1. Explicit override passed via argument (e.g., --env databricks or --env local)
      2. Environment variable NHL_ENV
      3. Databricks runtime environment variable DATABRICKS_RUNTIME_VERSION or DATABRICKS_HOST
    """
    if override_env and override_env.lower() in ("local", "databricks"):
        return override_env.lower()

    env_var = os.environ.get("NHL_ENV", "").lower()
    if env_var in ("local", "databricks"):
        return env_var

    if "DATABRICKS_RUNTIME_VERSION" in os.environ or "DATABRICKS_HOST" in os.environ:
        return "databricks"

    return "local"


class HitEDAConfig:
    """Encapsulates file paths, Unity Catalog targets, and engine settings."""

    def __init__(self, env: str = None):
        self.env = detect_environment(env)
        self.repo_root = get_repo_root()

        # Databricks Unity Catalog Settings
        self.catalog = "nhl-databricks"
        self.schema = "data"
        self.unity_catalog_volume_base = f"/Volumes/{self.catalog}/{self.schema}"

        # Delta Table Names (Unity Catalog 3-level namespace)
        self.table_sequences = f"`{self.catalog}`.{self.schema}.hit_eda_sequences"
        self.table_zone_kpi = f"`{self.catalog}`.{self.schema}.hit_eda_zone_kpi"
        self.table_next_events = f"`{self.catalog}`.{self.schema}.hit_eda_next_events"

        if self.is_databricks():
            # Databricks Volume Paths
            self.play_dir = f"{self.unity_catalog_volume_base}/play"
            self.output_dir = f"{self.unity_catalog_volume_base}/hit_eda"
        else:
            # Local Paths relative to repository root
            self.play_dir = str(self.repo_root / "latest" / "play")
            self.output_dir = str(self.repo_root / "dev" / "decision_science" / "hit_eda" / "data")

    def is_databricks(self) -> bool:
        return self.env == "databricks"

    def ensure_output_dirs(self):
        """Create output directories if running locally or in Volume paths."""
        if not self.is_databricks() or os.path.exists(os.path.dirname(self.output_dir)):
            os.makedirs(self.output_dir, exist_ok=True)

    def __repr__(self) -> str:
        return (
            f"HitEDAConfig(\n"
            f"  environment='{self.env}',\n"
            f"  play_dir='{self.play_dir}',\n"
            f"  output_dir='{self.output_dir}',\n"
            f"  unity_catalog_tables=[\n"
            f"    '{self.table_sequences}',\n"
            f"    '{self.table_zone_kpi}',\n"
            f"    '{self.table_next_events}'\n"
            f"  ]\n"
            f")"
        )
