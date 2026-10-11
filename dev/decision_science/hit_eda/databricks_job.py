# Databricks notebook source
# MAGIC %md
# MAGIC # NHL Analytics: Physical Hits Sequence & EDA Workflow
# MAGIC 
# MAGIC This notebook orchestrates the extraction of post-hit sequence telemetry, reaction latencies ($\Delta t$),
# MAGIC zone posture metrics, and defensive zone clearances across historical NHL play-by-play data.
# MAGIC 
# MAGIC ### Runtime Capabilities
# MAGIC - **Dual-Engine Trigger:** Automatically detects whether it is running on a Databricks cluster with PySpark or in local mode.
# MAGIC - **Unity Catalog Integration:** Reads raw play-by-play datasets from `/Volumes/nhl-databricks/data/play/` and writes outputs to `/Volumes/nhl-databricks/data/hit_eda/` as well as Delta tables (`nhl-databricks.data.hit_eda_sequences`, `hit_eda_zone_kpi`).

# COMMAND ----------

import os
import sys

# Ensure repository root is in sys.path when running from Databricks Repos
workspace_root = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from dev.decision_science.hit_eda.config import HitEDAConfig, detect_environment
from dev.decision_science.hit_eda.dataproc import run_spark_pipeline, run_pandas_pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Parameter Triggers & Widgets

# COMMAND ----------

try:
    dbutils.widgets.dropdown("environment", "databricks", ["databricks", "local"], "1. Environment")
    dbutils.widgets.text("seasons", "all", "2. Target Seasons ('all' or '2024,2025,2026')")
    dbutils.widgets.dropdown("engine", "spark", ["spark", "pandas"], "3. Execution Engine")
    dbutils.widgets.dropdown("save_delta", "true", ["true", "false"], "4. Save Unity Catalog Delta Tables")
    
    target_env = dbutils.widgets.get("environment")
    seasons_str = dbutils.widgets.get("seasons")
    target_engine = dbutils.widgets.get("engine")
    save_delta_str = dbutils.widgets.get("save_delta")
    save_delta = save_delta_str.lower() == "true"
except NameError:
    # Running outside Databricks interactive notebook (e.g. testing or CLI)
    target_env = detect_environment()
    seasons_str = os.environ.get("NHL_SEASONS", "all")
    target_engine = "spark" if target_env == "databricks" else "pandas"
    save_delta = True

seasons = [s.strip() for s in seasons_str.split(",")]

print(f"Trigger Settings:")
print(f"  Environment:  {target_env}")
print(f"  Engine:       {target_engine}")
print(f"  Seasons:      {seasons}")
print(f"  Save Delta:   {save_delta}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Execute Hit EDA Sequence Extraction

# COMMAND ----------

config = HitEDAConfig(env=target_env)
print(f"Config initialized:\n{config}")

if target_engine == "spark":
    run_spark_pipeline(config, seasons_filter=seasons, save_delta=save_delta)
else:
    run_pandas_pipeline(config, seasons_filter=seasons)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Verification & Unity Catalog Table Preview

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Preview the generated Delta Table if saved to Unity Catalog
# MAGIC SELECT * FROM `nhl-databricks`.data.hit_eda_sequences LIMIT 10;
