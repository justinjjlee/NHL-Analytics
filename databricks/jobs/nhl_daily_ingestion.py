# Databricks notebook source
# MAGIC %md
# MAGIC # NHL Analytics Daily Data Ingestion Workflow
# MAGIC This notebook orchestrates the scheduling and ingestion of NHL records, replacing the old GitHub Actions process.
# MAGIC Because the underlying `src/data` module scripts natively check for the Databricks runtime environment, they will automatically read and write outputs to Unity Catalog volumes instead of local directories.

# COMMAND ----------
import subprocess
import os
import sys

# COMMAND ----------
# MAGIC %md
# MAGIC ### Execute Data Collection Pipeline

# COMMAND ----------
# Determine the directory containing the pulling scripts
# Assuming this notebook sits in databricks/jobs/
workspace_root = os.path.abspath(os.path.join(os.getcwd(), "../../"))
apinhle_dir = os.path.join(workspace_root, "src/data/apinhle")

if not os.path.exists(apinhle_dir):
    raise FileNotFoundError(f"Cannot find script directory: {apinhle_dir}. Ensure Databricks Repos is synced properly.")

scripts_to_run = [
    "data_pull_box.py",
    "data_pull_box_odds.py",
    "data_proc_gameSummary.py",
    "data_pull_plays.py",
    "data_pull_lines.py",
    "data_pull_player.py"
]

successes = 0

for script in scripts_to_run:
    print(f"==================================================")
    print(f"Starting execution of: {script}...")
    print(f"==================================================")
    
    # We run these as separate subprocesses to encapsulate their logic and handle their internal relative imports cleanly
    result = subprocess.run(["python", script], cwd=apinhle_dir, capture_output=True, text=True)
    
    print(result.stdout)
    
    if result.returncode != 0:
        print(f"🚨 Error executing {script}:\n{result.stderr}")
        raise Exception(f"Failed to execute {script}")
    else:
        print(f"✅ Successfully completed {script}\n")
        successes += 1

print(f"Workflow Complete! Successfully executed {successes}/{len(scripts_to_run)} jobs.")
