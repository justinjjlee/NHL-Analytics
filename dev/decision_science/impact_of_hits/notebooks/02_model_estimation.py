# Databricks notebook source
# MAGIC %md
# MAGIC # Impact of Hits: Estimation Pipeline
# MAGIC Imports the curated event tensor dataset and launches PyTorch training directly on Databricks clusters (or locally).

ENV = "LOCAL"

# COMMAND ----------
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

import sys
import os
import torch
import numpy as np
import pandas as pd

try:
    # Local Python execution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
except NameError:
    # Databricks execution (no __file__ defined)
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

data_dir = os.path.join(parent_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

from src.model.momentum_ssm import MomentumSSM
from src.model.estimation import fit_ssm, bootstrap_B
from src.model.inference import compute_half_life, impulse_response
from src.viz.plotting import plot_B_coefficients, plot_impulse_response

# COMMAND ----------
# MAGIC %md
# MAGIC ### 1. Dataset Tensor Extraction
# MAGIC Since PyTorch relies on iterative batches, we group the master table by game ID into standard `list[dict[Tensor]]` components required by `src.model.estimation`.

# COMMAND ----------

if ENV == "LOCAL":
    df_master = pd.read_parquet(os.path.join(data_dir, "df_master.parquet"))
else:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    # Load processed Master Table
    df_master = spark.table("`nhl-databricks`.compute.impact_of_hits_master").toPandas()

event_cols = [
    "hit_for_O", "hit_for_D", "hit_for_N", "hit_against_O", "hit_against_D", "hit_against_N", 
    "shot_for", "shot_against",
    "takeaway", "giveaway", "block_for", "block_against", "faceoff_win"
]
baseline_cols = ["is_home", "focal_delta_pythagorean", "focal_delta_CF_pct", "score_state"]

games = []
for game_id, group in df_master.groupby("game_id"):
    
    U = torch.tensor(group[event_cols].values, dtype=torch.float32)
    Y = torch.tensor(group["rolling_zone_share_90s"].values, dtype=torch.float32)
    Z = torch.tensor(group[baseline_cols].values, dtype=torch.float32)
    
    # Calculate time deltas for the state decay
    dt = torch.tensor(np.diff(group["elapsed_sec"].values, prepend=0.0), dtype=torch.float32)
    
    games.append({
        "game_id": game_id,
        "U": U,
        "Y": Y,
        "Z": Z,
        "dt": dt
    })

print(f"Constructed PyTorch Dictionary sequences for {len(games)} distinct games.")

# COMMAND ----------
# MAGIC %md
# MAGIC ### 2. Model Estimation & Bootstrap Check
# MAGIC Execute the ML Kalman Loop

# COMMAND ----------
# Train Base Model to convergence (Adjust epochs in practice)
print("Initiating Base Model Fits...")
base_model = fit_ssm(games, n_epochs=1000, lr=1e-2, verbose=True)

# Generate Bootstrapped Confidence Intervals
print("\nInitiating Boostrap sequence...")
B_samples = bootstrap_B(games, n_boot=500, n_epochs=400)

# Extract B Stats
B_hat = base_model.B.detach().numpy()
B_lo = np.percentile(B_samples, 2.5, axis=0)
B_hi = np.percentile(B_samples, 97.5, axis=0)

# Print final coefficient array
for i, ev in enumerate(event_cols):
    print(f"{ev:15s}: {B_hat[i]:.4f}  (95% CI: [{B_lo[i]:.4f}, {B_hi[i]:.4f}])")

# COMMAND ----------
# MAGIC %md
# MAGIC ### 3. Visualize & Analyze Output

# COMMAND ----------
# Visualize Momentum Impacts Plot natively back to the Databricks UI window
fig_B = plot_B_coefficients(B_hat, B_lo, B_hi, event_cols)
# display(fig_B) 
# Save to disk instead of just showing
fig_B.savefig(os.path.join(data_dir, "viz3_coefficients.png"))

# Examine the Impulse flow curves
a0_hat = torch.sigmoid(base_model.log_a0).item()
half_life = compute_half_life(a0_hat)
print(f"Decay Parameter a0: {a0_hat:.5f}")
print(f"Event Half-Life: {half_life:.1f} seconds")

# Grab selected weights
hit_b = B_hat[event_cols.index("hit_for_O")]  # using O-zone hit as proxy for impulse visualization
takeaway_b = B_hat[event_cols.index("takeaway")]
shot_b = B_hat[event_cols.index("shot_for")]

fig_impulse = plot_impulse_response(
    a0_val=a0_hat, 
    b_vals={"Hit (O-Zone)": hit_b, "Takeaway": takeaway_b, "Shot On Goal": shot_b},
    max_sec=180
)
# Save to disk instead of just showing
fig_impulse.savefig(os.path.join(data_dir, "viz4_impulse_response.png"))

# Save the trained model for downstream insights Notebook
torch.save(base_model, os.path.join(data_dir, "base_model.pt"))
np.save(os.path.join(data_dir, "B_samples.npy"), B_samples)
print("Model saved successfully.")
