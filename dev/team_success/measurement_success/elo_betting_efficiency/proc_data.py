# Databricks notebook source
# MAGIC %md
# MAGIC # ELO & Betting Benchmark: Data Integration Workspace
# MAGIC This notebook aggregates raw sources from Unity Catalog and constructs an analytical panel matrix 
# MAGIC matching the theoretical framework outlined in section 3 of our project documentation.
# MAGIC 
# MAGIC The processing logic is abstracted into `src/data_prep.py` functions to keep this notebook clean and reusable.

# COMMAND ----------
# MAGIC %md
# MAGIC ### 1. Setup & Imports

# COMMAND ----------
# Utilizing local functions instead of pip packages

# COMMAND ----------
import pandas as pd
import numpy as np
import pyspark.sql.functions as F

# Import our modularized data prep workflow
import sys
import os

# Ensure the subdirectories are reachable for imports
sys.path.append(os.path.abspath(os.getcwd()))
from src.data_prep import load_odds, load_game_outcomes, build_team_panel

# COMMAND ----------
# MAGIC %md
# MAGIC ### 2. Extract Base Datasets
# MAGIC Generating Spark Dataframes extracting explicitly filtered data (2024–2025 seasons) and generating cumulative variables avoiding look-ahead-bias natively.

# COMMAND ----------
# Extract Odds (Filtered to 2024-2025, no Canada lines)
df_odds = load_odds(spark)

# Extract Game Outcomes (Win/Loss matrices)
df_box = load_game_outcomes(spark)

# Extract and Compute Process Metrics (Rolling Fenwick, Corsi, Pythagorean Expectation)
df_panel = build_team_panel(spark)

# COMMAND ----------
# MAGIC %md
# MAGIC ### 3. Combine to Analytical Matrix
# MAGIC Here we merge the home and away representations of the team panel back onto the game core to form the `g`-indexed predictive pairs.

# COMMAND ----------
# Create distinct aliases to join home and away rollups onto the master game log
df_home_panel = df_panel.select(
    F.col("gameid"),
    F.col("n_g").alias("n_g_home"),
    F.col("rolling_CF_pct").alias("home_CF_pct"),
    F.col("rolling_FF_pct").alias("home_FF_pct"),
    F.col("rolling_pythagorean").alias("home_pythagorean")
)

df_away_panel = df_panel.select(
    F.col("gameid"),
    F.col("n_g").alias("n_g_away"),
    F.col("rolling_CF_pct").alias("away_CF_pct"),
    F.col("rolling_FF_pct").alias("away_FF_pct"),
    F.col("rolling_pythagorean").alias("away_pythagorean")
)

# Merge all three core data domains
# Note: we drop duplicates to avoid Cartesian multiplication if multiple odds lines exist per gameid
df_analytical = df_box.join(df_odds, on=["gameid", "season", "home_team", "away_team"], how="inner").dropDuplicates(["gameid", "odds_description"]) \
    .join(df_home_panel, on="gameid", how="left") \
    .join(df_away_panel, on="gameid", how="left")

# Restrict to complete line observations
df_analytical = df_analytical.dropna(subset=["odds_decimal_home", "odds_decimal_away"])

# Calculate delta metrics (Delta_CF%, Delta_FF%) representing the difference in process expected value heading into the game
df_analytical = df_analytical.withColumn("delta_CF_pct", F.col("home_CF_pct") - F.col("away_CF_pct")) \
    .withColumn("delta_FF_pct", F.col("home_FF_pct") - F.col("away_FF_pct")) \
    .withColumn("delta_pythagorean", F.col("home_pythagorean") - F.col("away_pythagorean"))

# COMMAND ----------
# MAGIC %md
# MAGIC ### 4. Compute Shin Probabilities
# MAGIC Since Shin processing is iterative, we will compute this natively via pandas using our `shin` library against the decimal-converted moneyline odds. The results represent true $\hat{p}_g^{\text{Shin}}$ market expectations correcting for the favorites-longshot overround variance.

# COMMAND ----------
# Push to pandas for UDF row-level evaluation given the optimization underlying the Shin package
df_pandas = df_analytical.toPandas()

# Pull native Shin solver
from src.shinprobabilities import implied_probabilities

# Isolated Shin calculator loop
def apply_shin_correction(row):
    if pd.isna(row['odds_decimal_home']) or pd.isna(row['odds_decimal_away']):
        return pd.Series({'prob_home_shin': np.nan, 'prob_away_shin': np.nan, 'shin_vig': np.nan})
    try:
        # Native repository wrapper
        probs, vig = implied_probabilities(
            odds=[row['odds_decimal_home'], row['odds_decimal_away']],
            odds_kind="decimal",
            remove_vig="shin"
        )
        return pd.Series({
            'prob_home_shin': probs[0],
            'prob_away_shin': probs[1],
            'shin_vig': vig
        })
    except:
        return pd.Series({'prob_home_shin': np.nan, 'prob_away_shin': np.nan, 'shin_vig': np.nan})

shin_results = df_pandas.apply(apply_shin_correction, axis=1)
df_pandas = pd.concat([df_pandas, shin_results], axis=1)

# COMMAND ----------
# MAGIC %md
# MAGIC ### 5. Save Analysis Dataset
# MAGIC The table is fully built and holds indices, win flags, lag metric evaluations prior to puck drop, and Shin probabilities. Write this back down to UC or keep in notebook cache for direct modeling passouts!

# COMMAND ----------
# Final conversion back to Spark DataFrame
final_spark_df = spark.createDataFrame(df_pandas)

# Ensure the compute database exists
spark.sql("CREATE DATABASE IF NOT EXISTS `nhl-databricks`.compute")

# Save directly to Unity Catalog for downstream analysis
save_path = "`nhl-databricks`.compute.elo_betting_efficiency"
final_spark_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(save_path)

print(f"Data Prep Complete! Processed {final_spark_df.count()} eligible game records and saved to {save_path}.")
