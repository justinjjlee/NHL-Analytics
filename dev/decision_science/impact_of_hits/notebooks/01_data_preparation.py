# Databricks notebook source
# MAGIC %md
# MAGIC # Impact of Hits: Data Preparation Workflow
# MAGIC Joins play-by-play constraints and pre-estimated `elo_betting_efficiency` matrices into unified tensor sets.

ENV = "LOCAL" # Toggle between "LOCAL" and "DATABRICKS"

# COMMAND ----------
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

import sys
import os

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

repo_root = os.path.abspath(os.path.join(parent_dir, '../../..'))
data_dir = os.path.join(parent_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

# COMMAND ----------
# MAGIC %md
# MAGIC ### 1. Data Ingestion & Scope

# COMMAND ----------
if ENV == "LOCAL":
    import pandas as pd
    from src.features.sessionizer import convert_time_in_period
    from src.features.strength_mappings import pandas_align_priors_to_focal_team
    from src.features.state_logic import pandas_assign_event_flags, pandas_identify_zone_entries, pandas_compute_possession_shift_index
    from src.features.local_elo import compute_local_elo
    
    raw_pbp = pd.read_csv(os.path.join(repo_root, "latest/play/2025_playbyplay.csv"))
    
    # Rename columns to match PySpark convention
    raw_pbp.columns = [c.replace('.', '_') for c in raw_pbp.columns]
    raw_pbp.rename(columns={"details_eventOwnerTeam": "details_eventOwnerTeamId"}, inplace=True)
    
    # Needs a home_team/homeTeam_abbrev mapping, which the PySpark raw.play_by_play has natively
    box_path = os.path.join(repo_root, "latest/box/2025_box_team.csv")
    prior_metrics = compute_local_elo(None, box_path, as_pandas=True)
    
    home_teams = prior_metrics[['gameid', 'home_team']].rename(columns={'home_team': 'homeTeam_abbrev', 'gameid': 'game_id'})
    raw_pbp = raw_pbp.rename(columns={'gameid': 'game_id'})
    
    raw_pbp = raw_pbp[raw_pbp["periodDescriptor_periodType"] == "REG"]
    if "situationCode" in raw_pbp.columns:
        raw_pbp = raw_pbp[raw_pbp["situationCode"] == 1551]
    raw_pbp = raw_pbp.merge(home_teams, on='game_id', how='left')
    
    # Reconstruct score if missing
    if "homeTeam_score" not in raw_pbp.columns:
        raw_pbp = raw_pbp.sort_values(by=["game_id", "eventId"])
        # We need to know who scored. typeDescKey == "goal"
        raw_pbp["is_home_goal"] = ((raw_pbp["typeDescKey"] == "goal") & (raw_pbp["details_eventOwnerTeamId"] == raw_pbp["homeTeam_abbrev"])).astype(int)
        raw_pbp["is_away_goal"] = ((raw_pbp["typeDescKey"] == "goal") & (raw_pbp["details_eventOwnerTeamId"] != raw_pbp["homeTeam_abbrev"])).astype(int)
        
        raw_pbp["homeTeam_score"] = raw_pbp.groupby("game_id")["is_home_goal"].cumsum().shift(1).fillna(0)
        raw_pbp["awayTeam_score"] = raw_pbp.groupby("game_id")["is_away_goal"].cumsum().shift(1).fillna(0)
else:
    import pyspark.sql.functions as F
    from pyspark.sql import SparkSession
    from src.features.sessionizer import spark_convert_time_in_period
    from src.features.strength_mappings import align_priors_to_focal_team
    from src.features.state_logic import assign_event_flags, identify_zone_entries, compute_possession_shift_index

    # Play-by-Play Pull (Filters for 5v5 REG 2024-2025 and 2025-2026 seasons)
    raw_pbp = spark.table("`nhl-databricks`.raw.play_by_play") \
        .filter(F.col("season").isin("20242025", "20252026")) \
        .filter(F.col("periodDescriptor_periodType") == "REG") \
        .filter(F.col("situationCode") == "1551")

    # Extract the ELO / Process Metrics Unity Catalog Matrix
    prior_metrics = spark.table("`nhl-databricks`.compute.elo_betting_efficiency")

# COMMAND ----------
# MAGIC %md
# MAGIC ### 2. Execute Extracted `src` Feature Logic

# COMMAND ----------
if ENV == "LOCAL":
    # a) Time translation
    df_events = convert_time_in_period(raw_pbp, period_col="periodDescriptor_number")
    
    # Assume perspective: Every game analyzed purely from the Home Team perspective
    df_events["focal_team_abbrev"] = df_events["homeTeam_abbrev"]
    
    # b) Structural metrics
    df_events = pandas_assign_event_flags(df_events, focal_team_col="focal_team_abbrev")
    df_events = pandas_identify_zone_entries(df_events, focal_team_col="focal_team_abbrev")
    df_events = pandas_compute_possession_shift_index(df_events, focal_team_col="focal_team_abbrev")
else:
    # a) Time translation
    df_events = spark_convert_time_in_period(raw_pbp)

    # Assume perspective: Every game analyzed purely from the Home Team perspective
    focal_team_strategy = "home_team" 
    df_events = df_events.withColumn("focal_team_abbrev", F.col("homeTeam_abbrev"))

    # b) Structural metrics
    df_events = assign_event_flags(df_events, focal_team_col="focal_team_abbrev")
    df_events = identify_zone_entries(df_events, focal_team_col="focal_team_abbrev")
    df_events = compute_possession_shift_index(df_events, focal_team_col="focal_team_abbrev")

# COMMAND ----------
# MAGIC %md
# MAGIC ### 3. Join Priors & Filter to Active Matrix Core

# COMMAND ----------
tensor_schema = [
    "game_id", "elapsed_sec", "focal_team_abbrev",
    "hit_for_O", "hit_for_D", "hit_for_N",
    "hit_against_O", "hit_against_D", "hit_against_N", 
    "shot_for", "shot_against",
    "takeaway", "giveaway", "block_for", "block_against", "faceoff_win",
    "rolling_zone_share_90s",
    "focal_delta_pythagorean", "focal_delta_CF_pct", "score_state"
]

if ENV == "LOCAL":
    df_master = pd.merge(
        df_events,
        prior_metrics,
        left_on=["game_id", "homeTeam_abbrev"],
        right_on=["gameid", "home_team"],
        how="left"
    )
    
    # Align to the focal team 
    df_master = pandas_align_priors_to_focal_team(df_master, focal_team_abbrev="focal_team_abbrev")
    
    # Add dummy is_home for local since we assumed home team perspective above
    df_master["is_home"] = 1.0
    tensor_schema_local = tensor_schema + ["is_home"]
    
    # Drop missing tensor cols if any
    for col in tensor_schema_local:
        if col not in df_master.columns:
            df_master[col] = 0.0
            
    df_master_clean = df_master[tensor_schema_local].sort_values(by=["game_id", "elapsed_sec"])
else:
    df_master = df_events.join(
        prior_metrics,
        on=(df_events["game_id"] == prior_metrics["gameid"]) & 
           (df_events["homeTeam_abbrev"] == prior_metrics["home_team"]),
        how="left"
    )

    # Align to the focal team 
    df_master = align_priors_to_focal_team(df_master, focal_team_abbrev=F.col("focal_team_abbrev"))

    df_master_clean = df_master.select(*tensor_schema).orderBy("game_id", "elapsed_sec")

# COMMAND ----------
# MAGIC %md
# MAGIC ### 4. Export

# COMMAND ----------
if ENV == "LOCAL":
    import os
    os.makedirs(data_dir, exist_ok=True)    
    out_path = os.path.join(data_dir, "df_master.parquet")
    df_master_clean.to_parquet(out_path, index=False)
    print(f"Local Data Prep Output saved successfully. Row Count: {len(df_master_clean)}")
else:
    # Save the event workflow payload to unity catalog to be pulled by Neural Model Notebook
    # spark.sql("CREATE DATABASE IF NOT EXISTS `nhl-databricks`.compute")
    df_master_clean.write.mode("overwrite").saveAsTable("`nhl-databricks`.compute.impact_of_hits_master")
    print(f"Data Prep Output saved successfully. Row Count: {df_master_clean.count()}")
