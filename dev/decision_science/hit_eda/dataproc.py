"""
Hit EDA Data Processing Pipeline.

Extracts 1D sequence micro-telemetry, reaction latencies, zone posture metrics,
and zone clearance indicators following physical hits.

Supports dual-execution:
  - Local mode (Pandas vectorized processing -> CSVs)
  - Databricks mode (PySpark processing -> Unity Catalog Volumes & Delta Tables)

Usage:
  # Run locally for latest season (2026):
  uv run python dev/decision_science/hit_eda/dataproc.py --env local --seasons 2026

  # Run locally for all available seasons:
  uv run python dev/decision_science/hit_eda/dataproc.py --env local --seasons all

  # Run in Databricks or trigger Databricks Unity Catalog save:
  python dev/decision_science/hit_eda/dataproc.py --env databricks --save-delta
"""

import os
import sys
import glob
import time
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Internal import from same directory
try:
    from config import HitEDAConfig, detect_environment
except ImportError:
    from dev.decision_science.hit_eda.config import HitEDAConfig, detect_environment


CATEGORY_MAP = {
    "shot-on-goal": "Shot Attempt",
    "missed-shot": "Shot Attempt",
    "blocked-shot": "Shot Attempt",
    "goal": "Goal",
    "takeaway": "Possession Change",
    "giveaway": "Possession Change",
    "hit": "Physical Battle",
    "stoppage": "Stoppage",
    "penalty": "Penalty",
    "faceoff": "Stoppage",
    "period-end": "Stoppage",
    "game-end": "Stoppage",
    "delayed-penalty": "Penalty"
}


def parse_time_to_seconds(time_str):
    """Convert 'MM:SS' string to integer seconds."""
    if pd.isna(time_str) or not isinstance(time_str, str) or ":" not in time_str:
        return 0
    parts = time_str.split(":")
    try:
        return int(parts[0]) * 60 + int(parts[1])
    except (ValueError, IndexError):
        return 0


# ==============================================================================
# Pandas Engine (Local Development & Edge Runtime)
# ==============================================================================

def process_file_pandas(file_path: str, limit: int = None) -> pd.DataFrame:
    """Read a single play-by-play CSV and extract post-hit sequence dynamics."""
    print(f"  --> Loading: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path, nrows=limit, low_memory=False)

    required_cols = ["gameid", "timeInPeriod", "typeDescKey"]
    for col in required_cols:
        if col not in df.columns:
            print(f"      [Warning] Missing {col} in {file_path}. Skipping file.")
            return pd.DataFrame()

    # Normalize period and timestamps
    if "periodDescriptor.number" in df.columns:
        df["period"] = df["periodDescriptor.number"].fillna(1).astype(int)
    elif "period" not in df.columns:
        df["period"] = 1

    df["period_sec"] = df["timeInPeriod"].apply(parse_time_to_seconds)
    df["elapsed_sec"] = (df["period"] - 1) * 1200 + df["period_sec"]

    # Normalize column names if nested with dots
    owner_col = "details.eventOwnerTeam" if "details.eventOwnerTeam" in df.columns else "eventOwnerTeam"
    zone_col = "details.zoneCode" if "details.zoneCode" in df.columns else "zoneCode"
    event_id_col = "eventId" if "eventId" in df.columns else "id"

    if owner_col not in df.columns:
        df[owner_col] = None
    if zone_col not in df.columns:
        df[zone_col] = "N"

    # Sort chronologically by contest, period, and clock
    df = df.sort_values(["gameid", "period", "elapsed_sec", event_id_col]).reset_index(drop=True)

    # Shift -1 within (gameid, period) window
    grp = df.groupby(["gameid", "period"])
    df["next_type"] = grp["typeDescKey"].shift(-1)
    df["next_owner"] = grp[owner_col].shift(-1)
    df["next_zone"] = grp[zone_col].shift(-1)
    df["next_elapsed_sec"] = grp["elapsed_sec"].shift(-1)

    # Filter strictly for hit events that have a subsequent event within the same period
    hits = df[df["typeDescKey"] == "hit"].copy()
    hits = hits.dropna(subset=["next_type"]).copy()

    # Latency calculation
    hits["delta_sec"] = (hits["next_elapsed_sec"] - hits["elapsed_sec"]).clip(lower=0)

    # Standardize zone
    hits["zoneCode"] = hits[zone_col].fillna("N")
    hits["next_zone"] = hits["next_zone"].fillna("N")

    # Ownership attribution (Relative to hitting team)
    hitting_team = hits[owner_col]
    hits["next_owner_rel"] = np.where(
        hits["next_owner"].isna(),
        "Stoppage / Neutral",
        np.where(hits["next_owner"] == hitting_team, "Hitting Team", "Opponent")
    )

    # Map category
    hits["next_category"] = hits["next_type"].map(CATEGORY_MAP).fillna("Other")

    # Critical outcome indicators
    hits["is_zone_exit"] = np.where(
        (hits["zoneCode"] == "D") & (hits["next_zone"].isin(["N", "O"])), 1, 0
    )
    hits["is_sog_for"] = np.where(
        (hits["next_type"] == "shot-on-goal") & (hits["next_owner_rel"] == "Hitting Team"), 1, 0
    )
    hits["is_sog_against"] = np.where(
        (hits["next_type"] == "shot-on-goal") & (hits["next_owner_rel"] == "Opponent"), 1, 0
    )
    hits["is_goal_for"] = np.where(
        (hits["next_type"] == "goal") & (hits["next_owner_rel"] == "Hitting Team"), 1, 0
    )
    hits["is_goal_against"] = np.where(
        (hits["next_type"] == "goal") & (hits["next_owner_rel"] == "Opponent"), 1, 0
    )

    cols_to_keep = [
        "gameid", "gameDate" if "gameDate" in hits.columns else "period",
        "period", "elapsed_sec", event_id_col,
        owner_col, "zoneCode",
        "next_type", "next_category", "next_owner", "next_owner_rel",
        "next_zone", "delta_sec",
        "is_zone_exit", "is_sog_for", "is_sog_against", "is_goal_for", "is_goal_against"
    ]
    cols_to_keep = [c for c in cols_to_keep if c in hits.columns]
    return hits[cols_to_keep]


def run_pandas_pipeline(config: HitEDAConfig, seasons_filter: list = None, limit: int = None):
    """Execute end-to-end Pandas data pipeline across play-by-play files."""
    print("=" * 70)
    print("Starting Hit EDA Pipeline (Pandas Engine)")
    print(f"Play Directory:   {config.play_dir}")
    print(f"Output Directory: {config.output_dir}")
    print("=" * 70)

    config.ensure_output_dirs()

    # Find candidate files
    all_files = sorted(glob.glob(os.path.join(config.play_dir, "*_playbyplay.csv")))
    # Exclude shift or player helper files
    play_files = [f for f in all_files if not f.endswith("_player.csv") and not f.endswith("_shift.csv")]

    if seasons_filter and "all" not in seasons_filter:
        play_files = [
            f for f in play_files
            if any(str(season) in os.path.basename(f) for season in seasons_filter)
        ]

    if not play_files:
        raise FileNotFoundError(f"No play-by-play files found in {config.play_dir} matching {seasons_filter}")

    print(f"Identified {len(play_files)} season files for processing.")

    processed_dfs = []
    for f in play_files:
        processed_hits = process_file_pandas(f, limit=limit)
        if not processed_hits.empty:
            processed_dfs.append(processed_hits)

    if not processed_dfs:
        print("No hit events found across files.")
        return

    master_hits = pd.concat(processed_dfs, ignore_index=True)
    total_hits = len(master_hits)
    print(f"\nSuccessfully extracted and standardized {total_hits:,} total hit events.")

    # --------------------------------------------------------------------------
    # Aggregation 1: Zone Macro KPI Table
    # --------------------------------------------------------------------------
    zone_kpi = master_hits.groupby("zoneCode").agg(
        total_hits=("gameid", "count"),
        recovery_pct=("next_owner_rel", lambda s: (s == "Hitting Team").mean() * 100),
        opponent_possession_pct=("next_owner_rel", lambda s: (s == "Opponent").mean() * 100),
        avg_delta_sec=("delta_sec", "mean"),
        median_delta_sec=("delta_sec", "median"),
        sog_for=("is_sog_for", "sum"),
        sog_against=("is_sog_against", "sum"),
        goals_for=("is_goal_for", "sum"),
        goals_against=("is_goal_against", "sum"),
        zone_exits=("is_zone_exit", "sum")
    ).reset_index()

    zone_kpi["share_of_all_hits_pct"] = (zone_kpi["total_hits"] / total_hits) * 100
    zone_kpi["sog_for_per_100"] = (zone_kpi["sog_for"] / zone_kpi["total_hits"]) * 100
    zone_kpi["sog_against_per_100"] = (zone_kpi["sog_against"] / zone_kpi["total_hits"]) * 100
    zone_kpi["net_sog_per_100"] = zone_kpi["sog_for_per_100"] - zone_kpi["sog_against_per_100"]
    zone_kpi["goals_for_per_100"] = (zone_kpi["goals_for"] / zone_kpi["total_hits"]) * 100
    zone_kpi["goals_against_per_100"] = (zone_kpi["goals_against"] / zone_kpi["total_hits"]) * 100
    zone_kpi["net_goals_per_100"] = zone_kpi["goals_for_per_100"] - zone_kpi["goals_against_per_100"]
    zone_kpi["zone_exit_pct"] = np.where(
        zone_kpi["zoneCode"] == "D",
        (zone_kpi["zone_exits"] / zone_kpi["total_hits"]) * 100,
        np.nan
    )

    # --------------------------------------------------------------------------
    # Aggregation 2: Next Event Category Breakdown Table
    # --------------------------------------------------------------------------
    next_events = master_hits.groupby(
        ["zoneCode", "next_category", "next_owner_rel"]
    ).agg(
        event_count=("gameid", "count"),
        avg_delta_sec=("delta_sec", "mean"),
        median_delta_sec=("delta_sec", "median")
    ).reset_index()

    next_events["pct_of_zone"] = next_events.groupby("zoneCode")["event_count"].transform(
        lambda x: (x / x.sum()) * 100
    )

    # --------------------------------------------------------------------------
    # Save Artifacts
    # --------------------------------------------------------------------------
    out_zone_kpi_path = os.path.join(config.output_dir, "hit_zone_summary.csv")
    out_next_events_path = os.path.join(config.output_dir, "hit_next_events.csv")
    out_sequences_sample_path = os.path.join(config.output_dir, "hit_sequences_sample.csv")

    zone_kpi.to_csv(out_zone_kpi_path, index=False)
    next_events.to_csv(out_next_events_path, index=False)
    # Save a head sample of detailed sequences for inspection
    master_hits.head(1000).to_csv(out_sequences_sample_path, index=False)

    print("\nGenerated Output Datasets:")
    print(f"  [1] Zone Summary KPI:     {out_zone_kpi_path}")
    print(f"  [2] Next Events Matrix:   {out_next_events_path}")
    print(f"  [3] Sequences Sample:     {out_sequences_sample_path}")

    # Display KPI table summary in console
    print("\n" + "=" * 70)
    print("Hit EDA Zone Summary Results:")
    print("=" * 70)
    print(zone_kpi[[
        "zoneCode", "total_hits", "share_of_all_hits_pct", "recovery_pct",
        "avg_delta_sec", "sog_for_per_100", "sog_against_per_100",
        "net_sog_per_100", "zone_exit_pct"
    ]].to_string(index=False))
    print("=" * 70)


# ==============================================================================
# PySpark Engine (Databricks Unity Catalog Runtime)
# ==============================================================================

def run_spark_pipeline(config: HitEDAConfig, seasons_filter: list = None, save_delta: bool = False):
    """Execute distributed PySpark pipeline in Databricks and write to Unity Catalog."""
    print("=" * 70)
    print("Starting Hit EDA Pipeline (PySpark Engine on Databricks)")
    print(f"Target Unity Catalog: {config.catalog}.{config.schema}")
    print(f"Volume Output:        {config.output_dir}")
    print("=" * 70)

    try:
        from pyspark.sql import SparkSession
        import pyspark.sql.functions as F
        from pyspark.sql.window import Window
    except ImportError:
        print("Error: PySpark is not available in the current Python environment.")
        print("Please install pyspark or run with --engine pandas.")
        sys.exit(1)

    spark = SparkSession.getActiveSession()
    if not spark:
        spark = SparkSession.builder.appName("Hit_EDA_Pipeline").getOrCreate()

    # Load from Unity Catalog Volume
    input_path = f"{config.play_dir}/*_playbyplay.csv"
    print(f"Reading play-by-play files from: {input_path}")
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(input_path)

    # Filter out helper files
    df = df.filter(~F.col("_metadata.file_path").rlike(r"(_player|_shift)\.csv$"))

    # Convert time string to seconds
    time_split = F.split(F.col("timeInPeriod"), ":")
    period_sec = (F.col(time_split[0]).cast("int") * 60) + F.col(time_split[1]).cast("int")
    period_num = F.coalesce(F.col("periodDescriptor.number"), F.col("period"), F.lit(1))
    elapsed_sec = (period_num - 1) * 1200 + period_sec

    df = df.withColumn("period", period_num)
    df = df.withColumn("elapsed_sec", elapsed_sec)

    owner_col = "details.eventOwnerTeam" if "details.eventOwnerTeam" in df.columns else "eventOwnerTeam"
    zone_col = "details.zoneCode" if "details.zoneCode" in df.columns else "zoneCode"
    event_id_col = "eventId" if "eventId" in df.columns else "id"

    # Lead window partitioned by contest and period
    w = Window.partitionBy("gameid", "period").orderBy("elapsed_sec", event_id_col)

    df_leads = df.withColumn("next_type", F.lead("typeDescKey", 1).over(w)) \
                 .withColumn("next_owner", F.lead(owner_col, 1).over(w)) \
                 .withColumn("next_zone", F.lead(zone_col, 1).over(w)) \
                 .withColumn("next_elapsed_sec", F.lead("elapsed_sec", 1).over(w))

    # Filter strictly for hit events with non-null next event
    hits_df = df_leads.filter((F.col("typeDescKey") == "hit") & F.col("next_type").isNotNull())
    hits_df = hits_df.withColumn("delta_sec", F.greatest(F.lit(0), F.col("next_elapsed_sec") - F.col("elapsed_sec")))

    # Relative owner mapping
    hits_df = hits_df.withColumn(
        "next_owner_rel",
        F.when(F.col("next_owner").isNull(), "Stoppage / Neutral")
         .when(F.col("next_owner") == F.col(owner_col), "Hitting Team")
         .otherwise("Opponent")
    )

    # Zone exit flag
    hits_df = hits_df.withColumn(
        "is_zone_exit",
        F.when((F.col(zone_col) == "D") & F.col("next_zone").isin("N", "O"), 1).otherwise(0)
    )

    # Write output to Unity Catalog Volume
    csv_out_path = f"{config.output_dir}/hit_sequences_spark.csv"
    print(f"Writing parsed sequences to Volume: {csv_out_path}")
    hits_df.write.mode("overwrite").option("header", "true").csv(csv_out_path)

    if save_delta:
        print(f"Saving managed Delta Table: {config.table_sequences}")
        spark.sql(f"CREATE DATABASE IF NOT EXISTS `{config.catalog}`.{config.schema}")
        hits_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(config.table_sequences)
        print(f"✅ Delta Table `{config.table_sequences}` created successfully.")

    print("Spark Hit EDA pipeline completed successfully.")


# ==============================================================================
# Main Trigger & CLI Parsing
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="NHL Hit Exploratory Data Analysis (EDA) Pipeline")
    parser.add_argument(
        "--env",
        choices=["auto", "local", "databricks"],
        default="auto",
        help="Execution environment trigger (default: auto-detect)"
    )
    parser.add_argument(
        "--seasons",
        default="2026",
        help="Comma-separated seasons to process (e.g., '2026', '2024,2025,2026', or 'all')"
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "pandas", "spark"],
        default="auto",
        help="Computation engine (default: auto selects based on environment)"
    )
    parser.add_argument(
        "--save-delta",
        action="store_true",
        help="Save output as Unity Catalog Delta Tables (Databricks mode only)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of rows read per file (useful for fast testing)"
    )

    args = parser.parse_args()

    # Determine environment
    resolved_env = detect_environment(None if args.env == "auto" else args.env)
    config = HitEDAConfig(env=resolved_env)

    # Determine engine
    engine = args.engine
    if engine == "auto":
        engine = "spark" if config.is_databricks() else "pandas"

    seasons = [s.strip() for s in args.seasons.split(",")]

    print(f"Hit EDA Trigger Initialized:")
    print(f"  Detected Environment: {config.env.upper()}")
    print(f"  Selected Engine:      {engine.upper()}")
    print(f"  Target Seasons:       {seasons}")

    start_time = time.time()
    if engine == "spark":
        run_spark_pipeline(config, seasons_filter=seasons, save_delta=args.save_delta)
    else:
        run_pandas_pipeline(config, seasons_filter=seasons, limit=args.limit)

    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
