#!/usr/bin/env python3
"""
NHL Analytics - Hugging Face Dataset Initialization & Bootstrap Script

This script consolidates historical NHL data across all available seasons
into unified, high-performance Apache Parquet panel datasets and uploads them
to a public or private Hugging Face Dataset repository.

Usage:
    # 1. Compile locally and upload to Hugging Face
    uv run --with huggingface_hub python deploy/huggingface/bootstrap_dataset.py --repo "jjerlee/nhl-analytics"

    # 2. Dry run: Compile locally only (without upload)
    python deploy/huggingface/bootstrap_dataset.py --skip-upload

    # 3. Using Hugging Face CLI for the upload stage:
    python deploy/huggingface/bootstrap_dataset.py --use-cli --repo "jjerlee/nhl-analytics"
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# CONSTANTS & REPOSITORY ROOT RESOLUTION
# -----------------------------------------------------------------------------

def get_repo_root() -> Path:
    """Returns the absolute path to the NHL-Analytics repository root."""
    return Path(__file__).resolve().parent.parent.parent


DATASET_CARD_TEMPLATE = """---
annotations_creators:
- machine-generated
language:
- en
license:
- mit
multilinguality:
- monolingual
size_categories:
- 1M<n<10M
source_datasets:
- original
task_categories:
- tabular-classification
- tabular-regression
- time-series-forecasting
tags:
- sports
- nhl
- hockey
- panel-data
- box-scores
- play-by-play
- sports-analytics
pretty_name: NHL Analytics - Multi-Season Longitudinal Panel Data (2013 ~ Present, Daily Updated)
configs:
- config_name: games
  data_files: "data/games.parquet"
- config_name: team_box
  data_files: "data/team_box.parquet"
- config_name: player_box
  data_files: "data/player_box.parquet"
- config_name: odds
  data_files: "data/odds.parquet"
- config_name: hit_impact
  data_files: "data/hit_impact.parquet"
- config_name: playbyplay
  data_files: "data/playbyplay.parquet"
- config_name: playbyplay_player
  data_files: "data/playbyplay_player.parquet"
- config_name: shifts
  data_files: "data/shifts.parquet"
- config_name: measurement_team_season
  data_files: "measurement/box/team_season.parquet"
- config_name: measurement_game_stats
  data_files: "measurement/box/game_stats.parquet"
- config_name: measurement_player_season
  data_files: "measurement/player/player_season.parquet"
- config_name: measurement_player_game
  data_files: "measurement/player/player_game.parquet"
- config_name: measurement_hit_kpi
  data_files: "measurement/action/hits/team_season_hit_kpi.parquet"
- config_name: measurement_hit_spatial
  data_files: "measurement/action/hits/team_season_spatial_hits.parquet"
- config_name: measurement_hit_impact
  data_files: "measurement/action/hits/hit_impact.parquet"
- config_name: measurement_hit_goal_window
  data_files: "measurement/action/hits/hit_goal_window_30s.parquet"
- config_name: measurement_hit_next_events
  data_files: "measurement/action/hits/hit_next_events.parquet"
- config_name: measurement_hit_transitions
  data_files: "measurement/action/hits/hit_sequence_transitions.parquet"
- config_name: measurement_hit_zone_summary
  data_files: "measurement/action/hits/hit_zone_summary.parquet"
- config_name: measurement_hit_zone_kpi
  data_files: "measurement/action/hits/hit_zone_summary_kpi.parquet"
- config_name: measurement_hit_grid
  data_files: "measurement/action/hits/rink_grid_hit_sequences.parquet"
- config_name: measurement_hit_tabulated
  data_files: "measurement/action/hits/tabulated_hit_sequences.parquet"
---

# NHL Analytics: Multi-Season Longitudinal Panel & Measurement Dataset (2013 ~ Present | Daily Updated)

> **Source Repository & Ingestion Pipelines:**  
> 🏒 **GitHub:** [`justinjjlee/NHL-Analytics` (Rinklytics)](https://github.com/justinjjlee/NHL-Analytics)  
>
> *"NHL Analytics (also known as Rinklytics) is a comprehensive sports analytics platform for collecting, processing, and visualizing National Hockey League (NHL) data. Automated daily GitHub Actions workflows incrementally append and update incoming game records, player rosters, betting odds, and play-by-play coordinates directly from the official NHL Edge API."*

## Dataset Description
This dataset provides comprehensive, multi-season **panel data** and **advanced analytical measurements** for the National Hockey League (NHL), spanning over 17,000 games, 600,000 player-game appearances, and millions of play-by-play event coordinates from **2013 to the present day (actively maintained and daily updated)**.

The repository is structured into two primary tiers:
1. **`data/`**: Raw longitudinal panel datasets (games, box scores, rosters, shifts, play-by-play events, closing betting lines).
2. **`measurement/`**: Derived analytical metrics, team-season benchmarks, expected goals (xG / ixG / GSAx), and physical action telemetry generated by automated pipelines, all indexed with standardized `yr_season` columns.

Automated daily GitHub Actions workflows incrementally append and update incoming game records from the NHL Edge API.

---

## Dataset Structure

### Tier 1: Raw Longitudinal Panels (`data/`)

#### 1. `data/games.parquet` (Game Master Panel)
* **Unit of Observation:** One row per game (`game_id`).
* **Columns:** `game_id`, `season`, `game_date`, `time_start`, `home_team`, `away_team`, `home_team_id`, `away_team_id`, `home_score`, `away_score`, `period_ending`, `winning_team`, `xg_for`, `xg_against`, `xg_diff`, `xg_pct_for`, `hdcf_for`, `hdcf_against`, `gsax_for`, `gsax_against`.

#### 2. `data/team_box.parquet` (Team-Game Box Scores)
* **Unit of Observation:** One row per team per game (`game_id`, `team`).
* **Columns:** `game_id`, `season`, `game_date`, `team`, `is_home`, `goals`, `shots`, `hits`, `shots_blocked`, `goals_pp`, `pim`, `takeaways`, `giveaways`, `faceoff_win_pct`, `win`, `xg`, `xga`, `xg_lvl`, `xg_pct`, `hdcf`, `hdca`, `gsax`.

#### 3. `data/player_box.parquet` (Individual Player Performance)
* **Unit of Observation:** One row per player per game (`game_id`, `player_id`).
* **Columns:** `game_id`, `season`, `game_date`, `player_id`, `player_name`, `team`, `position`, `sweater_number`, `toi`, `goals`, `assists`, `points`, `plus_minus`, `pim`, `hits`, `sog`, `blocked_shots`, `faceoff_pct`, `saves`, `shots_against`, `save_pct`, `ixg`, `g_minus_ixg`, `hd_shots`, `xga`, `gsax`, `xsave_pct`, `dsave_pct`, `cum_ixg`, `cum_gsax`, `is_season_end`.

#### 4. `data/odds.parquet` (Betting Closing Lines)
* **Unit of Observation:** Closing market odds per game and country market (`game_id`, `country`, `odds_description`).
* **Columns:** `game_id`, `country` (`"US"` or `"CAN"`), `game_date`, `home_team`, `away_team`, `odds_description`, `home_odds`, `away_odds`, `betting_partner`, `start_time_utc`, `last_updated_utc`.

#### 5. `data/playbyplay.parquet` (Granular Event Microdata Panel)
* **Unit of Observation:** Granular play-by-play events (`game_id`, `event_id`).
* **Format:** Single Parquet file (~58 MB compressed for ~5.4 million events across 14 seasons).
* **Columns:** `game_id`, `season`, `game_type`, `game_date`, `event_id`, `time_in_period`, `time_remaining`, `period`, `period_type`, `event_type`, `reason`, `owner_team`, `x_coord`, `y_coord`, `zone_code`, `home_defending_side`, `shot_type`, `away_sog`, `home_sog`, `shooting_player_id`, `blocking_player_id`, `goalie_in_net_id`, `scoring_player_id`, `assist1_player_id`, `assist2_player_id`, `hitting_player_id`, `hittee_player_id`, `duration`, `committed_by_player_id`, `drawn_by_player_id`.

#### 6. `data/playbyplay_player.parquet` (Game Roster & Participating Players)
* **Unit of Observation:** One row per skater/goaltender present on the official game sheet (`game_id`, `player_id`).
* **Columns:** `game_id`, `season`, `player_id`, `team`, `sweater_number`, `position`, `first_name`, `last_name`, `headshot`.

#### 7. `data/shifts.parquet` (Shift-Level Tracking)
* **Unit of Observation:** Individual on-ice player shift (`shift_id`).
* **Columns:** `shift_id`, `game_id`, `player_id`, `season`, `team`, `player_name`, `period`, `shift_number`, `start_time`, `end_time`, `duration`, `type_code`, `detail_code`, `event_number`, `event_description`, `event_details`.

---

### Tier 2: Measured & Computed Statistics (`measurement/`)

All measurement datasets contain the standardized season identifier **`yr_season`** (e.g. 2013 to 2026).

#### A. Box & Team Measurements (`measurement/box/`)
* **`measurement/box/team_season.parquet`**: Multi-season team KPI aggregates, Fenwick/Corsi differential ratings, Expected Goals (`xg_for`, `xg_against`, `xg_lvl`, `xg_pct`, `hdcf_pct`, `gsax`, `kpi_xg`, `kpi_gsax`), Pythagorean win expectation, RPI, and win streaks.
* **`measurement/box/game_stats.parquet`**: Game-level and rolling cumulative (`rxg_for`, `rxg_against`, `rxg_pct`, `rhdcf_pct`, `rgsax_for`) performance, momentum indicators, and advanced game metrics.

#### B. Player Game, Rolling & Season Aggregates (`measurement/player/`)
* **`measurement/player/player_season.parquet`**: Player-level aggregated season statistics (rolling season-to-date for the active season and finalized season-ending totals for completed seasons in the same unified Parquet file), including skater `ixg`, `ixg_per_game`, `goals_above_expected`, `hd_shots`, `md_shots`, `ld_shots`, and goaltender `xga`, `gsax`, `gsax_per_game`, `xsave_pct`, `dsave_pct`, `hd_save_pct`.
* **`measurement/player/player_game.parquet`**: Player game-by-game and cumulative rolling season-to-date statistics (`ixg`, `g_minus_ixg`, `hd_shots`, `xga`, `gsax`, `cum_gp`, `cum_goals`, `cum_points`, `cum_ixg`, `cum_g_minus_ixg`, `cum_xga`, `cum_gsax`, `cum_sv_pct`, `cum_xsv_pct`, `cum_dsv_pct`, and `is_season_end` indicator).

#### C. Physical Actions & Hit Analytics (`measurement/action/hits/`)
* **`measurement/action/hits/hit_impact.parquet`**: Downstream sequence outcomes and 30-second net shot/goal impact sequences.
* **`measurement/action/hits/team_season_hit_kpi.parquet`**: Team-level physical KPIs and zone breakdown rates.
* **`measurement/action/hits/team_season_spatial_hits.parquet`**: Team spatial hit density grid coordinates.
* **`measurement/action/hits/hit_goal_window_30s.parquet`**: 30-second scoring telemetry and outcome tracking post-contact.
* **`measurement/action/hits/hit_next_events.parquet`**: Reaction latency and subsequent event categories.
* **`measurement/action/hits/hit_sequence_transitions.parquet`**: Markov chain state transitions following hits.
* **`measurement/action/hits/hit_zone_summary_kpi.parquet`**: Zone-level recovery rates, takeaways, and shot differentials.
* **`measurement/action/hits/hit_zone_summary.parquet`**: Zone-level recovery speed and opponent possession.
* **`measurement/action/hits/rink_grid_hit_sequences.parquet`**: Normalized rink coordinate sequences.
* **`measurement/action/hits/tabulated_hit_sequences.parquet`**: Multi-step post-hit event progressions (+1 to +4).

---

## Quickstart & Query Examples

### A. Zero-Install Streaming SQL with DuckDB
```sql
-- Query directly from Hugging Face over HTTP without local downloads
SELECT 
    season, 
    home_team, 
    COUNT(*) AS total_home_games,
    ROUND(AVG(home_score), 2) AS avg_goals_scored
FROM 'hf://datasets/{repo_id}/data/games.parquet'
WHERE season >= 20212022
GROUP BY season, home_team
ORDER BY season DESC, avg_goals_scored DESC;
```

### B. Python (`polars` or `pandas`)
```python
import polars as pl

url = "https://huggingface.co/datasets/{repo_id}/resolve/main/data/team_box.parquet"
df_teams = pl.read_parquet(url)
print(df_teams.filter(pl.col("team") == "MTL").tail(10))
```

### C. Hugging Face `datasets`
```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}", "games", split="train")
print(dataset[0])
```

---

## Data Source & Automation

> All underlying data is scraped, parsed, and transformed from the official NHL Edge API and market data providers.  
> 🏒 **Upstream Source Repository:** [GitHub — `justinjjlee/NHL-Analytics`](https://github.com/justinjjlee/NHL-Analytics)  
>
> Automated daily GitHub Actions workflows running during the NHL season:
> - **Game Stats & Plays:** Daily at 09:00 UTC (runs box score parsing, game summary metrics, and play-by-play events).
> - **Betting Odds:** Daily at 20:00 UTC (captures pre-game closing lines across US and Canadian markets).
> - **Player Aggregates:** Weekly runs updating skater and goalie season performance metrics.
"""


# -----------------------------------------------------------------------------
# CREDENTIAL RESOLUTION
# -----------------------------------------------------------------------------

def get_hf_token() -> Optional[str]:
    """Retrieves the Hugging Face access token from environment or local .env."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("rinklytics")
    if token:
        return token.strip().strip('"').strip("'")

    env_path = get_repo_root() / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("HF_TOKEN=") or line.startswith("rinklytics="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass
    return None


# -----------------------------------------------------------------------------
# DATA TRANSFORMATION FUNCTIONS
# -----------------------------------------------------------------------------

def build_games_panel(repo_root: Path, season_year: Optional[int] = None) -> pd.DataFrame:
    """Compiles game-level box files into a single panel table. If season_year is provided, compiles only that season."""
    pattern = f"{season_year}_box.csv" if season_year else "*_box.csv"
    box_files = sorted(glob.glob(str(repo_root / "latest" / "box" / pattern)))
    desc = f"season {season_year}" if season_year else f"{len(box_files)} historical files"
    print(f"📦 Compiling Games Panel from {desc}...")

    dfs = []
    for f in box_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            if "gameid" in df.columns:
                df = df.rename(columns={"gameid": "game_id"})
            dfs.append(df)
        except Exception as e:
            print(f"   ⚠️ Warning reading {os.path.basename(f)}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["game_id"]).copy()

    # Normalize column names
    col_mapping = {
        "date": "game_date",
        "tricode_for": "home_team",
        "id_for": "home_team_id",
        "tricode_against": "away_team",
        "id_against": "away_team_id",
        "metric_score_for": "home_score",
        "metric_score_against": "away_score",
        "tricode_winteam": "winning_team"
    }
    for old_col, new_col in col_mapping.items():
        if old_col in combined.columns:
            combined = combined.rename(columns={old_col: new_col})

    # Derive season identifier from game_id (first 4 digits)
    combined["game_id"] = pd.to_numeric(combined["game_id"], errors="coerce").fillna(0).astype("int64")
    start_year = combined["game_id"].astype(str).str[:4].astype(int)
    combined["season"] = start_year * 10000 + (start_year + 1)

    combined = combined.sort_values(by=["game_date", "game_id"]).reset_index(drop=True)
    print(f"   ✅ Games panel compiled: {len(combined):,} total games.")
    return combined


def build_team_box_panel(repo_root: Path, season_year: Optional[int] = None) -> pd.DataFrame:
    """Compiles team-level game statistics across seasons. If season_year is provided, compiles only that season."""
    pattern = f"{season_year}_box_team.csv" if season_year else "*_box_team.csv"
    team_files = sorted(glob.glob(str(repo_root / "latest" / "box" / pattern)))
    desc = f"season {season_year}" if season_year else f"{len(team_files)} historical files"
    print(f"📦 Compiling Team Box Panel from {desc}...")

    dfs = []
    for f in team_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            if "gameid" in df.columns:
                df = df.rename(columns={"gameid": "game_id"})
            dfs.append(df)
        except Exception as e:
            print(f"   ⚠️ Warning reading {os.path.basename(f)}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    if "team_tri" in combined.columns:
        combined = combined.drop_duplicates(subset=["game_id", "team_tri"]).copy()

    col_mapping = {
        "team_tri": "team",
        "gameDate": "game_date",
        "seasonIdx": "game_type",
        "poss_giveaway": "giveaways",
        "poss_takeaway": "takeaways",
        "shots_blocked": "blocked_shots",
        "goals_pp": "powerplay_goals",
        "faceoffWinningPctg": "faceoff_win_pct"
    }
    for old_col, new_col in col_mapping.items():
        if old_col in combined.columns:
            combined = combined.rename(columns={old_col: new_col})

    if "teamloc" in combined.columns:
        combined["is_home"] = combined["teamloc"].astype(str).str.lower() == "home"

    combined["game_id"] = pd.to_numeric(combined["game_id"], errors="coerce").fillna(0).astype("int64")
    start_year = combined["game_id"].astype(str).str[:4].astype(int)
    combined["season"] = start_year * 10000 + (start_year + 1)

    combined = combined.sort_values(by=["game_id", "team"]).reset_index(drop=True)
    print(f"   ✅ Team Box panel compiled: {len(combined):,} rows.")
    return combined


def build_player_box_panel(repo_root: Path, season_year: Optional[int] = None) -> pd.DataFrame:
    """Compiles individual player-game statistics across seasons. If season_year is provided, compiles only that season."""
    pattern = f"{season_year}_box_player.csv" if season_year else "*_box_player.csv"
    player_files = sorted(glob.glob(str(repo_root / "latest" / "box" / pattern)))
    desc = f"season {season_year}" if season_year else f"{len(player_files)} historical files"
    print(f"📦 Compiling Player Box Panel from {desc}...")

    dfs = []
    for f in player_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            if "gameid" in df.columns:
                df = df.rename(columns={"gameid": "game_id"})
            if "playerId" in df.columns:
                df = df.rename(columns={"playerId": "player_id"})
            dfs.append(df)
        except Exception as e:
            print(f"   ⚠️ Warning reading {os.path.basename(f)}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["game_id", "player_id"]).copy()

    # Map team abbreviation and opponent for every player row using teamloc
    if "teamloc" in combined.columns and "abbrev" in combined.columns:
        loc_map = combined[combined["abbrev"].notna()][["game_id", "teamloc", "abbrev"]].drop_duplicates()
        home_m = loc_map[loc_map["teamloc"] == "home"].set_index("game_id")["abbrev"]
        away_m = loc_map[loc_map["teamloc"] == "away"].set_index("game_id")["abbrev"]
        is_h = combined["teamloc"] == "home"
        combined["team"] = np.where(is_h, combined["game_id"].map(home_m), combined["game_id"].map(away_m))
        combined["opponent"] = np.where(is_h, combined["game_id"].map(away_m), combined["game_id"].map(home_m))

    col_mapping = {
        "gameDate": "game_date",
        "seasonIdx": "game_type",
        "name.default": "player_name",
        "sweaterNumber": "sweater_number",
        "plusMinus": "plus_minus",
        "blockedShots": "blocked_shots",
        "faceoffWinningPctg": "faceoff_pct",
        "powerPlayGoals": "powerplay_goals",
        "savePctg": "save_pct",
        "shotsAgainst": "shots_against",
        "goalsAgainst": "goals_against"
    }
    for old_col, new_col in col_mapping.items():
        if old_col in combined.columns:
            combined = combined.rename(columns={old_col: new_col})

    combined["game_id"] = pd.to_numeric(combined["game_id"], errors="coerce").fillna(0).astype("int64")
    combined["player_id"] = pd.to_numeric(combined["player_id"], errors="coerce").fillna(0).astype("int64")
    start_year = combined["game_id"].astype(str).str[:4].astype(int)
    combined["season"] = start_year * 10000 + (start_year + 1)

    # Keep key analytical columns including xG, GSAx, and rolling season-to-date metrics
    preferred_cols = [
        "game_id", "season", "game_date", "player_id", "player_name", "team", "opponent", "teamloc", "position",
        "sweater_number", "toi", "goals", "assists", "points", "plus_minus", "pim",
        "hits", "sog", "blocked_shots", "faceoff_pct", "powerplay_goals", "shifts", "giveaways", "takeaways",
        "saves", "shots_against", "goals_against", "save_pct", "decision",
        "ixg", "g_minus_ixg", "fenwick_shots", "hd_shots", "md_shots", "ld_shots", "rebound_shots", "rush_shots",
        "xga", "gsax", "xsave_pct", "dsave_pct", "hd_shots_against", "hd_goals_against", "hd_save_pct",
        "cum_gp", "cum_goals", "cum_assists", "cum_points", "cum_ixg", "cum_g_minus_ixg", "cum_hd_shots",
        "cum_xga", "cum_gsax", "cum_sv_pct", "cum_xsv_pct", "cum_dsv_pct", "is_season_end",
    ]
    actual_cols = [c for c in preferred_cols if c in combined.columns]
    combined = combined[actual_cols].sort_values(by=["game_id", "player_id"]).reset_index(drop=True)

    print(f"   ✅ Player Box panel compiled: {len(combined):,} rows.")
    return combined


def build_odds_panel(repo_root: Path, season_year: Optional[int] = None) -> pd.DataFrame:
    """Compiles closing betting lines across seasons. If season_year is provided, compiles only that season."""
    pattern = f"{season_year}_odds*.csv" if season_year else "*_odds*.csv"
    odds_files = sorted(glob.glob(str(repo_root / "latest" / "box" / pattern)))
    desc = f"season {season_year}" if season_year else f"{len(odds_files)} historical files"
    print(f"📦 Compiling Betting Odds Panel from {desc}...")

    dfs = []
    for f in odds_files:
        base_name = os.path.basename(f)
        try:
            df = pd.read_csv(f, low_memory=False)
            if "gameId" in df.columns:
                df = df.rename(columns={"gameId": "game_id"})
            elif "gameid" in df.columns:
                df = df.rename(columns={"gameid": "game_id"})

            # Tag country: "_CA" -> "CAN", "_US" -> "US", unadorned historical (e.g. 2024_odds.csv) -> "US"
            if "_CA" in base_name:
                df["country"] = "CAN"
            elif "_US" in base_name:
                df["country"] = "US"
            else:
                df["country"] = "US"

            dfs.append(df)
        except Exception as e:
            print(f"   ⚠️ Warning reading {os.path.basename(f)}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    col_mapping = {
        "homeTeam.abbrev": "home_team",
        "awayTeam.abbrev": "away_team",
        "home_odds_value": "home_odds",
        "away_odds_value": "away_odds",
        "bettingPartner": "betting_partner",
        "startTimeUTC": "start_time_utc",
        "lastUpdatedUTC": "last_updated_utc",
        "currentOddsDate": "game_date"
    }
    for old_col, new_col in col_mapping.items():
        if old_col in combined.columns:
            combined = combined.rename(columns={old_col: new_col})

    combined["game_id"] = pd.to_numeric(combined["game_id"], errors="coerce").fillna(0).astype("int64")

    # Deduplicate preserving distinct country markets (US vs CAN)
    dedup_cols = ["game_id", "odds_description", "country"]
    if all(c in combined.columns for c in dedup_cols):
        combined = combined.drop_duplicates(subset=dedup_cols).copy()

    preferred_cols = [
        "game_id", "country", "game_date", "home_team", "away_team",
        "odds_description", "home_odds", "away_odds", "betting_partner",
        "start_time_utc", "last_updated_utc"
    ]
    actual_cols = [c for c in preferred_cols if c in combined.columns]
    combined = combined[actual_cols].sort_values(by=["game_id", "country", "odds_description"]).reset_index(drop=True)

    us_count = (combined["country"] == "US").sum() if "country" in combined.columns else 0
    can_count = (combined["country"] == "CAN").sum() if "country" in combined.columns else 0
    print(f"   ✅ Odds panel compiled: {len(combined):,} rows (US: {us_count:,}, CAN: {can_count:,}).")
    return combined


def build_hit_impact_panel(repo_root: Path) -> pd.DataFrame:
    """Extracts pre-computed sequence analytics and hit impact records with yr_season."""
    master_path = repo_root / "dev" / "decision_science" / "impact_of_hits" / "data" / "df_master.parquet"
    if master_path.exists():
        print("📦 Compiling Hit Impact Sequence Panel from master parquet...")
        df = pd.read_parquet(master_path)
        if "yr_season" not in df.columns and "game_id" in df.columns:
            df["yr_season"] = (df["game_id"] // 1000000).astype("int64")
        print(f"   ✅ Hit impact panel loaded: {len(df):,} rows.")
        return df
    return pd.DataFrame()


def build_playbyplay_panel(repo_root: Path, output_file: Path) -> int:
    """Compiles all play-by-play events across all seasons into a single flat parquet table."""
    pbp_pattern = str(repo_root / "latest" / "play" / "*_playbyplay.csv")
    print(f"📦 Compiling Consolidated Play-by-Play Panel from {pbp_pattern}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        import duckdb
        query = f"""
        COPY (
            SELECT 
                TRY_CAST(gameid AS BIGINT) AS game_id,
                TRY_CAST(SUBSTRING(TRY_CAST(gameid AS VARCHAR), 1, 4) AS BIGINT) * 10000 + (TRY_CAST(SUBSTRING(TRY_CAST(gameid AS VARCHAR), 1, 4) AS BIGINT) + 1) AS season,
                TRY_CAST(seasonIdx AS INTEGER) AS game_type,
                TRY_CAST(gameDate AS VARCHAR) AS game_date,
                TRY_CAST(eventId AS BIGINT) AS event_id,
                TRY_CAST(timeInPeriod AS VARCHAR) AS time_in_period,
                TRY_CAST(timeRemaining AS VARCHAR) AS time_remaining,
                TRY_CAST("periodDescriptor.number" AS INTEGER) AS period,
                "periodDescriptor.periodType" AS period_type,
                typeDescKey AS event_type,
                "details.reason" AS reason,
                "details.eventOwnerTeam" AS owner_team,
                TRY_CAST("details.xCoord" AS DOUBLE) AS x_coord,
                TRY_CAST("details.yCoord" AS DOUBLE) AS y_coord,
                "details.zoneCode" AS zone_code,
                homeTeamDefendingSide AS home_defending_side,
                TRY_CAST("details.losingPlayerId" AS BIGINT) AS losing_player_id,
                TRY_CAST("details.winningPlayerId" AS BIGINT) AS winning_player_id,
                "details.shotType" AS shot_type,
                TRY_CAST("details.awaySOG" AS INTEGER) AS away_sog,
                TRY_CAST("details.homeSOG" AS INTEGER) AS home_sog,
                TRY_CAST("details.shootingPlayerId" AS BIGINT) AS shooting_player_id,
                TRY_CAST("details.blockingPlayerId" AS BIGINT) AS blocking_player_id,
                TRY_CAST("details.goalieInNetId" AS BIGINT) AS goalie_in_net_id,
                TRY_CAST("details.scoringPlayerId" AS BIGINT) AS scoring_player_id,
                TRY_CAST("details.assist1PlayerId" AS BIGINT) AS assist1_player_id,
                TRY_CAST("details.assist2PlayerId" AS BIGINT) AS assist2_player_id,
                TRY_CAST("details.hittingPlayerId" AS BIGINT) AS hitting_player_id,
                TRY_CAST("details.hitteePlayerId" AS BIGINT) AS hittee_player_id,
                "details.descKey" AS desc_key,
                TRY_CAST("details.duration" AS DOUBLE) AS duration,
                TRY_CAST("details.committedByPlayerId" AS BIGINT) AS committed_by_player_id,
                TRY_CAST("details.drawnByPlayerId" AS BIGINT) AS drawn_by_player_id
            FROM read_csv_auto('{pbp_pattern}', union_by_name=true)
            ORDER BY game_id, event_id
        ) TO '{str(output_file)}' (FORMAT PARQUET, COMPRESSION SNAPPY);
        """
        duckdb.query(query)
        count = duckdb.query(f"SELECT count(*) FROM '{str(output_file)}'").fetchone()[0]
        print(f"   ✅ Play-by-Play panel compiled: {count:,} rows into {output_file.name}")
        return count
    except Exception as e:
        print(f"   ⚠️ DuckDB pipeline encountered: {e}. Falling back to Pandas...")
        pbp_files = sorted(glob.glob(str(repo_root / "latest" / "play" / "*_playbyplay.csv")))
        dfs = []
        for f in pbp_files:
            try:
                df = pd.read_csv(f, low_memory=False)
                if "gameid" in df.columns:
                    df = df.rename(columns={"gameid": "game_id"})
                if "eventId" in df.columns:
                    df = df.rename(columns={"eventId": "event_id"})
                if "idx_season" in df.columns and "seasonIdx" in df.columns:
                    df = df.drop(columns=["idx_season"])
                if "seasonIdx" in df.columns:
                    df = df.rename(columns={"seasonIdx": "season"})
                dfs.append(df)
            except Exception as read_err:
                print(f"   ⚠️ Warning reading {os.path.basename(f)}: {read_err}")
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_parquet(output_file, index=False, compression="snappy")
        print(f"   ✅ Play-by-Play panel compiled via Pandas: {len(combined):,} rows.")
        return len(combined)


def build_playbyplay_player_panel(repo_root: Path, season_year: Optional[int] = None) -> pd.DataFrame:
    """Compiles active game roster and participant information across seasons. If season_year is provided, compiles only that season."""
    pattern = f"{season_year}_playbyplay_player.csv" if season_year else "*_playbyplay_player.csv"
    pbp_player_files = sorted(glob.glob(str(repo_root / "latest" / "play" / pattern)))
    desc = f"season {season_year}" if season_year else f"{len(pbp_player_files)} files"
    print(f"📦 Compiling Play-by-Play Player Panel from {desc}...")

    dfs = []
    for f in pbp_player_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            if "gameid" in df.columns:
                df = df.rename(columns={"gameid": "game_id"})
            if "playerId" in df.columns:
                df = df.rename(columns={"playerId": "player_id"})
            dfs.append(df)
        except Exception as e:
            print(f"   ⚠️ Warning reading {os.path.basename(f)}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    col_mapping = {
        "team_tri": "team",
        "sweaterNumber": "sweater_number",
        "positionCode": "position",
        "firstName.default": "first_name",
        "lastName.default": "last_name"
    }
    for oc, nc in col_mapping.items():
        if oc in combined.columns:
            combined = combined.rename(columns={oc: nc})

    combined["game_id"] = pd.to_numeric(combined["game_id"], errors="coerce").fillna(0).astype("int64")
    combined["player_id"] = pd.to_numeric(combined["player_id"], errors="coerce").fillna(0).astype("int64")
    start_year = combined["game_id"].astype(str).str[:4].astype(int)
    combined["season"] = start_year * 10000 + (start_year + 1)
    combined = combined.drop_duplicates(subset=["game_id", "player_id"]).sort_values(by=["game_id", "player_id"]).reset_index(drop=True)
    print(f"   ✅ Play-by-Play Player panel compiled: {len(combined):,} rows.")
    return combined


def build_shifts_panel(repo_root: Path, season_year: Optional[int] = None) -> pd.DataFrame:
    """Compiles player shift-level tracking data across available seasons. If season_year is provided, compiles only that season."""
    pattern = f"{season_year}_*shift*.csv" if season_year else "*_shift.csv"
    shift_files = sorted(glob.glob(str(repo_root / "latest" / "play" / pattern)))
    desc = f"season {season_year}" if season_year else f"{len(shift_files)} files"
    print(f"📦 Compiling Player Shifts Panel from {desc}...")

    dfs = []
    for f in shift_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            if "gameid" in df.columns:
                df = df.rename(columns={"gameid": "game_id"})
            if "id" in df.columns:
                df = df.rename(columns={"id": "shift_id"})
            if "playerid" in df.columns:
                df = df.rename(columns={"playerid": "player_id"})
            dfs.append(df)
        except Exception as e:
            print(f"   ⚠️ Warning reading {os.path.basename(f)}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    col_mapping = {
        "teamabbrev": "team",
        "shiftnumber": "shift_number",
        "starttime": "start_time",
        "endtime": "end_time",
        "typecode": "type_code",
        "detailcode": "detail_code",
        "eventnumber": "event_number",
        "eventdescription": "event_description",
        "eventdetails": "event_details"
    }
    for oc, nc in col_mapping.items():
        if oc in combined.columns:
            combined = combined.rename(columns={oc: nc})

    combined["game_id"] = pd.to_numeric(combined["game_id"], errors="coerce").fillna(0).astype("int64")
    combined["player_id"] = pd.to_numeric(combined["player_id"], errors="coerce").fillna(0).astype("int64")
    if "firstname" in combined.columns and "lastname" in combined.columns:
        combined["player_name"] = combined["firstname"].astype(str) + " " + combined["lastname"].astype(str)

    start_year = combined["game_id"].astype(str).str[:4].astype(int)
    combined["season"] = start_year * 10000 + (start_year + 1)
    if "shift_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["shift_id", "game_id", "player_id"])
    combined = combined.sort_values(by=["game_id", "period", "start_time"]).reset_index(drop=True)
    print(f"   ✅ Shifts panel compiled: {len(combined):,} rows.")
    return combined


# -----------------------------------------------------------------------------
# MEASUREMENT BUILDERS (Derived/Calculated Analytics with yr_season)
# -----------------------------------------------------------------------------

def build_team_season_df(repo_root: Path, season_year: Optional[int] = None) -> pd.DataFrame:
    """Compiles team_season metrics across seasons. If season_year is provided, compiles only that season."""
    pattern = f"{season_year}_team_season.csv" if season_year else "*_team_season.csv"
    files = sorted(glob.glob(str(repo_root / "latest" / "team" / "season" / pattern)))
    dfs = []
    for f in files:
        base = os.path.basename(f)
        yr_part = base.split("_")[0]
        if not yr_part.isdigit():
            continue
        yr = int(yr_part)
        try:
            df = pd.read_csv(f, low_memory=False)
            df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
            if "yr_season" not in df.columns:
                df["yr_season"] = yr
            else:
                df["yr_season"] = pd.to_numeric(df["yr_season"], errors="coerce").fillna(yr).astype("int64")
            dfs.append(df)
        except Exception as e:
            print(f"   ⚠️ Warning reading {base}: {e}")
    if not dfs:
        return pd.DataFrame()
    comb = pd.concat(dfs, ignore_index=True)
    dedup_cols = ["yr_season", "team_tri_for"] if "team_tri_for" in comb.columns else ["yr_season"]
    return comb.drop_duplicates(subset=dedup_cols).sort_values(by=dedup_cols).reset_index(drop=True)


def build_game_stats_df(repo_root: Path, season_year: Optional[int] = None) -> pd.DataFrame:
    """Compiles game_stats metrics across seasons. If season_year is provided, compiles only that season."""
    pattern = f"{season_year}_box_gameStats.csv" if season_year else "*_box_gameStats.csv"
    files = sorted(glob.glob(str(repo_root / "latest" / "box" / pattern)))
    dfs = []
    for f in files:
        base = os.path.basename(f)
        yr_part = base.split("_")[0]
        if not yr_part.isdigit():
            continue
        yr = int(yr_part)
        try:
            df = pd.read_csv(f, low_memory=False)
            df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
            if "yr_season" not in df.columns:
                df["yr_season"] = yr
            else:
                df["yr_season"] = pd.to_numeric(df["yr_season"], errors="coerce").fillna(yr).astype("int64")
            dfs.append(df)
        except Exception as e:
            print(f"   ⚠️ Warning reading {base}: {e}")
    if not dfs:
        return pd.DataFrame()
    comb = pd.concat(dfs, ignore_index=True)
    dedup_cols = ["gameid_for", "team_tri_for"] if all(c in comb.columns for c in ["gameid_for", "team_tri_for"]) else ["yr_season"]
    return comb.drop_duplicates(subset=dedup_cols).reset_index(drop=True)


def build_player_season_df(repo_root: Path, season_year: Optional[int] = None) -> pd.DataFrame:
    """
    Compiles player season metrics across seasons (rolling season-to-date for the active
    season and finalized season-ending totals for completed seasons in the same table).
    If season_year is provided, compiles only that season.
    """
    pattern = f"{season_year}_player.csv" if season_year else "*_player.csv"
    files = sorted(glob.glob(str(repo_root / "latest" / "player" / pattern)))
    all_files = sorted(glob.glob(str(repo_root / "latest" / "player" / "*_player.csv")))
    max_yr = max(
        [int(os.path.basename(f).split("_")[0]) for f in all_files if os.path.basename(f).split("_")[0].isdigit()]
        or [2025]
    )
    dfs = []
    for f in files:
        base = os.path.basename(f)
        yr_part = base.split("_")[0]
        if not yr_part.isdigit():
            continue
        yr = int(yr_part)
        try:
            df = pd.read_csv(f, low_memory=False)
            df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
            if "yr_season" not in df.columns:
                df["yr_season"] = yr
            else:
                df["yr_season"] = pd.to_numeric(df["yr_season"], errors="coerce").fillna(yr).astype("int64")
            df["season_status"] = "rolling_to_date" if yr >= max_yr else "season_final"
            dfs.append(df)
        except Exception as e:
            print(f"   ⚠️ Warning reading {base}: {e}")
    if not dfs:
        return pd.DataFrame()
    comb = pd.concat(dfs, ignore_index=True)
    return comb.drop_duplicates().reset_index(drop=True)


def build_player_game_df(repo_root: Path, season_year: Optional[int] = None) -> pd.DataFrame:
    """
    Compiles player game-level and cumulative rolling season-to-date metrics
    (including ixg, g_minus_ixg, hd_shots, xga, gsax, cum_ixg, cum_gsax, and is_season_end)
    for `measurement/player/player_game.parquet`.
    """
    df = build_player_box_panel(repo_root, season_year=season_year)
    if df.empty:
        return df
    df["yr_season"] = (df["game_id"] // 1000000).astype("int64")
    return df


def build_hits_measurement_df(repo_root: Path, ht: str, season_year: Optional[int] = None) -> pd.DataFrame:
    """Compiles individual action/hits measurement table across seasons. If season_year is provided, compiles only that season."""
    pattern = f"{season_year}_{ht}.csv" if season_year else f"*_{ht}.csv"
    files = sorted(glob.glob(str(repo_root / "latest" / "action" / "hits" / pattern)))
    dfs = []
    for f in files:
        base = os.path.basename(f)
        yr_part = base.split("_")[0]
        yr = int(yr_part) if yr_part.isdigit() else (season_year or 2026)
        try:
            df = pd.read_csv(f, low_memory=False)
            df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
            if "yr_season" not in df.columns:
                df["yr_season"] = yr
            else:
                df["yr_season"] = pd.to_numeric(df["yr_season"], errors="coerce").fillna(yr).astype("int64")
            dfs.append(df)
        except Exception as e:
            print(f"   ⚠️ Warning reading {base}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True).drop_duplicates().reset_index(drop=True)


def build_measurement_box_panels(repo_root: Path, box_dir: Path):
    """Compiles box-derived measurements: team_season and game_stats."""
    box_dir.mkdir(parents=True, exist_ok=True)
    print("📦 Compiling Measurement Box Panels (team_season, game_stats)...")

    comb_ts = build_team_season_df(repo_root)
    if not comb_ts.empty:
        dest_ts = box_dir / "team_season.parquet"
        comb_ts.to_parquet(dest_ts, index=False, compression="snappy")
        print(f"   ✅ measurement/box/team_season.parquet compiled: {len(comb_ts):,} rows.")

    comb_gs = build_game_stats_df(repo_root)
    if not comb_gs.empty:
        dest_gs = box_dir / "game_stats.parquet"
        comb_gs.to_parquet(dest_gs, index=False, compression="snappy")
        print(f"   ✅ measurement/box/game_stats.parquet compiled: {len(comb_gs):,} rows.")


def build_measurement_player_panels(repo_root: Path, player_dir: Path):
    """Compiles player season and game-level + rolling aggregates into measurement/player/."""
    player_dir.mkdir(parents=True, exist_ok=True)
    print("📦 Compiling Measurement Player Panels (player_season, player_game)...")

    comb = build_player_season_df(repo_root)
    if not comb.empty:
        dest = player_dir / "player_season.parquet"
        comb.to_parquet(dest, index=False, compression="snappy")
        print(f"   ✅ measurement/player/player_season.parquet compiled: {len(comb):,} rows.")

    comb_pg = build_player_game_df(repo_root)
    if not comb_pg.empty:
        dest_pg = player_dir / "player_game.parquet"
        comb_pg.to_parquet(dest_pg, index=False, compression="snappy")
        print(f"   ✅ measurement/player/player_game.parquet compiled: {len(comb_pg):,} rows.")


def build_measurement_action_hits_panels(repo_root: Path, hits_dir: Path):
    """Compiles action/hits measurements across all seasons."""
    hits_dir.mkdir(parents=True, exist_ok=True)
    print("📦 Compiling Measurement Action/Hits Panels...")

    hit_types = [
        "team_season_hit_kpi",
        "team_season_spatial_hits",
        "hit_zone_summary_kpi",
        "hit_zone_summary",
        "hit_goal_window_30s",
        "hit_next_events",
        "hit_sequence_transitions",
        "rink_grid_hit_sequences",
        "tabulated_hit_sequences"
    ]

    for ht in hit_types:
        comb = build_hits_measurement_df(repo_root, ht)
        if not comb.empty:
            dest = hits_dir / f"{ht}.parquet"
            comb.to_parquet(dest, index=False, compression="snappy")
            print(f"   ✅ measurement/action/hits/{ht}.parquet compiled: {len(comb):,} rows.")

    # Also compile hit_impact into measurement/action/hits/
    master_path = repo_root / "dev" / "decision_science" / "impact_of_hits" / "data" / "df_master.parquet"
    if master_path.exists():
        df_impact = pd.read_parquet(master_path)
        if "yr_season" not in df_impact.columns and "game_id" in df_impact.columns:
            df_impact["yr_season"] = (df_impact["game_id"] // 1000000).astype("int64")
        df_impact.to_parquet(hits_dir / "hit_impact.parquet", index=False, compression="snappy")
        print(f"   ✅ measurement/action/hits/hit_impact.parquet compiled: {len(df_impact):,} rows.")


# -----------------------------------------------------------------------------
# MAIN BOOTSTRAP WORKFLOW
# -----------------------------------------------------------------------------

def bootstrap(
    repo_id: str,
    token: Optional[str] = None,
    staging_dir: Optional[Path] = None,
    skip_upload: bool = False,
    use_cli: bool = False,
    is_private: bool = False,
    commit_message: str = "Update NHL Analytics longitudinal panel dataset"
):
    repo_root = get_repo_root()
    staging_root = staging_dir or (repo_root / "deploy" / "huggingface" / "staging_data")
    data_dir = staging_root / "data"

    print("\n" + "=" * 70)
    print("🚀 NHL ANALYTICS — HUGGING FACE DATASET BOOTSTRAP PIPELINE")
    print("=" * 70)
    print(f"Repository Root: {repo_root}")
    print(f"Staging Directory: {staging_root}")
    print(f"Target HF Dataset: {repo_id}")
    print(f"Skip Upload: {skip_upload} | Use HF CLI: {use_cli}")
    print("-" * 70)

    # 1. Clean Staging Directory
    if staging_root.exists():
        shutil.rmtree(staging_root)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 2. Build Tables
    start_time = time.time()

    df_games = build_games_panel(repo_root)
    if not df_games.empty:
        df_games.to_parquet(data_dir / "games.parquet", index=False, compression="snappy")

    df_team = build_team_box_panel(repo_root)
    if not df_team.empty:
        df_team.to_parquet(data_dir / "team_box.parquet", index=False, compression="snappy")

    df_player = build_player_box_panel(repo_root)
    if not df_player.empty:
        df_player.to_parquet(data_dir / "player_box.parquet", index=False, compression="snappy")

    df_odds = build_odds_panel(repo_root)
    if not df_odds.empty:
        df_odds.to_parquet(data_dir / "odds.parquet", index=False, compression="snappy")

    df_hits = build_hit_impact_panel(repo_root)
    if not df_hits.empty:
        df_hits.to_parquet(data_dir / "hit_impact.parquet", index=False, compression="snappy")

    # Build consolidated play-by-play panel
    build_playbyplay_panel(repo_root, data_dir / "playbyplay.parquet")

    df_pbp_player = build_playbyplay_player_panel(repo_root)
    if not df_pbp_player.empty:
        df_pbp_player.to_parquet(data_dir / "playbyplay_player.parquet", index=False, compression="snappy")

    df_shifts = build_shifts_panel(repo_root)
    if not df_shifts.empty:
        df_shifts.to_parquet(data_dir / "shifts.parquet", index=False, compression="snappy")

    # 3. Build Measurement Tier (measurement/)
    measurement_dir = staging_root / "measurement"
    build_measurement_box_panels(repo_root, measurement_dir / "box")
    build_measurement_player_panels(repo_root, measurement_dir / "player")
    build_measurement_action_hits_panels(repo_root, measurement_dir / "action" / "hits")

    # 4. Generate Dataset Card (README.md)
    readme_content = DATASET_CARD_TEMPLATE.replace("{repo_id}", repo_id)
    (staging_root / "README.md").write_text(readme_content, encoding="utf-8")
    print("\n   📄 Dataset Card (README.md) generated with YAML metadata.")

    elapsed_compile = time.time() - start_time
    print(f"\n✨ Compilation complete in {elapsed_compile:.1f}s.")

    # Print summary of generated artifacts
    total_size = sum(f.stat().st_size for f in staging_root.rglob("*") if f.is_file())
    print(f"📊 Staged Payload Size: {total_size / (1024 * 1024):.1f} MB across {len(list(staging_root.rglob('*')))} files.")

    if skip_upload:
        print("\n✅ Skipping upload as requested (--skip-upload). Staged files remain in:")
        print(f"   {staging_root}")
        return

    # 4. Upload to Hugging Face
    resolved_token = token or get_hf_token()
    if not resolved_token:
        print("\n❌ Error: Hugging Face access token not found.")
        print("   Set HF_TOKEN in your environment or add rinklytics=<token> in .env.")
        sys.exit(1)

    print("\n" + "-" * 70)
    print(f"🌐 Initiating Upload to Hugging Face Dataset: {repo_id}...")
    print("-" * 70)

    if use_cli:
        # Upload via huggingface-cli
        cli_cmd = [
            "huggingface-cli", "upload",
            repo_id,
            str(staging_root),
            ".",
            "--repo-type", "dataset",
            "--token", resolved_token
        ]
        print(f"Running command: {' '.join(cli_cmd[:4])} ...")
        res = subprocess.run(cli_cmd)
        if res.returncode != 0:
            print("\n❌ huggingface-cli upload returned non-zero exit code. Falling back to HfApi...")
            _upload_via_api(repo_id, resolved_token, staging_root, is_private, commit_message)
        else:
            print(f"\n🎉 SUCCESS! Dataset published at: https://huggingface.co/datasets/{repo_id}")
    else:
        _upload_via_api(repo_id, resolved_token, staging_root, is_private, commit_message)


def _upload_via_api(repo_id: str, token: str, staging_root: Path, is_private: bool, commit_message: str = "Update NHL Analytics longitudinal panel dataset"):
    """Uploads directory via Python huggingface_hub HfApi."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("❌ huggingface_hub is required. Install with: pip install huggingface_hub")
        print("Or run via: uv run --with huggingface_hub python deploy/huggingface/bootstrap_dataset.py")
        sys.exit(1)

    api = HfApi(token=token)
    user = api.whoami()
    print(f"Authenticated as: {user.get('name', 'Unknown User')}")

    print(f"Ensuring repository exists ({repo_id})...")
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=is_private, exist_ok=True)

    print(f"Uploading files from {staging_root} in an atomic commit...")
    api.upload_folder(
        folder_path=str(staging_root),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
        delete_patterns=["data/playbyplay/*"]
    )

    print("\n" + "=" * 70)
    print(f"🎉 SUCCESS! Dataset published successfully at:")
    print(f"   👉 https://huggingface.co/datasets/{repo_id}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap and upload NHL Analytics panel datasets to Hugging Face.")
    parser.add_argument("--repo", default="jjerlee/nhl-analytics", help="Target Hugging Face Dataset repo ID (default: jjerlee/nhl-analytics)")
    parser.add_argument("--token", default=None, help="Hugging Face access token (defaults to HF_TOKEN env variable)")
    parser.add_argument("--staging-dir", default=None, type=Path, help="Custom local staging directory")
    parser.add_argument("--skip-upload", action="store_true", help="Compile tables locally without uploading")
    parser.add_argument("--use-cli", action="store_true", help="Upload using huggingface-cli subprocess")
    parser.add_argument("--private", action="store_true", help="Create the dataset repository as private")
    parser.add_argument("--message", default="Update NHL Analytics longitudinal panel dataset with rosters and shifts", help="Git commit message for the upload")

    args = parser.parse_args()
    bootstrap(
        repo_id=args.repo,
        token=args.token,
        staging_dir=args.staging_dir,
        skip_upload=args.skip_upload,
        use_cli=args.use_cli,
        is_private=args.private,
        commit_message=args.message
    )
