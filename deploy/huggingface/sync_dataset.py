#!/usr/bin/env python3
"""
NHL Analytics - Incremental Dataset Sync Engine for Hugging Face

Synchronizes incoming daily NHL API updates (game records, box scores, betting odds,
play-by-play events, and measurement metrics) to the Hugging Face Dataset repository.

Key features:
1. Active season partition replacement: isolates and updates only the active season
   while strictly preserving historical vintages (2013-2024).
2. Primary key deduplication: ensures idempotent updates across all data tiers
   (e.g. game_id, [game_id, team], [game_id, player_id], [game_id, odds_description, country]).
3. Change detection: compares candidate updates with remote snapshots to skip unnecessary
   commits when data has not changed.
4. Country-aware odds harmonization: preserves distinct 'US' and 'CAN' betting lines.
5. Atomic upload: stages all modified tables and executes a single atomic Git commit.

Usage:
    # Sync box scores, game stats, and play-by-play (09:00 UTC workflow)
    python deploy/huggingface/sync_dataset.py --target box

    # Sync pre-game betting closing lines (20:00 UTC workflow)
    python deploy/huggingface/sync_dataset.py --target odds

    # Sync player season metrics (weekly Monday workflow)
    python deploy/huggingface/sync_dataset.py --target player

    # Sync all domains
    python deploy/huggingface/sync_dataset.py --target all
"""

import argparse
import glob
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:
    print("❌ Error: huggingface_hub is required. Install via: pip install huggingface_hub")
    sys.exit(1)

# Import builders from bootstrap_dataset
current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from bootstrap_dataset import (
        build_games_panel,
        build_team_box_panel,
        build_player_box_panel,
        build_odds_panel,
        build_playbyplay_player_panel,
        build_shifts_panel,
        build_team_season_df,
        build_game_stats_df,
        build_player_season_df,
        build_player_game_df,
        build_hits_measurement_df,
        get_repo_root,
        get_hf_token,
    )
except ImportError:
    from deploy.huggingface.bootstrap_dataset import (
        build_games_panel,
        build_team_box_panel,
        build_player_box_panel,
        build_odds_panel,
        build_playbyplay_player_panel,
        build_shifts_panel,
        build_team_season_df,
        build_game_stats_df,
        build_player_season_df,
        build_player_game_df,
        build_hits_measurement_df,
        get_repo_root,
        get_hf_token,
    )


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def resolve_active_season_year(repo_root: Path) -> int:
    """
    Determines the active season calendar prefix from latest files or current date.
    Looks for the highest year prefix in latest/box/*_box.csv.
    """
    box_files = glob.glob(str(repo_root / "latest" / "box" / "*_box.csv"))
    years = []
    for f in box_files:
        base = os.path.basename(f)
        part = base.split("_")[0]
        if part.isdigit():
            years.append(int(part))
    if years:
        return max(years)

    # Fallback to current year logic
    import datetime
    today = datetime.datetime.today()
    return today.year if today.month >= 8 else today.year - 1


def load_remote_parquet(repo_id: str, rel_path: str, token: Optional[str]) -> Optional[pd.DataFrame]:
    """
    Attempts to download and load a Parquet table from the remote Hugging Face dataset.
    Returns None if the file does not exist or fails to download.
    """
    try:
        cached_file = hf_hub_download(
            repo_id=repo_id,
            filename=rel_path,
            repo_type="dataset",
            token=token,
        )
        return pd.read_parquet(cached_file)
    except Exception as e:
        print(f"   ℹ️ Remote {rel_path} not found or unavailable ({e}). Fallback to local build.")
        return None


def has_changed(
    df_new: pd.DataFrame,
    df_old: Optional[pd.DataFrame],
    key_cols: List[str],
    season_year: Optional[int] = None
) -> bool:
    """
    Checks whether df_new differs from df_old.
    Returns True if length or active season content differs; False if identical.
    """
    if df_old is None:
        return True
    if len(df_new) != len(df_old):
        return True
    if list(df_new.columns) != list(df_old.columns):
        return True
    if df_new.equals(df_old):
        return False

    # Focus deep equality test on the active season slice for high performance
    if season_year is not None:
        if "yr_season" in df_new.columns and "yr_season" in df_old.columns:
            s_new = df_new[df_new["yr_season"] == season_year].reset_index(drop=True)
            s_old = df_old[df_old["yr_season"] == season_year].reset_index(drop=True)
        elif "game_id" in df_new.columns and "game_id" in df_old.columns:
            s_new = df_new[df_new["game_id"].astype(str).str[:4].astype(int) == season_year].reset_index(drop=True)
            s_old = df_old[df_old["game_id"].astype(str).str[:4].astype(int) == season_year].reset_index(drop=True)
        else:
            s_new = df_new
            s_old = df_old
    else:
        s_new = df_new
        s_old = df_old

    if len(s_new) != len(s_old):
        return True
    if s_new.equals(s_old):
        return False

    try:
        norm_new = s_new.fillna(pd.NA)
        norm_old = s_old.fillna(pd.NA)
        if norm_new.equals(norm_old):
            return False
        pd.testing.assert_frame_equal(norm_new, norm_old, check_dtype=False, check_like=True)
        return False
    except (AssertionError, Exception):
        return True


# -----------------------------------------------------------------------------
# SYNCHRONIZATION WORKERS BY DOMAIN
# -----------------------------------------------------------------------------

def sync_games(repo_root: Path, repo_id: str, token: Optional[str], season_year: int, staging_dir: Path, force: bool = False) -> bool:
    """Syncs data/games.parquet."""
    rel_path = "data/games.parquet"
    df_remote = load_remote_parquet(repo_id, rel_path, token)
    df_fresh = build_games_panel(repo_root, season_year=None if force else season_year)
    if df_fresh.empty and df_remote is not None:
        return False

    if not force and df_remote is not None and "xg_for" in df_remote.columns and not df_fresh.empty:
        remote_start_years = df_remote["game_id"].astype(str).str[:4].astype(int)
        df_hist = df_remote[remote_start_years != season_year].copy()
        df_combined = pd.concat([df_hist, df_fresh], ignore_index=True)
    elif not force and df_remote is not None and "xg_for" in df_remote.columns:
        df_combined = df_remote
    else:
        df_combined = df_fresh if force else build_games_panel(repo_root)

    df_updated = df_combined.drop_duplicates(subset=["game_id"], keep="last")
    df_updated = df_updated.sort_values(by=["game_date", "game_id"]).reset_index(drop=True)

    if force or has_changed(df_updated, df_remote, ["game_id"], season_year=None if force else season_year):
        out_path = staging_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_updated.to_parquet(out_path, index=False, compression="snappy")
        print(f"   🔄 Staged {rel_path}: {len(df_updated):,} rows (was {len(df_remote) if df_remote is not None else 0:,}).")
        return True
    print(f"   ✓ {rel_path} is up-to-date ({len(df_updated):,} rows).")
    return False


def sync_team_box(repo_root: Path, repo_id: str, token: Optional[str], season_year: int, staging_dir: Path, force: bool = False) -> bool:
    """Syncs data/team_box.parquet."""
    rel_path = "data/team_box.parquet"
    df_remote = load_remote_parquet(repo_id, rel_path, token)
    df_fresh = build_team_box_panel(repo_root, season_year=None if force else season_year)
    if df_fresh.empty and df_remote is not None:
        return False

    if not force and df_remote is not None and "xg" in df_remote.columns and not df_fresh.empty:
        remote_start_years = df_remote["game_id"].astype(str).str[:4].astype(int)
        df_hist = df_remote[remote_start_years != season_year].copy()
        df_combined = pd.concat([df_hist, df_fresh], ignore_index=True)
    elif not force and df_remote is not None and "xg" in df_remote.columns:
        df_combined = df_remote
    else:
        df_combined = df_fresh if force else build_team_box_panel(repo_root)

    df_updated = df_combined.drop_duplicates(subset=["game_id", "team"], keep="last")
    df_updated = df_updated.sort_values(by=["game_id", "team"]).reset_index(drop=True)

    if force or has_changed(df_updated, df_remote, ["game_id", "team"], season_year=None if force else season_year):
        out_path = staging_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_updated.to_parquet(out_path, index=False, compression="snappy")
        print(f"   🔄 Staged {rel_path}: {len(df_updated):,} rows (was {len(df_remote) if df_remote is not None else 0:,}).")
        return True
    print(f"   ✓ {rel_path} is up-to-date ({len(df_updated):,} rows).")
    return False


def sync_player_box(repo_root: Path, repo_id: str, token: Optional[str], season_year: int, staging_dir: Path, force: bool = False) -> bool:
    """Syncs data/player_box.parquet."""
    rel_path = "data/player_box.parquet"
    df_remote = load_remote_parquet(repo_id, rel_path, token)
    df_fresh = build_player_box_panel(repo_root, season_year=None if force else season_year)
    if df_fresh.empty and df_remote is not None:
        return False

    if not force and df_remote is not None and "opponent" in df_remote.columns and "ixg" in df_remote.columns and not df_fresh.empty:
        remote_start_years = df_remote["game_id"].astype(str).str[:4].astype(int)
        df_hist = df_remote[remote_start_years != season_year].copy()
        df_combined = pd.concat([df_hist, df_fresh], ignore_index=True)
    elif not force and df_remote is not None and "opponent" in df_remote.columns and "ixg" in df_remote.columns:
        df_combined = df_remote
    else:
        df_combined = df_fresh if force else build_player_box_panel(repo_root)

    df_updated = df_combined.drop_duplicates(subset=["game_id", "player_id"], keep="last")
    df_updated = df_updated.sort_values(by=["game_id", "player_id"]).reset_index(drop=True)

    if force or has_changed(df_updated, df_remote, ["game_id", "player_id"], season_year=None if force else season_year):
        out_path = staging_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_updated.to_parquet(out_path, index=False, compression="snappy")
        print(f"   🔄 Staged {rel_path}: {len(df_updated):,} rows (was {len(df_remote) if df_remote is not None else 0:,}).")
        return True
    print(f"   ✓ {rel_path} is up-to-date ({len(df_updated):,} rows).")
    return False


def sync_odds(repo_root: Path, repo_id: str, token: Optional[str], season_year: int, staging_dir: Path) -> bool:
    """Syncs data/odds.parquet preserving US vs CAN market segments."""
    rel_path = "data/odds.parquet"
    df_remote = load_remote_parquet(repo_id, rel_path, token)
    df_fresh = build_odds_panel(repo_root, season_year=season_year)
    if df_fresh.empty and df_remote is not None:
        return False

    if df_remote is not None and not df_fresh.empty:
        remote_start_years = df_remote["game_id"].astype(str).str[:4].astype(int)
        df_hist = df_remote[remote_start_years != season_year].copy()
        df_combined = pd.concat([df_hist, df_fresh], ignore_index=True)
    elif df_remote is not None:
        df_combined = df_remote
    else:
        df_combined = build_odds_panel(repo_root)

    dedup_cols = ["game_id", "odds_description", "country"]
    df_updated = df_combined.drop_duplicates(subset=dedup_cols, keep="last")
    df_updated = df_updated.sort_values(by=["game_id", "country", "odds_description"]).reset_index(drop=True)

    if has_changed(df_updated, df_remote, dedup_cols, season_year=season_year):
        out_path = staging_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_updated.to_parquet(out_path, index=False, compression="snappy")
        us_c = (df_updated["country"] == "US").sum() if "country" in df_updated.columns else 0
        can_c = (df_updated["country"] == "CAN").sum() if "country" in df_updated.columns else 0
        print(f"   🔄 Staged {rel_path}: {len(df_updated):,} rows (US: {us_c:,}, CAN: {can_c:,}).")
        return True
    print(f"   ✓ {rel_path} is up-to-date ({len(df_updated):,} rows).")
    return False


def sync_playbyplay(repo_root: Path, repo_id: str, token: Optional[str], season_year: int, staging_dir: Path) -> bool:
    """Syncs data/playbyplay.parquet using DuckDB extraction for the active season."""
    rel_path = "data/playbyplay.parquet"
    df_remote = load_remote_parquet(repo_id, rel_path, token)

    csv_path = repo_root / "latest" / "play" / f"{season_year}_playbyplay.csv"
    if not csv_path.exists():
        if df_remote is not None:
            return False
        out_path = staging_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        from deploy.huggingface.bootstrap_dataset import build_playbyplay_panel
        build_playbyplay_panel(repo_root, out_path)
        return True

    # Parse active season plays via DuckDB
    try:
        import duckdb
        query = f"""
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
            FROM read_csv_auto('{str(csv_path)}', union_by_name=true)
            ORDER BY game_id, event_id
        """
        df_fresh = duckdb.query(query).df()
    except Exception as e:
        print(f"   ⚠️ DuckDB parse failed ({e}), using pandas fallback...")
        df_fresh = pd.read_csv(csv_path, low_memory=False)
        if "gameid" in df_fresh.columns:
            df_fresh = df_fresh.rename(columns={"gameid": "game_id"})
        if "eventId" in df_fresh.columns:
            df_fresh = df_fresh.rename(columns={"eventId": "event_id"})

    if df_remote is not None and not df_fresh.empty:
        remote_start_years = df_remote["game_id"].astype(str).str[:4].astype(int)
        df_hist = df_remote[remote_start_years != season_year].copy()
        df_combined = pd.concat([df_hist, df_fresh], ignore_index=True)
    elif df_remote is not None:
        df_combined = df_remote
    else:
        df_combined = df_fresh

    df_updated = df_combined.drop_duplicates(subset=["game_id", "event_id"], keep="last")
    df_updated = df_updated.sort_values(by=["game_id", "event_id"]).reset_index(drop=True)

    if has_changed(df_updated, df_remote, ["game_id", "event_id"], season_year=season_year):
        out_path = staging_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_updated.to_parquet(out_path, index=False, compression="snappy")
        print(f"   🔄 Staged {rel_path}: {len(df_updated):,} rows (was {len(df_remote) if df_remote is not None else 0:,}).")
        return True
    print(f"   ✓ {rel_path} is up-to-date ({len(df_updated):,} rows).")
    return False


def sync_playbyplay_player(repo_root: Path, repo_id: str, token: Optional[str], season_year: int, staging_dir: Path) -> bool:
    """Syncs data/playbyplay_player.parquet."""
    rel_path = "data/playbyplay_player.parquet"
    df_remote = load_remote_parquet(repo_id, rel_path, token)
    df_fresh = build_playbyplay_player_panel(repo_root, season_year=season_year)
    if df_fresh.empty and df_remote is not None:
        return False

    if df_remote is not None and not df_fresh.empty:
        remote_start_years = df_remote["game_id"].astype(str).str[:4].astype(int)
        df_hist = df_remote[remote_start_years != season_year].copy()
        df_combined = pd.concat([df_hist, df_fresh], ignore_index=True)
    elif df_remote is not None:
        df_combined = df_remote
    else:
        df_combined = build_playbyplay_player_panel(repo_root)

    df_updated = df_combined.drop_duplicates(subset=["game_id", "player_id"], keep="last")
    df_updated = df_updated.sort_values(by=["game_id", "player_id"]).reset_index(drop=True)

    if has_changed(df_updated, df_remote, ["game_id", "player_id"], season_year=season_year):
        out_path = staging_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_updated.to_parquet(out_path, index=False, compression="snappy")
        print(f"   🔄 Staged {rel_path}: {len(df_updated):,} rows (was {len(df_remote) if df_remote is not None else 0:,}).")
        return True
    print(f"   ✓ {rel_path} is up-to-date ({len(df_updated):,} rows).")
    return False


def sync_shifts(repo_root: Path, repo_id: str, token: Optional[str], season_year: int, staging_dir: Path) -> bool:
    """Syncs data/shifts.parquet."""
    rel_path = "data/shifts.parquet"
    df_remote = load_remote_parquet(repo_id, rel_path, token)
    df_fresh = build_shifts_panel(repo_root, season_year=season_year)
    if df_fresh.empty and df_remote is not None:
        return False

    if df_remote is not None and not df_fresh.empty:
        remote_start_years = df_remote["game_id"].astype(str).str[:4].astype(int)
        df_hist = df_remote[remote_start_years != season_year].copy()
        df_combined = pd.concat([df_hist, df_fresh], ignore_index=True)
    elif df_remote is not None:
        df_combined = df_remote
    else:
        df_combined = build_shifts_panel(repo_root)

    dedup_cols = ["game_id", "player_id", "shift_number", "start_time"] if all(c in df_combined.columns for c in ["game_id", "player_id", "shift_number", "start_time"]) else ["shift_id"]
    sort_cols = [c for c in ["game_id", "period", "start_time"] if c in df_combined.columns]
    df_updated = df_combined.drop_duplicates(subset=dedup_cols, keep="last")
    if sort_cols:
        df_updated = df_updated.sort_values(by=sort_cols)
    df_updated = df_updated.reset_index(drop=True)

    if has_changed(df_updated, df_remote, dedup_cols, season_year=season_year):
        out_path = staging_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_updated.to_parquet(out_path, index=False, compression="snappy")
        print(f"   🔄 Staged {rel_path}: {len(df_updated):,} rows (was {len(df_remote) if df_remote is not None else 0:,}).")
        return True
    print(f"   ✓ {rel_path} is up-to-date ({len(df_updated):,} rows).")
    return False


def sync_measurement_box(repo_root: Path, repo_id: str, token: Optional[str], season_year: int, staging_dir: Path, force: bool = False) -> bool:
    """Syncs measurement/box/team_season.parquet and game_stats.parquet."""
    modified = False

    # 1. Team Season
    rel_ts = "measurement/box/team_season.parquet"
    df_rem_ts = load_remote_parquet(repo_id, rel_ts, token)
    df_fr_ts = build_team_season_df(repo_root, season_year=None if force else season_year)
    if not df_fr_ts.empty or df_rem_ts is None:
        if not force and df_rem_ts is not None and "xg_for" in df_rem_ts.columns:
            df_hist = df_rem_ts[df_rem_ts["yr_season"] != season_year].copy()
            df_comb = pd.concat([df_hist, df_fr_ts], ignore_index=True)
        else:
            df_comb = df_fr_ts if force else build_team_season_df(repo_root)
        dedup_cols = ["yr_season", "team_tri_for"] if "team_tri_for" in df_comb.columns else ["yr_season"]
        df_up_ts = df_comb.drop_duplicates(subset=dedup_cols, keep="last").sort_values(by=dedup_cols).reset_index(drop=True)
        if force or has_changed(df_up_ts, df_rem_ts, dedup_cols, season_year=None if force else season_year):
            p = staging_dir / rel_ts
            p.parent.mkdir(parents=True, exist_ok=True)
            df_up_ts.to_parquet(p, index=False, compression="snappy")
            print(f"   🔄 Staged {rel_ts}: {len(df_up_ts):,} rows (was {len(df_rem_ts) if df_rem_ts is not None else 0:,}).")
            modified = True
        else:
            print(f"   ✓ {rel_ts} is up-to-date ({len(df_up_ts):,} rows).")

    # 2. Game Stats
    rel_gs = "measurement/box/game_stats.parquet"
    df_rem_gs = load_remote_parquet(repo_id, rel_gs, token)
    df_fr_gs = build_game_stats_df(repo_root, season_year=None if force else season_year)
    if not df_fr_gs.empty or df_rem_gs is None:
        if not force and df_rem_gs is not None and "xg_for" in df_rem_gs.columns:
            df_hist = df_rem_gs[df_rem_gs["yr_season"] != season_year].copy()
            df_comb = pd.concat([df_hist, df_fr_gs], ignore_index=True)
        else:
            df_comb = df_fr_gs if force else build_game_stats_df(repo_root)
        dedup_cols = ["gameid_for", "team_tri_for"] if all(c in df_comb.columns for c in ["gameid_for", "team_tri_for"]) else ["yr_season"]
        df_up_gs = df_comb.drop_duplicates(subset=dedup_cols, keep="last").reset_index(drop=True)
        if force or has_changed(df_up_gs, df_rem_gs, dedup_cols, season_year=None if force else season_year):
            p = staging_dir / rel_gs
            p.parent.mkdir(parents=True, exist_ok=True)
            df_up_gs.to_parquet(p, index=False, compression="snappy")
            print(f"   🔄 Staged {rel_gs}: {len(df_up_gs):,} rows (was {len(df_rem_gs) if df_rem_gs is not None else 0:,}).")
            modified = True
        else:
            print(f"   ✓ {rel_gs} is up-to-date ({len(df_up_gs):,} rows).")

    return modified


def sync_measurement_player(repo_root: Path, repo_id: str, token: Optional[str], season_year: int, staging_dir: Path, force: bool = False) -> bool:
    """
    Syncs measurement/player/player_season.parquet (rolling season-to-date + season-ending
    player aggregates in the same parquet file) and measurement/player/player_game.parquet
    (per-game + cumulative rolling season-to-date player statistics).
    """
    modified = False

    # 1. Player Season (Rolling active season + Finalized historical seasons)
    rel_path = "measurement/player/player_season.parquet"
    df_remote = load_remote_parquet(repo_id, rel_path, token)
    df_fresh = build_player_season_df(repo_root, season_year=None if force else season_year)
    if not df_fresh.empty or df_remote is None:
        if not force and df_remote is not None and "ixg" in df_remote.columns and "season_status" in df_remote.columns:
            df_hist = df_remote[df_remote["yr_season"] != season_year].copy()
            df_combined = pd.concat([df_hist, df_fresh], ignore_index=True)
        else:
            df_combined = df_fresh if force else build_player_season_df(repo_root)

        df_updated = df_combined.drop_duplicates().reset_index(drop=True)
        if force or has_changed(df_updated, df_remote, ["yr_season"], season_year=None if force else season_year):
            out_path = staging_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df_updated.to_parquet(out_path, index=False, compression="snappy")
            print(f"   🔄 Staged {rel_path}: {len(df_updated):,} rows (was {len(df_remote) if df_remote is not None else 0:,}).")
            modified = True
        else:
            print(f"   ✓ {rel_path} is up-to-date ({len(df_updated):,} rows).")

    # 2. Player Game + Rolling Season-to-Date
    rel_pg = "measurement/player/player_game.parquet"
    df_rem_pg = load_remote_parquet(repo_id, rel_pg, token)
    df_fr_pg = build_player_game_df(repo_root, season_year=None if force else season_year)
    if not df_fr_pg.empty or df_rem_pg is None:
        if not force and df_rem_pg is not None and "ixg" in df_rem_pg.columns and "cum_ixg" in df_rem_pg.columns:
            df_hist_pg = df_rem_pg[df_rem_pg["yr_season"] != season_year].copy()
            df_comb_pg = pd.concat([df_hist_pg, df_fr_pg], ignore_index=True)
        else:
            df_comb_pg = df_fr_pg if force else build_player_game_df(repo_root)

        df_up_pg = df_comb_pg.drop_duplicates(subset=["game_id", "player_id"], keep="last").sort_values(by=["game_id", "player_id"]).reset_index(drop=True)
        if force or has_changed(df_up_pg, df_rem_pg, ["game_id", "player_id"], season_year=None if force else season_year):
            out_pg = staging_dir / rel_pg
            out_pg.parent.mkdir(parents=True, exist_ok=True)
            df_up_pg.to_parquet(out_pg, index=False, compression="snappy")
            print(f"   🔄 Staged {rel_pg}: {len(df_up_pg):,} rows (was {len(df_rem_pg) if df_rem_pg is not None else 0:,}).")
            modified = True
        else:
            print(f"   ✓ {rel_pg} is up-to-date ({len(df_up_pg):,} rows).")

    return modified


def sync_measurement_hits(repo_root: Path, repo_id: str, token: Optional[str], season_year: int, staging_dir: Path) -> bool:
    """Syncs measurement/action/hits/*.parquet metrics."""
    hit_types = [
        ("team_season_hit_kpi", ["yr_season", "team"]),
        ("team_season_spatial_hits", ["yr_season", "team", "x_bin", "y_bin"]),
        ("hit_zone_summary_kpi", []),
        ("hit_zone_summary", []),
        ("hit_goal_window_30s", []),
        ("hit_next_events", []),
        ("hit_sequence_transitions", []),
        ("rink_grid_hit_sequences", []),
        ("tabulated_hit_sequences", [])
    ]
    modified = False

    for ht, preferred_keys in hit_types:
        rel_path = f"measurement/action/hits/{ht}.parquet"
        df_remote = load_remote_parquet(repo_id, rel_path, token)
        df_fresh = build_hits_measurement_df(repo_root, ht, season_year=season_year)
        if df_fresh.empty and df_remote is not None:
            continue

        if df_remote is not None and "yr_season" in df_remote.columns:
            df_hist = df_remote[df_remote["yr_season"] != season_year].copy()
            df_combined = pd.concat([df_hist, df_fresh], ignore_index=True)
        elif df_remote is not None:
            df_combined = df_fresh if not df_fresh.empty else df_remote
        else:
            df_combined = build_hits_measurement_df(repo_root, ht)

        dedup_keys = [c for c in preferred_keys if c in df_combined.columns]
        if dedup_keys:
            df_updated = df_combined.drop_duplicates(subset=dedup_keys, keep="last").reset_index(drop=True)
        else:
            df_updated = df_combined.drop_duplicates().reset_index(drop=True)

        if has_changed(df_updated, df_remote, dedup_keys, season_year=season_year):
            out_path = staging_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df_updated.to_parquet(out_path, index=False, compression="snappy")
            print(f"   🔄 Staged {rel_path}: {len(df_updated):,} rows.")
            modified = True
        else:
            print(f"   ✓ {rel_path} is up-to-date ({len(df_updated):,} rows).")

    return modified


def sync_measurement_shifts(repo_root: Path, repo_id: str, token: Optional[str], season_year: int, staging_dir: Path) -> bool:
    """Syncs measurement/action/shifts/*.parquet metrics (line shifts, L5 line trends, L5 team trends)."""
    shift_tables = [
        ("game_line_shifts", ["season", "gameid", "team", "unit_type", "unit_key"]),
        ("game_line_trends_l5", ["season", "gameid", "team", "unit_type", "unit_key"]),
        ("team_game_trends_l5", ["season", "gameid", "team"]),
    ]
    modified = False
    shifts_dir = repo_root / "latest" / "action" / "shifts"

    for tbl_name, dedup_cols in shift_tables:
        rel_path = f"measurement/action/shifts/{tbl_name}.parquet"
        df_remote = load_remote_parquet(repo_id, rel_path, token)

        season_csv = shifts_dir / f"{season_year}_{tbl_name}.csv"
        master_csv = shifts_dir / f"{tbl_name}.csv"
        if season_csv.exists():
            df_fresh = pd.read_csv(season_csv, low_memory=False)
        elif master_csv.exists():
            df_fresh = pd.read_csv(master_csv, low_memory=False)
        else:
            df_fresh = pd.DataFrame()

        if df_fresh.empty and df_remote is not None:
            continue

        if df_remote is not None and "season" in df_remote.columns and not df_fresh.empty:
            df_hist = df_remote[df_remote["season"] != season_year].copy()
            df_combined = pd.concat([df_hist, df_fresh], ignore_index=True)
        elif not df_fresh.empty:
            df_combined = df_fresh
        else:
            continue

        valid_keys = [c for c in dedup_cols if c in df_combined.columns]
        if valid_keys:
            df_updated = df_combined.drop_duplicates(subset=valid_keys, keep="last").reset_index(drop=True)
        else:
            df_updated = df_combined.drop_duplicates().reset_index(drop=True)

        if has_changed(df_updated, df_remote, valid_keys, season_year=season_year):
            out_path = staging_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df_updated.to_parquet(out_path, index=False, compression="snappy")
            print(f"   🔄 Staged {rel_path}: {len(df_updated):,} rows.")
            modified = True
        else:
            print(f"   ✓ {rel_path} is up-to-date ({len(df_updated):,} rows).")

    return modified


def sync_scoreboard_recaps(
    repo_root: Path,
    repo_id: str,
    token: Optional[str],
    season_year: int,
    staging_dir: Path,
    recent_dates_limit: int = 2,
) -> bool:
    """Ensures recent completed game dates in the active season have Scoreboard recaps generated and staged."""
    try:
        sys.path.insert(0, str(repo_root))
        from dev.agent.scoreboard.agent import ScoreboardAgent
        from dev.agent.scoreboard.dataset_store import HFScoreboardStore

        box_file = repo_root / "latest" / "box" / f"{season_year}_box.csv"
        if not box_file.exists():
            return False

        df_box = pd.read_csv(box_file, usecols=lambda c: c in ("date", "gameDate"))
        d_col = "date" if "date" in df_box.columns else ("gameDate" if "gameDate" in df_box.columns else None)
        if not d_col or df_box.empty:
            return False

        avail_dates = sorted(pd.to_datetime(df_box[d_col]).dt.strftime("%Y-%m-%d").dropna().unique().tolist(), reverse=True)
        target_dates = avail_dates[:recent_dates_limit]

        store = HFScoreboardStore(repo_id=repo_id, token=token)
        store._load_tables(refresh_remote=True)

        agent = ScoreboardAgent()
        can_generate_llm = False
        if agent.narrator._can_use_zerogpu():
            can_generate_llm = True
        else:
            is_lm_up, _, _ = agent.narrator.check_lm_studio_health()
            if is_lm_up:
                can_generate_llm = True

        if can_generate_llm:
            missing_dates = [d for d in target_dates if not store.is_slate_llm_generated(d)]
        else:
            missing_dates = [d for d in target_dates if not store.is_slate_recorded(d)]

        modified = False
        for d_str in sorted(missing_dates):
            print(f"   ✍️ Generating Scoreboard recaps for {d_str}...")
            slate = agent.analyze_slate(d_str)
            if slate and slate.games:
                store.record_slate(slate, generator_model=getattr(slate, "generator_model", "algorithmic-guardrails-v1"))
                store.record_games(slate.games, generator_model=getattr(slate.games[0], "generator_model", "algorithmic-guardrails-v1"))
                modified = True

        if not modified and all(store.is_slate_llm_generated(d) for d in target_dates):
            print(f"   ✓ insights/scoreboard LLM recaps are up-to-date for recent dates ({', '.join(target_dates)}).")

        if modified:
            recap_stage_dir = staging_dir / "insights" / "scoreboard"
            recap_stage_dir.mkdir(parents=True, exist_ok=True)
            if store.game_recaps_file.exists():
                shutil.copy2(store.game_recaps_file, recap_stage_dir / "game_recaps.parquet")
            if store.slate_recaps_file.exists():
                shutil.copy2(store.slate_recaps_file, recap_stage_dir / "slate_recaps.parquet")
            print("   🔄 Staged insights/scoreboard/game_recaps.parquet and slate_recaps.parquet.")

        # If any recent date is still only algorithmic (e.g. synced from a headless CI runner),
        # wake/restart the Hugging Face Space so its ZeroGPU worker upgrades it to Tier-2 LLM recaps.
        non_llm_dates = [d for d in target_dates if not store.is_slate_llm_generated(d)]
        if non_llm_dates:
            try:
                if token and HF_HUB_AVAILABLE:
                    HfApi(token=token).restart_space("jjerlee/rinklytics")
                    print(f"   ⚡ Triggered Hugging Face Space (jjerlee/rinklytics) ZeroGPU worker to upgrade {', '.join(non_llm_dates)} to LLM recaps.")
            except Exception as wake_err:
                print(f"   ℹ️ Note waking Hugging Face Space: {wake_err}")

        return modified
    except Exception as e:
        print(f"   ⚠️ Warning in sync_scoreboard_recaps: {e}")
        return False



# -----------------------------------------------------------------------------
# MAIN SYNC DISPATCHER
# -----------------------------------------------------------------------------

def run_sync(
    repo_id: str,
    target: str = "all",
    season_year: Optional[int] = None,
    token: Optional[str] = None,
    skip_upload: bool = False,
    force: bool = False,
    custom_message: Optional[str] = None,
) -> bool:
    """Executes target domain synchronization and commits updates to Hugging Face."""
    repo_root = get_repo_root()
    active_year = season_year or resolve_active_season_year(repo_root)
    resolved_token = token or get_hf_token()

    staging_dir = repo_root / "deploy" / "huggingface" / "staging_sync"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("🏒 NHL ANALYTICS — HUGGING FACE INCREMENTAL DATASET SYNC")
    print("=" * 70)
    print(f"Repository Root: {repo_root}")
    print(f"Target HF Dataset: {repo_id}")
    print(f"Active Season Year: {active_year}")
    print(f"Sync Target Domain: {target.upper()}")
    print(f"Skip Upload: {skip_upload} | Force: {force}")
    print("-" * 70)

    any_modified = False

    # 1. Target: Box & Plays (09:00 UTC pipeline)
    if target in ("box", "all"):
        print("\n📦 Processing Box Scores & Play-by-Play Event Panels...")
        any_modified |= sync_games(repo_root, repo_id, resolved_token, active_year, staging_dir, force=force)
        any_modified |= sync_team_box(repo_root, repo_id, resolved_token, active_year, staging_dir, force=force)
        any_modified |= sync_player_box(repo_root, repo_id, resolved_token, active_year, staging_dir, force=force)
        any_modified |= sync_playbyplay(repo_root, repo_id, resolved_token, active_year, staging_dir)
        any_modified |= sync_playbyplay_player(repo_root, repo_id, resolved_token, active_year, staging_dir)
        any_modified |= sync_shifts(repo_root, repo_id, resolved_token, active_year, staging_dir)
        any_modified |= sync_measurement_box(repo_root, repo_id, resolved_token, active_year, staging_dir, force=force)
        any_modified |= sync_measurement_hits(repo_root, repo_id, resolved_token, active_year, staging_dir)
        any_modified |= sync_measurement_shifts(repo_root, repo_id, resolved_token, active_year, staging_dir)
        any_modified |= sync_scoreboard_recaps(repo_root, repo_id, resolved_token, active_year, staging_dir)

    # 1b. Target: Shifts standalone
    if target == "shifts":
        print("\n⏱️ Processing Play-by-Play Shift Panels...")
        any_modified |= sync_shifts(repo_root, repo_id, resolved_token, active_year, staging_dir)
        any_modified |= sync_measurement_shifts(repo_root, repo_id, resolved_token, active_year, staging_dir)

    # 1c. Target: Scoreboard Recaps standalone
    if target == "recaps":
        print("\n📰 Processing Scoreboard Recaps...")
        any_modified |= sync_scoreboard_recaps(repo_root, repo_id, resolved_token, active_year, staging_dir)

    # 2. Target: Odds (20:00 UTC pipeline)
    if target in ("odds", "all"):
        print("\n🎲 Processing Closing Betting Odds Panels...")
        any_modified |= sync_odds(repo_root, repo_id, resolved_token, active_year, staging_dir)

    # 3. Target: Player Season & Game Rolling Aggregates
    if target in ("player", "box", "all"):
        print("\n👤 Processing Player Game & Season (Rolling + Season-Ending) Aggregates...")
        any_modified |= sync_measurement_player(repo_root, repo_id, resolved_token, active_year, staging_dir, force=force)

    # Check staging results
    staged_files = list(staging_dir.rglob("*.parquet"))
    if not staged_files and not force:
        print("\n" + "=" * 70)
        print("✅ All target datasets are completely up to date with remote.")
        print("   No changes detected. Skipping upload to preserve commit history.")
        print("=" * 70)
        shutil.rmtree(staging_dir, ignore_errors=True)
        return False

    print("\n" + "-" * 70)
    print(f"📊 Summary of Staged Updates: {len(staged_files)} Parquet tables modified.")
    for f in staged_files:
        rel = f.relative_to(staging_dir)
        print(f"   • {rel} ({f.stat().st_size / (1024 * 1024):.2f} MB)")
    print("-" * 70)

    if skip_upload:
        print("\n✅ Skipping upload as requested (--skip-upload). Staged files remain in:")
        print(f"   {staging_dir}")
        return True

    if not resolved_token:
        print("\n❌ Error: Hugging Face access token not found.")
        print("   Set HF_TOKEN environment variable or add rinklytics=<token> in .env.")
        sys.exit(1)

    # Upload staged files in a single atomic commit
    commit_msg = custom_message or f"Automated sync: updated {target} datasets (season {active_year})"
    print(f"\n🌐 Uploading staged updates to {repo_id}...")
    api = HfApi(token=resolved_token)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(staging_dir),
        repo_type="dataset",
        commit_message=commit_msg,
    )

    print("\n" + "=" * 70)
    print(f"🎉 SUCCESS! Dataset update published successfully to:")
    print(f"   👉 https://huggingface.co/datasets/{repo_id}")
    print("=" * 70)

    shutil.rmtree(staging_dir, ignore_errors=True)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synchronize daily NHL Analytics updates to Hugging Face Datasets.")
    parser.add_argument("--repo", default="jjerlee/nhl-analytics", help="Hugging Face Dataset repo ID (default: jjerlee/nhl-analytics)")
    parser.add_argument("--target", choices=["all", "box", "shifts", "recaps", "odds", "player"], default="all", help="Target domain to sync (default: all)")
    parser.add_argument("--season", type=int, default=None, help="Specific season year to process (e.g. 2026)")
    parser.add_argument("--token", default=None, help="Hugging Face access token (defaults to HF_TOKEN env variable)")
    parser.add_argument("--skip-upload", action="store_true", help="Stage updates locally without uploading")
    parser.add_argument("--force", action="store_true", help="Force commit even if no differences are detected")
    parser.add_argument("--message", default=None, help="Custom commit message")

    args = parser.parse_args()
    run_sync(
        repo_id=args.repo,
        target=args.target,
        season_year=args.season,
        token=args.token,
        skip_upload=args.skip_upload,
        force=args.force,
        custom_message=args.message,
    )
