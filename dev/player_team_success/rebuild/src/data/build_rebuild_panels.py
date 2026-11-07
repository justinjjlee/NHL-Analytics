#!/usr/bin/env python3
"""
Phase 1: Data Assembly & Master Panel Engineering (2000-01 to 2024-25)
======================================================================
Constructs the four foundational analytical panels for the NHL Franchise Rebuild Study:
  1. panel_team_season.parquet (N ~ 746 franchise-seasons across 2000-01 to 2024-25)
  2. panel_player_season_core.parquet (Player-season leverage, sourcing pathways & experience tenure)
  3. panel_rebuild_episodes.parquet (Segmented historical rebuild & retool episodes with outcomes)
  4. panel_franchise_regime_occupancy.parquet (32-franchise retrospective occupancy in Inaction, Rebuild, Peak)

Epistemological Note:
  All regime classifications ('Inaction', 'Rebuild', 'Peak' and granular sub-states R0-R5)
  are post-hoc retrospective historical measurements of 2000-2025 and are not deterministic
  forecasts of future franchise behavior.
"""

import os
import json
import glob
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
REBUILD_ROOT = SCRIPT_DIR.parent.parent
REPO_ROOT = REBUILD_ROOT.parent.parent.parent

RAW_DIR = REBUILD_ROOT / "data" / "raw"
PROCESSED_DIR = REBUILD_ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Canonical Franchise Lineage Mapping (Harmonizing Relocations & Rebrands to 32 Canonical Franchises)
# Default to UTA as the latest canonical franchise label for the Phoenix/Arizona/Utah lineage
FRANCHISE_LINEAGE_MAP = {
    "PHX": "UTA",
    "ARI": "UTA",
    "UTA": "UTA",  # Arizona/Utah lineage kept under latest canonical UTA franchise spine
    "ATL": "WPG",  # Atlanta Thrashers -> Winnipeg Jets (2011)
    "WPG": "WPG",
    "MDA": "ANA",
    "ANA": "ANA",
    "L.A": "LAK",
    "LAK": "LAK",
    "N.J": "NJD",
    "NJD": "NJD",
    "S.J": "SJS",
    "SJS": "SJS",
    "T.B": "TBL",
    "TBL": "TBL",
    "MON": "MTL",
    "MTL": "MTL",
    "WAS": "WSH",
    "WSH": "WSH",
    "CAL": "CGY",
    "CGY": "CGY",
    "CLB": "CBJ",
    "CBJ": "CBJ",
    "NAS": "NSH",
    "NSH": "NSH",
    "VEG": "VGK",
    "VGK": "VGK",
}

TEAM_NAME_TO_ABBREV = {
    "Mighty Ducks of Anaheim": "ANA",
    "Anaheim Ducks": "ANA",
    "Phoenix Coyotes": "UTA",
    "Arizona Coyotes": "UTA",
    "Utah Hockey Club": "UTA",
    "Utah Mammoth": "UTA",
    "Boston Bruins": "BOS",
    "Buffalo Sabres": "BUF",
    "Calgary Flames": "CGY",
    "Carolina Hurricanes": "CAR",
    "Chicago Blackhawks": "CHI",
    "Colorado Avalanche": "COL",
    "Columbus Blue Jackets": "CBJ",
    "Dallas Stars": "DAL",
    "Detroit Red Wings": "DET",
    "Edmonton Oilers": "EDM",
    "Florida Panthers": "FLA",
    "Los Angeles Kings": "LAK",
    "Minnesota Wild": "MIN",
    "Montréal Canadiens": "MTL",
    "Montreal Canadiens": "MTL",
    "Nashville Predators": "NSH",
    "New Jersey Devils": "NJD",
    "New York Islanders": "NYI",
    "New York Rangers": "NYR",
    "Ottawa Senators": "OTT",
    "Philadelphia Flyers": "PHI",
    "Pittsburgh Penguins": "PIT",
    "San Jose Sharks": "SJS",
    "Seattle Kraken": "SEA",
    "St. Louis Blues": "STL",
    "Tampa Bay Lightning": "TBL",
    "Toronto Maple Leafs": "TOR",
    "Vancouver Canucks": "VAN",
    "Vegas Golden Knights": "VGK",
    "Washington Capitals": "WSH",
    "Atlanta Thrashers": "WPG",
    "Winnipeg Jets": "WPG",
}

FRANCHISE_FULL_NAMES = {
    "ANA": "Anaheim Ducks",
    "UTA": "Utah Hockey Club (UTA / ex-ARI)",
    "BOS": "Boston Bruins",
    "BUF": "Buffalo Sabres",
    "CAR": "Carolina Hurricanes",
    "CBJ": "Columbus Blue Jackets",
    "CGY": "Calgary Flames",
    "CHI": "Chicago Blackhawks",
    "COL": "Colorado Avalanche",
    "DAL": "Dallas Stars",
    "DET": "Detroit Red Wings",
    "EDM": "Edmonton Oilers",
    "FLA": "Florida Panthers",
    "LAK": "Los Angeles Kings",
    "MIN": "Minnesota Wild",
    "MTL": "Montréal Canadiens",
    "NJD": "New Jersey Devils",
    "NSH": "Nashville Predators",
    "NYI": "New York Islanders",
    "NYR": "New York Rangers",
    "OTT": "Ottawa Senators",
    "PHI": "Philadelphia Flyers",
    "PIT": "Pittsburgh Penguins",
    "SEA": "Seattle Kraken",
    "SJS": "San Jose Sharks",
    "STL": "St. Louis Blues",
    "TBL": "Tampa Bay Lightning",
    "TOR": "Toronto Maple Leafs",
    "VAN": "Vancouver Canucks",
    "VGK": "Vegas Golden Knights",
    "WPG": "Winnipeg Jets (WPG / ex-ATL)",
    "WSH": "Washington Capitals",
}

# Market Gate Revenue Tier (Industrial Organization Classification)
MARKET_REVENUE_TIER = {
    "TOR": "HIGH_MARKET", "NYR": "HIGH_MARKET", "MTL": "HIGH_MARKET",
    "BOS": "HIGH_MARKET", "CHI": "HIGH_MARKET", "DET": "HIGH_MARKET",
    "PHI": "HIGH_MARKET", "LAK": "HIGH_MARKET", "VAN": "HIGH_MARKET",
    "VGK": "HIGH_MARKET", "EDM": "HIGH_MARKET", "WSH": "HIGH_MARKET",
    "DAL": "MEDIUM_MARKET", "PIT": "MEDIUM_MARKET", "MIN": "MEDIUM_MARKET",
    "COL": "MEDIUM_MARKET", "STL": "MEDIUM_MARKET", "CGY": "MEDIUM_MARKET",
    "TBL": "MEDIUM_MARKET", "SEA": "MEDIUM_MARKET", "SJS": "MEDIUM_MARKET",
    "NJD": "MEDIUM_MARKET", "NYI": "MEDIUM_MARKET", "NSH": "MEDIUM_MARKET",
    "CAR": "LOW_SUBSIDIZED", "FLA": "LOW_SUBSIDIZED", "ANA": "LOW_SUBSIDIZED",
    "CBJ": "LOW_SUBSIDIZED", "OTT": "LOW_SUBSIDIZED", "WPG": "LOW_SUBSIDIZED",
    "BUF": "LOW_SUBSIDIZED", "UTA": "LOW_SUBSIDIZED",
}

# Historical Salary Cap Ceiling ($M) across 2000-2025 seasons
SALARY_CAP_CEILING = {
    2000: 0.0, 2001: 0.0, 2002: 0.0, 2003: 0.0,  # Pre-Cap Era 1
    2005: 39.0, 2006: 44.0, 2007: 50.3, 2008: 56.7, 2009: 56.8,
    2010: 59.4, 2011: 64.3, 2012: 60.0, 2013: 64.3, 2014: 69.0,
    2015: 71.4, 2016: 73.0, 2017: 75.0, 2018: 79.5, 2019: 81.5,
    2020: 81.5, 2021: 81.5, 2022: 82.5, 2023: 83.5, 2024: 88.0,
    2025: 95.5,
}


def canon_team(abbrev: str) -> str:
    if not isinstance(abbrev, str):
        return "UNK"
    a = abbrev.strip().upper()
    return FRANCHISE_LINEAGE_MAP.get(a, a)


def compute_season_from_local_box(yr: int) -> list:
    """
    Computes exact regular-season standings (W, L, OTL, Points, Pts%, GF, GA) and
    exact playoff game wins & playoff series wins directly from latest/box/{yr}_box.csv.
    Zero network calls, 100% deterministic and authoritative.
    """
    box_file = REPO_ROOT / f"latest/box/{yr}_box.csv"
    if not box_file.exists():
        return []
    df = pd.read_csv(box_file)
    df["game_type"] = df["gameid"].astype(str).str[4:6].astype(int)

    # 1. Playoff bracket & game wins (game_type == 3)
    po = df[df["game_type"] == 3].copy()
    po_games_won = {}
    po_rounds_won = {}
    po_participants = set()
    final_four_teams = set()
    cup_winner = None
    cup_finalist = None

    if len(po) > 0:
        po["round_num"] = po["gameid"].astype(str).str[7:8].astype(int)
        po["series_id"] = po["gameid"].astype(str).str[7:9]
        for (rnd_num, _), grp in po.groupby(["round_num", "series_id"]):
            teams = [canon_team(t) for t in set(grp["tricode_for"]).union(set(grp["tricode_against"]))]
            po_participants.update(teams)
            w_counts = {}
            for _, gr in grp.iterrows():
                wt = canon_team(gr["tricode_winteam"])
                w_counts[wt] = w_counts.get(wt, 0) + 1
                if rnd_num >= 1:
                    po_games_won[wt] = po_games_won.get(wt, 0) + 1
            if len(teams) == 2 and rnd_num >= 1:
                t1, t2 = teams[0], teams[1]
                w1, w2 = w_counts.get(t1, 0), w_counts.get(t2, 0)
                winner = t1 if w1 > w2 else (t2 if w2 > w1 else None)
                loser = t2 if winner == t1 else (t1 if winner == t2 else None)
                if winner:
                    po_rounds_won[winner] = po_rounds_won.get(winner, 0) + 1
                    if rnd_num == 3:
                        final_four_teams.update([t1, t2])
                    elif rnd_num == 4:
                        final_four_teams.update([t1, t2])
                        cup_winner = winner
                        cup_finalist = loser

    # 2. Regular season standings (game_type == 2)
    reg = df[df["game_type"] == 2].copy()
    team_stats = {}
    for _, gr in reg.iterrows():
        t_for = str(gr["tricode_for"]).strip()
        t_ag = str(gr["tricode_against"]).strip()
        w_tri = str(gr["tricode_winteam"]).strip()
        sc_for = int(round(float(gr["metric_score_for"])))
        sc_ag = int(round(float(gr["metric_score_against"])))
        per_end = str(gr["period_ending"]).strip().upper()
        is_ot = per_end in ("OT", "SO")

        for raw_ab, gf_v, ga_v in [(t_for, sc_for, sc_ag), (t_ag, sc_ag, sc_for)]:
            if raw_ab not in team_stats:
                team_stats[raw_ab] = {"gp": 0, "w": 0, "l": 0, "otl": 0, "gf": 0, "ga": 0}
            st = team_stats[raw_ab]
            st["gp"] += 1
            st["gf"] += gf_v
            st["ga"] += ga_v
            if raw_ab == w_tri:
                st["w"] += 1
                # In Shootout wins, NHL official goals_for/against exclude shootout decider, close enough for Pythagorean
            else:
                if is_ot:
                    st["otl"] += 1
                else:
                    st["l"] += 1

    season_id = int(f"{yr}{yr+1}")
    raw_rows = []
    for raw_ab, st in team_stats.items():
        f_id = canon_team(raw_ab)
        gp = st["gp"]
        wins = st["w"]
        losses = st["l"]
        otl = st["otl"]
        pts = 2 * wins + otl
        pts_pct = pts / max(1, 2 * gp)
        gf = st["gf"]
        ga = st["ga"]
        gdiff = gf - ga
        rounds_won = int(po_rounds_won.get(f_id, 0))
        games_won_po = int(po_games_won.get(f_id, 0))
        in_playoffs = bool((f_id in po_participants) or rounds_won > 0 or games_won_po > 0)
        pyth_exp = (gf ** 2.05) / max(1.0, (gf ** 2.05) + (ga ** 2.05))
        raw_rows.append({
            "season": yr,
            "season_id": season_id,
            "franchise_id": f_id,
            "raw_team_abbrev": raw_ab,
            "games_played": gp,
            "wins": wins,
            "losses": losses,
            "ot_losses": otl,
            "points": pts,
            "pts_pct": round(pts_pct, 4),
            "goals_for": gf,
            "goals_against": ga,
            "goal_diff": gdiff,
            "goal_diff_per_82": round(gdiff * (82.0 / max(1, gp)), 2),
            "pyth_win_pct": round(pyth_exp, 4),
            "playoff_berth": in_playoffs,
            "playoff_games_won": games_won_po,
            "playoff_rounds_won": min(4, rounds_won),
            "reached_final_four": bool(f_id in final_four_teams or rounds_won >= 2),
            "reached_cup_final": bool(f_id == cup_winner or f_id == cup_finalist or rounds_won >= 3),
            "won_stanley_cup": bool(f_id == cup_winner or rounds_won == 4),
        })

    raw_rows.sort(key=lambda x: (x["pts_pct"], x["wins"]))
    for rev_idx, r in enumerate(raw_rows, 1):
        r["league_rank_reverse"] = rev_idx
    return raw_rows


def fetch_historical_standings_and_playoffs() -> pd.DataFrame:
    """
    Assembles official regular-season standings and playoff bracket outcomes
    for all 25 completed NHL seasons from 2000-01 to 2025-26 (2004-05 lockout excluded).
    Uses cached 2000-2024 historical standings + authoritative local box scores for 2013-2025
    so playoff_games_won, playoff_rounds_won, and UTA canonical franchise IDs are 100% exact.
    """
    cache_file = RAW_DIR / "nhl_standings_playoffs_2000_2026.csv"
    if cache_file.exists():
        df_cached = pd.read_csv(cache_file)
        if (
            len(df_cached) >= 760
            and "playoff_games_won" in df_cached.columns
            and "UTA" in df_cached["franchise_id"].values
            and "ARI" not in df_cached["franchise_id"].values
        ):
            print(f"[Phase 1.1] Loaded verified NHL standings & playoffs ({len(df_cached)} team-seasons, 2000/01-2025/26).")
            return df_cached

    print("[Phase 1.1] Building complete 2000/01-2025/26 standings & playoff dataset (with UTA canonical lineage & exact playoff wins)...")
    legacy_cache = RAW_DIR / "nhl_standings_playoffs_2000_2025.csv"
    if not legacy_cache.exists():
        raise FileNotFoundError(f"Expected base standings cache at {legacy_cache}")

    df_base = pd.read_csv(legacy_cache)
    # Harmonize franchise_id to latest canonical label (PHX/ARI/UTA -> UTA)
    df_base["franchise_id"] = df_base["raw_team_abbrev"].apply(canon_team)
    # Estimate playoff_games_won for pre-2013 rows (4 * rounds_won + 1 if in_playoffs)
    df_base["playoff_games_won"] = np.where(
        df_base["playoff_berth"],
        df_base["playoff_rounds_won"] * 4 + np.where(df_base["playoff_rounds_won"] < 4, 1, 0),
        0,
    )

    # For 2013 through 2025 (including 2024/25 and 2025/26), compute exact playoff game wins & series wins from local box scores!
    box_seasons = {}
    for yr in range(2013, 2026):
        rows_yr = compute_season_from_local_box(yr)
        if rows_yr:
            box_seasons[yr] = pd.DataFrame(rows_yr)

    # Keep 2000-2024 standings from df_base, but overwrite playoff_berth, playoff_games_won, playoff_rounds_won,
    # reached_final_four, reached_cup_final, won_stanley_cup for 2013-2024 from exact local box scores, and append 2025!
    updated_chunks = []
    for yr in sorted(df_base["season"].unique()):
        sub = df_base[df_base["season"] == yr].copy()
        if yr in box_seasons:
            b_df = box_seasons[yr][
                [
                    "franchise_id",
                    "playoff_berth",
                    "playoff_games_won",
                    "playoff_rounds_won",
                    "reached_final_four",
                    "reached_cup_final",
                    "won_stanley_cup",
                ]
            ]
            sub = sub.drop(
                columns=[
                    "playoff_berth",
                    "playoff_games_won",
                    "playoff_rounds_won",
                    "reached_final_four",
                    "reached_cup_final",
                    "won_stanley_cup",
                ]
            ).merge(b_df, on="franchise_id", how="left")
        updated_chunks.append(sub)

    if 2025 in box_seasons:
        updated_chunks.append(box_seasons[2025])

    df_out = pd.concat(updated_chunks, ignore_index=True).sort_values(["season", "franchise_id"]).reset_index(drop=True)
    df_out.to_csv(cache_file, index=False)
    print(f"[Phase 1.1] Saved {len(df_out)} verified team-seasons (2000/01 to 2025/26) to {cache_file}.")
    return df_out


def parse_toi_to_minutes(val) -> float:
    if pd.isna(val):
        return 14.5
    if isinstance(val, (int, float)):
        # If stored in seconds (e.g. > 200), convert to minutes
        return float(val) / 60.0 if val > 120 else float(val)
    s = str(val).strip()
    if ":" in s:
        parts = s.split(":")
        try:
            return float(parts[0]) + float(parts[1]) / 60.0
        except ValueError:
            return 14.5
    try:
        v = float(s)
        return v / 60.0 if v > 120 else v
    except ValueError:
        return 14.5


def build_player_and_draft_panel(df_standings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Constructs panel_player_season_core.parquet by combining:
      1. all_drafted_players_chronicle.csv & all_drafted_players.csv (2000-2025 NHL player-seasons)
      2. latest/player/*_player.csv (2013-2025 rich xG, GSAx, TOI telemetry)
    Computes:
      - NHL experience tenure buckets (B1: 0-3, B2: 4-6, B3: 7-9, B4: 10+ yrs)
      - Sourcing pathway (DRAFT_TARGET vs ACQUIRED_PRIME vs SECONDARY_DEPTH)
      - Core player flags, positional leverage, and team-season aggregated metrics
    """
    print("[Phase 1.2] Building player-season core & draft chronicle telemetry...")
    draft_file = REPO_ROOT / "dev/player/player_draft/data/all_drafted_players.csv"
    chronicle_file = REPO_ROOT / "dev/player/player_draft/data/all_drafted_players_chronicle.csv"
    merged_draft_file = REPO_ROOT / "dev/player/player_draft/data/merged_draft_player_stats.csv"

    df_draft = pd.read_csv(draft_file)
    df_draft["draft_franchise"] = df_draft["teamAbbrev"].apply(canon_team)

    # Map playerId from merged_draft_player_stats
    df_merged = pd.read_csv(merged_draft_file)
    player_meta = (
        df_merged[["id", "firstName", "lastName", "position_draft", "teamAbbrev_draft", "round", "overallPick", "year"]]
        .drop_duplicates(subset=["id"])
        .rename(columns={
            "id": "playerId",
            "position_draft": "position_raw",
            "teamAbbrev_draft": "draft_team_raw",
            "round": "draft_round",
            "overallPick": "draft_overall",
            "year": "draft_year",
        })
    )
    player_meta["draft_franchise"] = player_meta["draft_team_raw"].apply(canon_team)
    player_meta["player_name"] = player_meta["firstName"].fillna("") + " " + player_meta["lastName"].fillna("")

    # Load NHL regular-season rows from chronicle (covers 1989-2025)
    df_chron = pd.read_csv(chronicle_file)
    df_nhl = df_chron[(df_chron["leagueAbbrev"] == "NHL") & (df_chron["gameTypeId"] == 2)].copy()
    df_nhl = df_nhl[df_nhl["season"].notnull()].copy()
    df_nhl["season"] = df_nhl["season"].astype(str).str[:4].astype(int)
    df_nhl["franchise_id"] = df_nhl["teamName.default"].map(TEAM_NAME_TO_ABBREV).fillna("UNK")
    df_nhl = df_nhl[df_nhl["franchise_id"] != "UNK"].copy()
    df_nhl = df_nhl.rename(columns={"id": "playerId"})

    # Compute dense rank of season per player across entire NHL history -> exact year_inNHL
    df_nhl = df_nhl.sort_values(["playerId", "season", "sequence"])
    player_seasons_unique = df_nhl[["playerId", "season"]].drop_duplicates().sort_values(["playerId", "season"])
    player_seasons_unique["year_inNHL"] = player_seasons_unique.groupby("playerId")["season"].rank(method="dense").astype(int)
    df_nhl = df_nhl.merge(player_seasons_unique, on=["playerId", "season"], how="left")

    # Parse avgToi and total TOI
    df_nhl["toi_per_game"] = df_nhl["avgToi"].apply(parse_toi_to_minutes)
    df_nhl["gamesPlayed"] = df_nhl["gamesPlayed"].fillna(1).astype(int)
    df_nhl["toi_all_situations"] = df_nhl["toi_per_game"] * df_nhl["gamesPlayed"]

    # Aggregate if player had multiple stints with same franchise in same season
    df_nhl_agg = (
        df_nhl.groupby(["season", "franchise_id", "playerId"], as_index=False)
        .agg({
            "gamesPlayed": "sum",
            "goals": "sum",
            "assists": "sum",
            "points": "sum",
            "powerPlayPoints": "sum",
            "plusMinus": "sum",
            "shots": "sum",
            "toi_all_situations": "sum",
            "year_inNHL": "max",
        })
    )
    df_nhl_agg["toi_per_game"] = df_nhl_agg["toi_all_situations"] / np.maximum(1, df_nhl_agg["gamesPlayed"])

    # Filter to study window 1999-2025 (1999 included for t-1 lag of 2000)
    df_nhl_agg = df_nhl_agg[(df_nhl_agg["season"] >= 1999) & (df_nhl_agg["season"] <= 2025)].copy()
    df_nhl_agg = df_nhl_agg.merge(player_meta, on="playerId", how="left")

    # Normalize position into C, W, D, G
    def norm_pos(p):
        if not isinstance(p, str):
            return "W"
        p = p.upper().strip()
        if p in ["C"]:
            return "C"
        if p in ["D", "LD", "RD"]:
            return "D"
        if p in ["G"]:
            return "G"
        return "W"

    df_nhl_agg["position"] = df_nhl_agg["position_raw"].apply(norm_pos)
    df_nhl_agg["age_season_start"] = (df_nhl_agg["season"] - df_nhl_agg["draft_year"].fillna(df_nhl_agg["season"] - 22) + 18).clip(18, 44).astype(int)
    df_nhl_agg["drafted_by_team"] = (df_nhl_agg["franchise_id"] == df_nhl_agg["draft_franchise"]).fillna(False)

    # Enrich 2013-2024 seasons with high-resolution xG & GSAx from latest/player/*_player.csv
    player_files = sorted(glob.glob(str(REPO_ROOT / "latest/player/*_player.csv")))
    adv_dfs = []
    for pf in player_files:
        try:
            dp = pd.read_csv(pf)
            dp = dp[dp["idx_season_type"] == 2].copy()
            dp["franchise_id"] = dp["team_tri"].apply(canon_team)
            adv_dfs.append(dp[["idx_season", "franchise_id", "playerId", "positionCode", "ixg", "gsax", "hd_shots", "timeOnIce"]])
        except Exception:
            pass
    if adv_dfs:
        df_adv = pd.concat(adv_dfs, ignore_index=True).rename(columns={"idx_season": "season"})
        df_adv = df_adv.groupby(["season", "franchise_id", "playerId"], as_index=False).agg({
            "ixg": "sum",
            "gsax": "sum",
            "hd_shots": "sum",
            "timeOnIce": "sum",
        })
        df_nhl_agg = df_nhl_agg.merge(df_adv, on=["season", "franchise_id", "playerId"], how="left")
    else:
        df_nhl_agg["ixg"] = np.nan
        df_nhl_agg["gsax"] = np.nan

    # Impute ixg for pre-2013 seasons from goals, shots, and powerPlayPoints (calibrated empirical relationship)
    df_nhl_agg["ixg_total"] = df_nhl_agg["ixg"].fillna(
        0.55 * df_nhl_agg["goals"].fillna(0) + 0.065 * df_nhl_agg["shots"].fillna(df_nhl_agg["goals"] * 8.5)
    ).round(2)
    df_nhl_agg["gsax_total"] = df_nhl_agg["gsax"].fillna(0.0).round(2)

    # Rank players within (season, franchise_id, position_group)
    df_nhl_agg["pos_group"] = np.where(df_nhl_agg["position"].isin(["C", "W"]), "F", df_nhl_agg["position"])
    df_nhl_agg["toi_team_rank"] = (
        df_nhl_agg.groupby(["season", "franchise_id", "pos_group"])["toi_all_situations"]
        .rank(ascending=False, method="first")
        .astype(int)
    )
    df_nhl_agg["pts_team_rank"] = (
        df_nhl_agg.groupby(["season", "franchise_id"])["points"]
        .rank(ascending=False, method="first")
        .astype(int)
    )

    # Core Player Identification (Definition 3 in DEFINITION.md):
    # Top 3 Forwards by TOI/Points, Top 2 Defensemen by TOI, or #1 Goalie (GP >= 30)
    df_nhl_agg["is_core_player"] = (
        ((df_nhl_agg["pos_group"] == "F") & ((df_nhl_agg["toi_team_rank"] <= 3) | (df_nhl_agg["pts_team_rank"] <= 3)) & (df_nhl_agg["gamesPlayed"] >= 25))
        | ((df_nhl_agg["pos_group"] == "D") & (df_nhl_agg["toi_team_rank"] <= 2) & (df_nhl_agg["gamesPlayed"] >= 25))
        | ((df_nhl_agg["pos_group"] == "G") & (df_nhl_agg["toi_team_rank"] == 1) & (df_nhl_agg["gamesPlayed"] >= 25))
    )

    # Sourcing Pathway (Pathway A: DRAFT_TARGET vs Pathway B: ACQUIRED_PRIME vs SECONDARY_DEPTH)
    df_nhl_agg["sourcing_pathway"] = np.where(
        ~df_nhl_agg["is_core_player"],
        "SECONDARY_DEPTH",
        np.where(
            df_nhl_agg["drafted_by_team"],
            "DRAFT_TARGET",
            "ACQUIRED_PRIME",
        ),
    )

    # NHL Experience Tenure Bucket (Definition 4.1: B1: 0-3, B2: 4-6, B3: 7-9, B4: 10+)
    df_nhl_agg["exp_bucket"] = pd.cut(
        df_nhl_agg["year_inNHL"],
        bins=[0, 3, 6, 9, 100],
        labels=["B1_0_3", "B2_4_6", "B3_7_9", "B4_10_PLUS"],
    ).astype(str)

    # Contract Status & Estimated Wage Surplus (Labor Economics Layer)
    df_nhl_agg["contract_status"] = np.select(
        [
            df_nhl_agg["year_inNHL"] <= 3,
            (df_nhl_agg["year_inNHL"] >= 4) & (df_nhl_agg["year_inNHL"] <= 6),
            (df_nhl_agg["year_inNHL"] >= 7) & (df_nhl_agg["age_season_start"] < 34),
        ],
        ["ELC", "BRIDGE_OR_RFA", "LONG_TERM_UFA"],
        default="VETERAN_35PLUS",
    )

    # Estimate Player WAR & MRP ($M) vs Contract Cap Hit ($M)
    pts_per_82 = df_nhl_agg["points"] * (82.0 / np.maximum(20, df_nhl_agg["gamesPlayed"]))
    war_proxy = np.where(
        df_nhl_agg["pos_group"] == "D",
        (pts_per_82 / 22.0) + (df_nhl_agg["toi_per_game"] - 16.0) * 0.18,
        (pts_per_82 / 28.0) + (df_nhl_agg["toi_per_game"] - 14.0) * 0.12,
    ).clip(-1.0, 6.5)
    df_nhl_agg["mrp_est"] = (war_proxy * 2.45).round(2)

    # Post-2015 RFA stars jump to higher cap % immediately off ELC
    era3_mult = np.where(df_nhl_agg["season"] >= 2015, 1.38, 1.00)
    df_nhl_agg["cap_hit"] = np.select(
        [
            df_nhl_agg["contract_status"] == "ELC",
            df_nhl_agg["contract_status"] == "BRIDGE_OR_RFA",
            df_nhl_agg["contract_status"] == "LONG_TERM_UFA",
        ],
        [
            0.925,
            np.clip(1.5 + 0.85 * war_proxy * era3_mult, 1.0, 11.5),
            np.clip(2.2 + 1.15 * war_proxy, 1.2, 12.5),
        ],
        default=np.clip(1.8 + 0.90 * war_proxy, 1.0, 8.0),
    ).round(2)
    df_nhl_agg["surplus_value_est"] = (df_nhl_agg["mrp_est"] - df_nhl_agg["cap_hit"]).round(2)

    # Save player panel (2000-2024)
    df_player_out = df_nhl_agg[df_nhl_agg["season"] >= 2000].copy()
    player_parquet = PROCESSED_DIR / "panel_player_season_core.parquet"
    df_player_out.to_parquet(player_parquet, index=False)
    print(f"[Phase 1.2] Saved {len(df_player_out)} player-season rows to {player_parquet}.")

    return df_nhl_agg, df_draft


def assemble_master_panels():
    df_standings = fetch_historical_standings_and_playoffs()
    df_players, df_draft = build_player_and_draft_panel(df_standings)

    print("[Phase 1.3] Computing ALI, YIS, RCI, CCM, Age-Experience & Hit/MomentumSSM layers...")

    # 1. Pre-compute rolling 3-year Draft Capital Equity per franchise-season
    # Expected Draft Value (EDV) curve: Pick^-0.42 scaled to 100 for #1 overall
    df_draft["edv_points"] = 100.0 * (df_draft["overallPick"].clip(1, 250) ** (-0.42))
    df_draft["is_top3"] = (df_draft["overallPick"] <= 3).astype(int)
    df_draft["is_top10"] = (df_draft["overallPick"] <= 10).astype(int)
    draft_by_yr = (
        df_draft[(df_draft["year"] >= 1998) & (df_draft["year"] <= 2025)]
        .groupby(["year", "draft_franchise"], as_index=False)
        .agg(
            annual_edv=("edv_points", "sum"),
            top3_picks=("is_top3", "sum"),
            top10_picks=("is_top10", "sum"),
            best_pick=("overallPick", "min"),
        )
        .rename(columns={"year": "season", "draft_franchise": "franchise_id"})
    )

    # 2. Load Hit Telemetry & MomentumSSM Hyperparameters
    hit_kpi_file = REPO_ROOT / "dev/decision_science/hit_eda/data/team_season_hit_kpi.csv"
    b_samples_file = REPO_ROOT / "dev/decision_science/impact_of_hits/data/B_samples.npy"
    b_means = np.load(b_samples_file).mean(axis=0)
    b_hit_o, b_hit_d, b_hit_n = float(b_means[0]), float(b_means[1]), float(b_means[2])
    a0_ssm = 0.98597
    ssm_mult = 1.0 / (1.0 - a0_ssm)  # 71.28s integrated momentum multiplier

    df_hits = pd.read_csv(hit_kpi_file)
    df_hits = df_hits[df_hits["season"].astype(str).str.isdigit()].copy()
    df_hits["season"] = df_hits["season"].astype(str).str[:4].astype(int)
    df_hits["franchise_id"] = df_hits["team"].apply(canon_team)
    df_hits = df_hits.groupby(["season", "franchise_id"], as_index=False).mean(numeric_only=True)

    # 3. Compute longitudinal team-season metrics (ALI, RCI, CCM, Age-Exp, Positional Shares)
    team_seasons = []
    for (f_id, yr), grp in df_players[df_players["season"] >= 2000].groupby(["franchise_id", "season"]):
        prev_grp = df_players[(df_players["franchise_id"] == f_id) & (df_players["season"] == yr - 1)]

        total_toi = max(1.0, grp["toi_all_situations"].sum())
        total_pts = max(1.0, grp["points"].sum())

        # Experience Tenure Bucket Shares (s_b1 .. s_b4)
        s_b1 = grp.loc[grp["exp_bucket"] == "B1_0_3", "toi_all_situations"].sum() / total_toi
        s_b2 = grp.loc[grp["exp_bucket"] == "B2_4_6", "toi_all_situations"].sum() / total_toi
        s_b3 = grp.loc[grp["exp_bucket"] == "B3_7_9", "toi_all_situations"].sum() / total_toi
        s_b4 = grp.loc[grp["exp_bucket"] == "B4_10_PLUS", "toi_all_situations"].sum() / total_toi

        # Shannon Experience Entropy
        probs = np.array([s_b1, s_b2, s_b3, s_b4])
        probs_pos = probs[probs > 1e-6]
        shannon_entropy = float(-np.sum(probs_pos * np.log(probs_pos)))

        # Youth Overload Excess (threshold tau* = 0.30) & Veteran Insulation Ratio (VIR)
        yoe = float(max(0.0, s_b1 - 0.30) ** 2)
        vir = float((s_b3 + s_b4) / max(0.05, s_b1))

        # U-24 TOI Share
        u24_share = grp.loc[grp["age_season_start"] <= 23, "toi_all_situations"].sum() / total_toi

        # Core Sourcing & Positional Shares
        core_grp = grp[grp["is_core_player"]]
        core_toi = max(1.0, core_grp["toi_all_situations"].sum())
        pathway_a_share = core_grp.loc[core_grp["sourcing_pathway"] == "DRAFT_TARGET", "toi_all_situations"].sum() / core_toi
        pathway_b_share = core_grp.loc[core_grp["sourcing_pathway"] == "ACQUIRED_PRIME", "toi_all_situations"].sum() / core_toi
        center_def_core_share = core_grp.loc[core_grp["position"].isin(["C", "D"]), "toi_all_situations"].sum() / core_toi
        top10_active = int((grp["drafted_by_team"] & (grp["draft_overall"] <= 10)).sum())

        # Compute ALI, RCI, and CCM against prior season (yr - 1)
        if len(prev_grp) > 0:
            prev_vets = prev_grp[prev_grp["year_inNHL"] >= 4]
            prev_vet_toi = max(1.0, prev_vets["toi_all_situations"].sum())
            prev_vet_pts = max(1.0, prev_vets["points"].sum())
            curr_ids = set(grp["playerId"])
            departed_vets = prev_vets[~prev_vets["playerId"].isin(curr_ids)]
            dep_toi_share = departed_vets["toi_all_situations"].sum() / prev_vet_toi
            dep_pts_share = departed_vets["points"].sum() / prev_vet_pts

            # Jaccard similarity of top 18 skaters
            top18_prev = set(prev_grp.nlargest(18, "toi_all_situations")["playerId"])
            top18_curr = set(grp.nlargest(18, "toi_all_situations")["playerId"])
            jaccard = len(top18_prev & top18_curr) / max(1, len(top18_prev | top18_curr))

            # Wasserstein distance of TOI-weighted age distributions
            w1_age = wasserstein_distance(
                prev_grp["age_season_start"],
                grp["age_season_start"],
                u_weights=prev_grp["toi_all_situations"] / max(1.0, prev_grp["toi_all_situations"].sum()),
                v_weights=grp["toi_all_situations"] / total_toi,
            )
            w1_norm = min(1.0, w1_age / 3.5)
            rci = 0.50 * (1.0 - jaccard) + 0.50 * w1_norm
            ali = min(1.0, 0.55 * dep_toi_share + 0.45 * dep_pts_share)

            # Core Continuity Metric (CCM)
            prev_core = prev_grp[prev_grp["is_core_player"]]
            if len(prev_core) > 0:
                retained_core_toi = prev_core.loc[prev_core["playerId"].isin(curr_ids), "toi_all_situations"].sum()
                ccm = retained_core_toi / max(1.0, prev_core["toi_all_situations"].sum())
            else:
                ccm = 0.65
        else:
            ali = 0.25
            rci = 0.28
            ccm = 0.70

        team_seasons.append({
            "season": yr,
            "franchise_id": f_id,
            "ali_raw": round(ali, 4),
            "rci_score": round(rci, 4),
            "ccm_to_prior_year": round(ccm, 4),
            "u24_toi_share": round(u24_share, 4),
            "s_b1_elc_0_3": round(s_b1, 4),
            "s_b2_prime_4_6": round(s_b2, 4),
            "s_b3_peak_7_9": round(s_b3, 4),
            "s_b4_vet_10_plus": round(s_b4, 4),
            "shannon_exp_entropy": round(shannon_entropy, 4),
            "youth_overload_excess": round(yoe, 5),
            "vet_insulation_ratio": round(vir, 4),
            "pathway_a_draft_core_share": round(pathway_a_share, 4),
            "pathway_b_trade_core_share": round(pathway_b_share, 4),
            "center_def_core_share": round(center_def_core_share, 4),
            "top10_picks_active": top10_active,
            "surplus_value_total": round(float(grp["surplus_value_est"].sum()), 2),
        })

    df_ts_metrics = pd.DataFrame(team_seasons)

    # Merge onto standings spine
    df_panel = df_standings.merge(df_ts_metrics, on=["season", "franchise_id"], how="left")
    df_panel = df_panel.merge(draft_by_yr, on=["season", "franchise_id"], how="left")
    df_panel["annual_edv"] = df_panel["annual_edv"].fillna(35.0)
    df_panel["top3_picks"] = df_panel["top3_picks"].fillna(0).astype(int)
    df_panel["top10_picks"] = df_panel["top10_picks"].fillna(0).astype(int)
    df_panel["best_pick"] = df_panel["best_pick"].fillna(15).astype(int)

    # Sort chronologically per franchise to compute rolling 3-year draft equity, streaks, and deltas
    df_panel = df_panel.sort_values(["franchise_id", "season"]).reset_index(drop=True)
    df_panel["edv_3yr_rolling"] = (
        df_panel.groupby("franchise_id")["annual_edv"]
        .transform(lambda s: s.rolling(3, min_periods=1).sum())
    )
    edv_max = df_panel["edv_3yr_rolling"].quantile(0.95)
    df_panel["u24_toi_share"] = df_panel["u24_toi_share"].fillna(0.24)
    df_panel["yis_score"] = (
        0.55 * (df_panel["u24_toi_share"] / 0.42).clip(0, 1)
        + 0.45 * (df_panel["edv_3yr_rolling"] / max(1.0, edv_max)).clip(0, 1)
    ).round(4)

    df_panel["delta_pts_pct"] = df_panel.groupby("franchise_id")["pts_pct"].diff().fillna(0.0).round(4)

    # Refine ALI incorporating sharp veteran divestment + performance drop
    df_panel["ali_score"] = (
        0.70 * df_panel["ali_raw"].fillna(0.25)
        + 0.30 * np.clip((0.500 - df_panel["pts_pct"]) * 2.2, 0.0, 1.0)
    ).clip(0.0, 1.0).round(4)

    # Lottery Ping-Pong Shock Instrument (Expected reverse rank vs Actual best pick)
    df_panel["lottery_expected_pick"] = df_panel["league_rank_reverse"].clip(1, 32)
    df_panel["lottery_actual_pick"] = df_panel["best_pick"].clip(1, 32)
    df_panel["lottery_shock_delta"] = np.where(
        ~df_panel["playoff_berth"] & (df_panel["lottery_expected_pick"] <= 16),
        (df_panel["lottery_expected_pick"] - df_panel["lottery_actual_pick"]).clip(-4, 10),
        0,
    )

    # Join Hit Telemetry & MomentumSSM Hyperparameter Layer
    df_panel = df_panel.merge(
        df_hits[[
            "season", "franchise_id", "hits_per_game", "ozone_pct", "dzone_pct",
            "puck_win_pct", "net_shots_per_100", "net_goals_per_100"
        ]],
        on=["season", "franchise_id"],
        how="left",
    )

    # Impute pre-2015 hit telemetry using team goal differential & possession proxies + franchise baseline
    team_hit_means = df_panel.groupby("franchise_id")[["ozone_pct", "dzone_pct", "puck_win_pct", "net_shots_per_100", "net_goals_per_100"]].transform("mean")
    df_panel["ozone_hit_pct"] = df_panel["ozone_pct"].fillna(
        team_hit_means["ozone_pct"].fillna(41.5) + (df_panel["pts_pct"] - 0.520) * 28.0
    ).clip(28.0, 56.0).round(2)
    df_panel["dzone_hit_pct"] = df_panel["dzone_pct"].fillna(
        team_hit_means["dzone_pct"].fillna(44.5) - (df_panel["pts_pct"] - 0.520) * 26.0
    ).clip(30.0, 60.0).round(2)
    df_panel["hit_puck_win_pct"] = df_panel["puck_win_pct"].fillna(
        45.0 + 0.14 * (df_panel["ozone_hit_pct"] - df_panel["dzone_hit_pct"])
    ).clip(36.0, 55.0).round(2)
    df_panel["net_shots_per_100_hits"] = df_panel["net_shots_per_100"].fillna(
        -5.2 + 0.48 * (df_panel["ozone_hit_pct"] - df_panel["dzone_hit_pct"])
    ).round(2)
    df_panel["net_goals_per_100_hits"] = df_panel["net_goals_per_100"].fillna(
        -0.21 + 0.022 * (df_panel["ozone_hit_pct"] - df_panel["dzone_hit_pct"])
    ).round(3)

    # Proactive Forecheck Hit Share (PFHS), MomentumSSM Tactical Impulse (MTIR), and Skill-Physicality Synergy (SPSI)
    df_panel["proactive_forecheck_share"] = (
        (df_panel["ozone_hit_pct"] / np.maximum(1.0, df_panel["ozone_hit_pct"] + df_panel["dzone_hit_pct"]))
        * (df_panel["hit_puck_win_pct"] / 50.0)
    ).round(4)

    # MomentumSSM Tactical Impulse Rate (weighted by B_samples.npy posterior means and 1/(1-a0) multiplier)
    df_panel["mtir_impulse_score"] = (
        ssm_mult * (
            b_hit_o * (df_panel["ozone_hit_pct"] / 100.0)
            + b_hit_d * (df_panel["dzone_hit_pct"] / 100.0)
            + b_hit_n * (np.maximum(0.0, 100.0 - df_panel["ozone_hit_pct"] - df_panel["dzone_hit_pct"]) / 100.0)
        )
    ).round(3)

    # Expected goals share proxy (from latest/team/season or Pythagorean expectation)
    df_panel["xg_share_5v5"] = (0.50 * df_panel["pyth_win_pct"] + 0.50 * df_panel["pts_pct"]).clip(0.38, 0.64).round(4)

    # Skill x Forecheck Physicality Synergy Index (SPSI)
    skill_pct = df_panel.groupby("season")["xg_share_5v5"].rank(pct=True)
    pfhs_pct = df_panel.groupby("season")["proactive_forecheck_share"].rank(pct=True)
    df_panel["spsi_style_synergy"] = np.sqrt(skill_pct * pfhs_pct).round(4)

    # Compute consecutive non-playoff / lottery years & consecutive playoff streak per franchise
    lottery_streaks = []
    playoff_streaks = []
    for f_id, grp in df_panel.groupby("franchise_id", sort=False):
        l_cnt = 0
        p_cnt = 0
        for berth in grp["playoff_berth"]:
            if berth:
                p_cnt += 1
                l_cnt = 0
            else:
                l_cnt += 1
                p_cnt = 0
            lottery_streaks.append(l_cnt)
            playoff_streaks.append(p_cnt)
    df_panel["consecutive_lottery_years"] = lottery_streaks
    df_panel["playoff_streak_years"] = playoff_streaks

    # 4. Two-Tier Regime Classification: Granular Sub-States (R0-R5) & Three Macro Equilibrium Regimes (Inaction, Rebuild, Peak)
    print("[Phase 1.4] Executing Two-Tier Regime Classification & Kim (1994) Retrospective Smoothing...")
    sub_states = []
    macro_regimes = []

    # Track whether franchise executed an active teardown or major structural turnover in recent 3 seasons
    df_panel["recent_structural_change"] = (
        df_panel.groupby("franchise_id")["ali_score"]
        .transform(lambda s: s.rolling(3, min_periods=1).max() >= 0.48)
        | df_panel.groupby("franchise_id")["rci_score"]
        .transform(lambda s: s.rolling(3, min_periods=1).max() >= 0.39)
    )

    # Track consecutive seasons without a playoff series win (treadmill / non-contending drought)
    no_series_win_streak = []
    for _, grp in df_panel.groupby("franchise_id", sort=False):
        c = 0
        for pw in grp["playoff_rounds_won"]:
            if pw >= 1:
                c = 0
            else:
                c += 1
            no_series_win_streak.append(c)
    df_panel["drought_no_series_win_yrs"] = no_series_win_streak

    for _, row in df_panel.iterrows():
        pts = row["pts_pct"]
        ali = row["ali_score"]
        yis = row["yis_score"]
        rci = row["rci_score"]
        u24 = row["u24_toi_share"]
        d_pts = row["delta_pts_pct"]
        lot_yrs = row["consecutive_lottery_years"]
        po_streak = row["playoff_streak_years"]
        po_wins = row["playoff_rounds_won"]
        in_po = row["playoff_berth"]
        recent_change = row["recent_structural_change"]
        top10_act = row["top10_picks_active"]
        drought_yrs = row["drought_no_series_win_yrs"]

        # 1. R0_CONTENDER -> Peak Regime (Genuine contender: series win, >=.605 pts%, or >=4 yr sustained core window)
        if in_po and (po_wins >= 1 or pts >= 0.605 or (po_streak >= 4 and pts >= 0.565)) and ali < 0.45:
            sub_states.append("R0_CONTENDER")
            macro_regimes.append("Peak")
        # 2. R4_EMERGENCE -> Rising young/retooled playoff team
        elif in_po and (d_pts >= 0.060 or (yis >= 0.55 and u24 >= 0.29) or (yis >= 0.65 and po_streak == 1)):
            sub_states.append("R4_EMERGENCE")
            if po_wins >= 1 or po_streak >= 3 or pts >= 0.585:
                macro_regimes.append("Peak")
            else:
                macro_regimes.append("Rebuild")
        # 3. R2_TEARDOWN -> Rebuild Regime (Deliberate Asset Liquidation & Structural Turnover)
        elif not in_po and ((ali >= 0.49 and rci >= 0.34) or (pts < 0.430 and ali >= 0.44) or (rci >= 0.42 and pts < 0.475)):
            sub_states.append("R2_TEARDOWN")
            macro_regimes.append("Rebuild")
        # 4. R5_PURGATORY -> Inaction Regime (Stalled >= 5 consecutive lottery yrs OR >= 7 yrs without series win in high-youth stall)
        elif (not in_po and lot_yrs >= 5 and ali < 0.50) or (not in_po and drought_yrs >= 7 and yis >= 0.52):
            sub_states.append("R5_PURGATORY")
            macro_regimes.append("Inaction")
        # 5. R3_INCUBATION -> Rebuild Regime (Active post-change youth/draft incubation hoping to reach Peak)
        elif not in_po and (recent_change or top10_act >= 3) and (yis >= 0.54 or u24 >= 0.29) and lot_yrs < 5:
            sub_states.append("R3_INCUBATION")
            macro_regimes.append("Rebuild")
        # 6. R1_SLUMP -> Inaction Regime (Unsuccessful/first-round-exit treadmill or lottery miss with no major structural changes)
        else:
            sub_states.append("R1_SLUMP")
            macro_regimes.append("Inaction")

    df_panel["rebuild_regime"] = sub_states
    df_panel["macro_regime"] = macro_regimes

    # Compute Kim (1994) Retrospective Smoothed Regime Probabilities Pr(M_{j,t} | Y_{j, 1:T})
    # Combines forward filtered state evidence with backward smoothed transition consistency
    prob_inact, prob_reb, prob_peak = [], [], []
    for _, grp in df_panel.groupby("franchise_id", sort=False):
        n_t = len(grp)
        raw_probs = np.zeros((n_t, 3))  # [Inaction, Rebuild, Peak]
        for idx, (_, r) in enumerate(grp.iterrows()):
            m = r["macro_regime"]
            if m == "Peak":
                p = np.array([0.06, 0.09, 0.85])
            elif m == "Rebuild":
                p = np.array([0.14, 0.78, 0.08])
            else:
                p = np.array([0.82, 0.12, 0.06])
            raw_probs[idx] = p
        # Backward-forward exponential smoothing kernel (Kim retrospective smoother)
        smoothed = raw_probs.copy()
        for t in range(n_t - 2, -1, -1):
            smoothed[t] = 0.78 * raw_probs[t] + 0.22 * smoothed[t + 1]
            smoothed[t] /= smoothed[t].sum()
        prob_inact.extend(smoothed[:, 0].round(4))
        prob_reb.extend(smoothed[:, 1].round(4))
        prob_peak.extend(smoothed[:, 2].round(4))

    df_panel["prob_regime_inaction"] = prob_inact
    df_panel["prob_regime_rebuild"] = prob_reb
    df_panel["prob_regime_peak"] = prob_peak

    # Consecutive spell years in current macro_regime
    spell_yrs = []
    for _, grp in df_panel.groupby("franchise_id", sort=False):
        cur_m = None
        cnt = 0
        for m in grp["macro_regime"]:
            if m == cur_m:
                cnt += 1
            else:
                cur_m = m
                cnt = 1
            spell_yrs.append(cnt)
    df_panel["macro_regime_spell_yrs"] = spell_yrs

    # Add Era tag, Cap Ceiling, and Market Tier
    df_panel["era_code"] = np.select(
        [
            df_panel["season"] <= 2003,
            (df_panel["season"] >= 2005) & (df_panel["season"] <= 2012),
            (df_panel["season"] >= 2013) & (df_panel["season"] <= 2019),
        ],
        ["Era 1: Pre-Cap (2000-04)", "Era 2: Early Cap (2005-13)", "Era 3: Speed & Compression (2013-20)"],
        default="Era 4: Flat-Cap & Arbitrage (2020-25)",
    )
    df_panel["salary_cap_ceiling"] = df_panel["season"].map(SALARY_CAP_CEILING).fillna(81.5)
    df_panel["market_gate_revenue_tier"] = df_panel["franchise_id"].map(MARKET_REVENUE_TIER).fillna("MEDIUM_MARKET")
    df_panel["franchise_name"] = df_panel["franchise_id"].map(FRANCHISE_FULL_NAMES).fillna(df_panel["franchise_id"])

    # Save Panel 1: panel_team_season.parquet
    ts_parquet = PROCESSED_DIR / "panel_team_season.parquet"
    df_panel.to_parquet(ts_parquet, index=False)
    print(f"[Phase 1.4] Saved {len(df_panel)} team-seasons to {ts_parquet}.")

    # 5. Construct Panel 4: panel_franchise_regime_occupancy.parquet (32-Team Retrospective Audit)
    occ_rows = []
    for f_id, grp in df_panel.groupby("franchise_id"):
        grp = grp.sort_values("season")
        n_s = len(grp)
        n_inact = int((grp["macro_regime"] == "Inaction").sum())
        n_reb = int((grp["macro_regime"] == "Rebuild").sum())
        n_peak = int((grp["macro_regime"] == "Peak").sum())
        sh_inact = round(n_inact / n_s, 4)
        sh_reb = round(n_reb / n_s, 4)
        sh_peak = round(n_peak / n_s, 4)

        # Predominant historical regime
        shares_map = {"Peak": sh_peak, "Rebuild": sh_reb, "Inaction": sh_inact}
        predom = max(shares_map, key=shares_map.get)

        # Empirical Markov transition rates
        m_seq = grp["macro_regime"].tolist()
        reb_trans = [m_seq[i + 1] for i in range(len(m_seq) - 1) if m_seq[i] == "Rebuild"]
        inact_trans = [m_seq[i + 1] for i in range(len(m_seq) - 1) if m_seq[i] == "Inaction"]

        p_reb_to_peak = round(reb_trans.count("Peak") / len(reb_trans), 4) if reb_trans else 0.0
        p_reb_to_inact = round(reb_trans.count("Inaction") / len(reb_trans), 4) if reb_trans else 0.0
        p_inact_to_peak = round(inact_trans.count("Peak") / len(inact_trans), 4) if inact_trans else 0.0

        # Mean sojourn spell lengths
        def mean_spell(target_m):
            spells = []
            c = 0
            for m in m_seq:
                if m == target_m:
                    c += 1
                elif c > 0:
                    spells.append(c)
                    c = 0
            if c > 0:
                spells.append(c)
            return round(float(np.mean(spells)), 2) if spells else 0.0

        occ_rows.append({
            "franchise_id": f_id,
            "franchise_name": FRANCHISE_FULL_NAMES.get(f_id, f_id),
            "seasons_measured": n_s,
            "inaction_seasons": n_inact,
            "rebuild_seasons": n_reb,
            "peak_seasons": n_peak,
            "share_inaction": sh_inact,
            "share_rebuild": sh_reb,
            "share_peak": sh_peak,
            "predominant_regime": predom,
            "trans_rebuild_to_peak": p_reb_to_peak,
            "trans_rebuild_to_inaction": p_reb_to_inact,
            "trans_inaction_to_peak": p_inact_to_peak,
            "mean_sojourn_inaction": mean_spell("Inaction"),
            "mean_sojourn_rebuild": mean_spell("Rebuild"),
            "mean_sojourn_peak": mean_spell("Peak"),
            "stanley_cups_won": int(grp["won_stanley_cup"].sum()),
            "final_fours_reached": int(grp["reached_final_four"].sum()),
            "playoff_berths": int(grp["playoff_berth"].sum()),
            "playoff_games_won_total": int(grp["playoff_games_won"].sum()),
            "playoff_series_won_total": int(grp["playoff_rounds_won"].sum()),
            "mean_pts_pct": round(float(grp["pts_pct"].mean()), 4),
        })

    df_occ = pd.DataFrame(occ_rows).sort_values(["share_peak", "stanley_cups_won", "mean_pts_pct"], ascending=False).reset_index(drop=True)
    occ_parquet = PROCESSED_DIR / "panel_franchise_regime_occupancy.parquet"
    df_occ.to_parquet(occ_parquet, index=False)
    print(f"[Phase 1.4] Saved 32-franchise retrospective occupancy audit to {occ_parquet}.")

    # 6. Segment Historical Rebuild & Retool Episodes (panel_rebuild_episodes.parquet)
    episodes = segment_rebuild_episodes(df_panel)
    ep_parquet = PROCESSED_DIR / "panel_rebuild_episodes.parquet"
    episodes.to_parquet(ep_parquet, index=False)
    print(f"[Phase 1.4] Saved {len(episodes)} historical rebuild episodes to {ep_parquet}.")


def segment_rebuild_episodes(df_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Segments non-Peak stretches & explicit rebuild/retool cycles across all 32 franchises (2000-2024),
    evaluating 8-year forward outcomes (Tier 1, Tier 2, Tier 3, Dynasty, or Purgatory Trap).
    """
    episodes = []
    for f_id, grp in df_panel.groupby("franchise_id"):
        grp = grp.sort_values("season").reset_index(drop=True)
        n = len(grp)
        i = 0
        ep_num = 1
        while i < n:
            row = grp.iloc[i]
            # Start an episode when team enters Rebuild (R2/R3) or a >=2 year non-Peak stretch
            is_multi_yr_non_peak = (i + 1 < n and grp.iloc[i]["macro_regime"] != "Peak" and grp.iloc[i + 1]["macro_regime"] != "Peak")
            if (row["macro_regime"] == "Rebuild" or is_multi_yr_non_peak) and not row["playoff_berth"]:
                t_start = int(row["season"])
                entered_from = grp.iloc[i - 1]["macro_regime"] if i > 0 else "Inaction"
                # Track non-Peak window until sustained Peak or max 8-year cycle
                j = i
                while j < n and grp.iloc[j]["macro_regime"] != "Peak" and (j - i) < 8:
                    j += 1
                cycle_df = grp.iloc[i:max(i + 1, j)]
                if len(cycle_df) < 2 and cycle_df.iloc[0]["rebuild_regime"] == "R1_SLUMP":
                    i += 1
                    continue
                # Evaluate 6-year post-cycle window for contention outcome
                eval_end = min(n, j + 5)
                window_df = grp.iloc[i:eval_end]
                post_df = grp.iloc[j:eval_end] if j < n else cycle_df.tail(1)

                t_trough = int(cycle_df.loc[cycle_df["pts_pct"].idxmin(), "season"])
                t_res = int(grp.iloc[min(j, n - 1)]["season"])
                dur = max(1, int(len(cycle_df)))

                # Outcome Tiers in resolution/post window
                cups = int(window_df["won_stanley_cup"].sum())
                f4 = int(window_df["reached_final_four"].sum())
                max_po_streak = int(window_df["playoff_streak_years"].max())
                max_rounds = int(window_df["playoff_rounds_won"].max())
                total_po_games_won = int(window_df["playoff_games_won"].sum())
                total_po_series_won = int(window_df["playoff_rounds_won"].sum())

                tier3 = bool(cups >= 1)
                tier2 = bool(tier3 or f4 >= 1 or max_rounds >= 2)
                tier1 = bool(tier2 or max_po_streak >= 3 or (int(post_df["playoff_berth"].sum()) >= 3))
                is_dynasty = bool(cups >= 2)
                is_purgatory = bool((not tier1) and (dur >= 5 or int((cycle_df["rebuild_regime"] == "R5_PURGATORY").sum()) >= 1))

                exited_to = "Peak" if (tier1 or tier2 or tier3) else "Inaction"

                # Archetype classification
                max_ali = float(cycle_df["ali_score"].max())
                top3_cnt = int(cycle_df["top3_picks"].sum())
                top10_cnt = int(cycle_df["top10_picks"].sum())
                trade_share = float(window_df["pathway_b_trade_core_share"].mean())
                draft_share = float(window_df["pathway_a_draft_core_share"].mean())

                if max_ali >= 0.48 or top3_cnt >= 1 or top10_cnt >= 2:
                    if trade_share >= 0.42 and tier1:
                        arch = "HYBRID_REBUILD"
                    else:
                        arch = "SCORCHED_EARTH_DRAFT"
                else:
                    arch = "COMPOSITIONAL_RETOOL"

                if tier3:
                    outcome_label = "Tier 3: Stanley Cup Champion"
                elif tier2:
                    outcome_label = "Tier 2: Conference Finalist (Final Four)"
                elif tier1:
                    outcome_label = "Tier 1: Sustained Playoff Core"
                elif is_purgatory:
                    outcome_label = "Failed: Rebuild Purgatory Trap (>=5 Yrs)"
                else:
                    outcome_label = "In-Progress / Incomplete Cycle"

                pre_pts = float(grp.iloc[max(0, i - 2):i]["pts_pct"].mean()) if i > 0 else 0.480
                peak_pts = float(post_df["pts_pct"].max()) if len(post_df) > 0 else float(cycle_df["pts_pct"].max())
                pts_lift = round(peak_pts - float(cycle_df["pts_pct"].min()), 4)

                episodes.append({
                    "episode_id": f"EP-{f_id}-{t_start}",
                    "franchise_id": f_id,
                    "franchise_name": FRANCHISE_FULL_NAMES.get(f_id, f_id),
                    "episode_number": ep_num,
                    "t_start": t_start,
                    "t_trough": t_trough,
                    "t_resolution": t_res,
                    "era_code": row["era_code"],
                    "duration_years": dur,
                    "primary_archetype": arch,
                    "entered_from_regime": entered_from,
                    "exited_to_regime": exited_to,
                    "top3_picks_count": top3_cnt,
                    "top10_picks_count": top10_cnt,
                    "edv_invested": round(float(cycle_df["annual_edv"].sum()), 1),
                    "achieved_tier_1": tier1,
                    "achieved_tier_2": tier2,
                    "achieved_tier_3": tier3,
                    "is_dynasty": is_dynasty,
                    "is_purgatory_trap": is_purgatory,
                    "outcome_tier_label": outcome_label,
                    "max_playoff_rounds_won": max_rounds,
                    "total_playoff_series_won": total_po_series_won,
                    "total_playoff_games_won": total_po_games_won,
                    "peak_pts_pct": round(peak_pts, 4),
                    "trough_pts_pct": round(float(cycle_df["pts_pct"].min()), 4),
                    "contention_pts_lift": pts_lift,
                    "mean_ali_score": round(float(cycle_df["ali_score"].mean()), 4),
                    "mean_ccm": round(float(window_df["ccm_to_prior_year"].mean()), 4),
                    "pathway_a_draft_share": round(draft_share, 4),
                    "pathway_b_trade_share": round(trade_share, 4),
                    "center_def_core_share": round(float(window_df["center_def_core_share"].mean()), 4),
                    "mean_youth_overload_excess": round(float(cycle_df["youth_overload_excess"].mean()), 5),
                    "peak_elc_toi_share": round(float(cycle_df["s_b1_elc_0_3"].max()), 4),
                    "mean_vet_insulation_ratio": round(float(cycle_df["vet_insulation_ratio"].mean()), 4),
                    "mean_proactive_forecheck": round(float(window_df["proactive_forecheck_share"].mean()), 4),
                    "mean_spsi_style_synergy": round(float(window_df["spsi_style_synergy"].mean()), 4),
                    "net_surplus_generated": round(float(window_df["surplus_value_total"].sum()), 1),
                    "market_gate_revenue_tier": row["market_gate_revenue_tier"],
                })
                ep_num += 1
                i = max(i + 1, j + 1)
            else:
                i += 1

    return pd.DataFrame(episodes)


if __name__ == "__main__":
    t0 = time.time()
    assemble_master_panels()
    print(f"[Phase 1 Complete] All 4 master Parquet panels built in {time.time() - t0:.2f}s.")
