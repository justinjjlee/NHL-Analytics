"""
data_loader.py — Unified Data Provider for Rinklytics Dashboard.

Architecture:
1. Ongoing Data (Scores, Odds, Player Performance, Team Season KPIs, Physical Hit Analytics):
   - Primary: Streams high-performance Parquet datasets from the public Hugging Face repository
     (https://huggingface.co/datasets/jjerlee/nhl-analytics).
   - Backup: Falls back to local files or GitHub repository raw CSV files if Hugging Face is unreachable.
   - Filtering & Polishing: Handles year-aggregated longitudinal panels, normalizing multi-season
     schemas, date types, and primary keys for seamless view rendering.

2. One-Off Data (Map Geography, Draft Progression, Blocking Shots Analysis):
   - Directly loaded from local files or the official GitHub repository
     (https://raw.githubusercontent.com/justinjjlee/NHL-Analytics/main/...).
"""

import os
import glob
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Tuple

# ── Endpoint Constants ────────────────────────────────────────────────────────
HF_DATASET_BASE = "https://huggingface.co/datasets/jjerlee/nhl-analytics/resolve/main"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/justinjjlee/NHL-Analytics/main"


def get_project_root() -> str:
    """Resolve absolute path to project root from app/exe/."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '..', '..'))


def get_local_path(rel_path: str) -> str:
    """Resolve a relative path against the project root."""
    return os.path.join(get_project_root(), rel_path)


# ── Generic Safe Loaders with Fallback ────────────────────────────────────────

def load_parquet_table(
    hf_subpath: str,
    fallback_loader=None,
    desc: str = ""
) -> pd.DataFrame:
    """
    Attempts to read a Parquet table from Hugging Face Datasets.
    If network or parsing fails, executes fallback_loader (local / GitHub raw CSVs).
    """
    hf_url = f"{HF_DATASET_BASE}/{hf_subpath}"
    try:
        df = pd.read_parquet(hf_url)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        pass

    # Execute fallback if Hugging Face read failed or returned empty
    if fallback_loader is not None:
        try:
            return fallback_loader()
        except Exception:
            pass

    return pd.DataFrame()


def load_oneoff_github_csv(rel_path: str) -> pd.DataFrame:
    """
    Loads one-off static datasets (e.g. map, draft, special studies).
    Checks local repository path first, then falls back to GitHub raw URL.
    """
    local_path = get_local_path(rel_path)
    if os.path.exists(local_path):
        try:
            return pd.read_csv(local_path)
        except Exception:
            pass

    # Fallback to GitHub raw URL
    gh_url = f"{GITHUB_RAW_BASE}/{rel_path.lstrip('/')}"
    try:
        return pd.read_csv(gh_url)
    except Exception as e:
        return pd.DataFrame()


# ── 1. ONGOING: Scoreboard & Odds Data ────────────────────────────────────────

@st.cache_data(max_entries=1, ttl=3600)
def load_scoreboard_box_and_odds() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], List]:
    """
    Loads games and odds data for the Scoreboard view.
    Primary: Hugging Face data/games.parquet and data/odds.parquet.
    Backup: local latest/box/*box.csv and *odds.csv.
    """
    def _fallback_local():
        box_dir = get_local_path('latest/box')
        box_files = sorted(glob.glob(f"{box_dir}/*box.csv"), reverse=True)
        if not box_files:
            return None, None, []
        box_dfs = []
        for f in box_files:
            try:
                box_dfs.append(pd.read_csv(f))
            except Exception:
                continue
        if not box_dfs:
            return None, None, []
        b_df = pd.concat(box_dfs, ignore_index=True)
        b_df['date'] = pd.to_datetime(b_df['date'])
        b_df = b_df.drop_duplicates().sort_values('date', ascending=False)

        odds_files = glob.glob(f"{box_dir}/*odds.csv") + glob.glob(f"{box_dir}/*odds_US.csv")
        o_df = None
        if odds_files:
            odds_dfs = []
            for f in sorted(odds_files, reverse=True):
                try:
                    tdf = pd.read_csv(f)
                    if 'gameId' in tdf.columns:
                        tdf = tdf.rename(columns={'gameId': 'gameid'})
                    tdf = tdf[tdf['odds_description'] == 'MONEY_LINE_2_WAY']
                    odds_dfs.append(tdf)
                except Exception:
                    continue
            if odds_dfs:
                o_df = pd.concat(odds_dfs, ignore_index=True).drop_duplicates(subset=['gameid'])
        return b_df, o_df, b_df['date'].dt.date.unique()

    try:
        # Load games from Hugging Face
        df_games = pd.read_parquet(f"{HF_DATASET_BASE}/data/games.parquet")
        if df_games is None or df_games.empty:
            return _fallback_local()

        box_df = df_games.rename(columns={
            'game_id': 'gameid',
            'game_date': 'date',
            'home_team': 'tricode_for',
            'away_team': 'tricode_against',
            'home_score': 'metric_score_for',
            'away_score': 'metric_score_against',
        })
        box_df['date'] = pd.to_datetime(box_df['date'])
        box_df = box_df.drop_duplicates(subset=['gameid']).sort_values('date', ascending=False).reset_index(drop=True)
        available_dates = box_df['date'].dt.date.unique()

        # Load odds from Hugging Face
        odds_df = None
        try:
            df_odds = pd.read_parquet(f"{HF_DATASET_BASE}/data/odds.parquet")
            if df_odds is not None and not df_odds.empty:
                ml_odds = df_odds[df_odds['odds_description'] == 'MONEY_LINE_2_WAY'].copy()
                if 'country' in ml_odds.columns:
                    ml_odds['country_prio'] = ml_odds['country'].map({'US': 0, 'CAN': 1}).fillna(2)
                    ml_odds = ml_odds.sort_values(['game_id', 'country_prio']).drop_duplicates(subset=['game_id'])
                else:
                    ml_odds = ml_odds.drop_duplicates(subset=['game_id'])

                odds_df = ml_odds.rename(columns={
                    'game_id': 'gameid',
                    'home_odds': 'home_odds_value',
                    'away_odds': 'away_odds_value',
                })
        except Exception:
            pass

        return box_df, odds_df, available_dates

    except Exception:
        return _fallback_local()


# ── 2. ONGOING: Player Season & Game Data ─────────────────────────────────────

@st.cache_data(max_entries=1, ttl=3600)
def load_player_season_dataset() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Loads multi-season player statistics.
    Primary: Hugging Face measurement/player/player_season.parquet.
    Backup: local latest/player/*_player.csv.
    """
    def _fallback_local():
        player_dir = get_local_path('latest/player')
        files = sorted(glob.glob(os.path.join(player_dir, '*_player.csv')))
        if not files:
            return None, None
        frames = []
        for f in files:
            year = os.path.basename(f).split('_')[0]
            try:
                df = pd.read_csv(f)
                df['season_label'] = f"{year}-{str(int(year)+1)[2:]}"
                df['season_year'] = int(year)
                frames.append(df)
            except Exception:
                pass
        if not frames:
            return None, None
        all_df = pd.concat(frames, ignore_index=True)
        all_df.rename(columns={
            'firstName.default': 'firstName',
            'lastName.default': 'lastName',
        }, inplace=True)
        all_df['fullName'] = all_df['firstName'].fillna('') + ' ' + all_df['lastName'].fillna('')
        all_df = all_df[all_df['idx_season_type'].fillna(2) == 2].copy()
        for col in ['gamesPlayed', 'goals', 'assists', 'points', 'plusMinus', 'shots',
                    'shootingPctg', 'avgTimeOnIcePerGame', 'faceoffWinPctg', 'penaltyMinutes',
                    'powerPlayGoals', 'shorthandedGoals', 'gameWinningGoals', 'overtimeGoals',
                    'avgShiftsPerGame', 'gamesStarted', 'wins', 'losses', 'overtimeLosses',
                    'goalsAgainstAverage', 'savePercentage', 'shotsAgainst', 'saves',
                    'goalsAgainst', 'shutouts']:
            if col in all_df.columns:
                all_df[col] = pd.to_numeric(all_df[col], errors='coerce')
        g = all_df[all_df['positionCode'] == 'G'].copy()
        s = all_df[all_df['positionCode'].isin(['C', 'L', 'R', 'D'])].copy()
        return s, g

    try:
        df = pd.read_parquet(f"{HF_DATASET_BASE}/measurement/player/player_season.parquet")
        if df is None or df.empty:
            return _fallback_local()

        # Schema alignment & seasonal indicators
        yr_col = 'yr_season' if 'yr_season' in df.columns else 'idx_season'
        df['season_year'] = df[yr_col].astype(int)
        df['season_label'] = df['season_year'].astype(str) + "-" + (df['season_year'] + 1).astype(str).str[2:]

        df.rename(columns={
            'firstName.default': 'firstName',
            'lastName.default': 'lastName',
        }, inplace=True)
        df['fullName'] = df['firstName'].fillna('') + ' ' + df['lastName'].fillna('')

        # Filter regular season
        if 'idx_season_type' in df.columns:
            df = df[df['idx_season_type'].fillna(2) == 2].copy()

        numeric_cols = [
            'gamesPlayed', 'goals', 'assists', 'points', 'plusMinus',
            'shots', 'shootingPctg', 'avgTimeOnIcePerGame', 'faceoffWinPctg',
            'penaltyMinutes', 'powerPlayGoals', 'shorthandedGoals',
            'gameWinningGoals', 'overtimeGoals', 'avgShiftsPerGame',
            'gamesStarted', 'wins', 'losses', 'overtimeLosses',
            'goalsAgainstAverage', 'savePercentage', 'shotsAgainst',
            'saves', 'goalsAgainst', 'shutouts',
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        skaters = df[df['positionCode'].isin(['C', 'L', 'R', 'D'])].copy()
        goalies = df[df['positionCode'] == 'G'].copy()

        def _sec_to_mmss(secs):
            try:
                secs = float(secs)
                return f"{int(secs // 60)}:{int(secs % 60):02d}"
            except Exception:
                return ""

        # Derived stats
        skaters['points_per_game'] = skaters['points'] / skaters['gamesPlayed'].replace(0, np.nan)
        skaters['goals_per_game'] = skaters['goals'] / skaters['gamesPlayed'].replace(0, np.nan)
        skaters['assists_per_game'] = skaters['assists'] / skaters['gamesPlayed'].replace(0, np.nan)
        skaters['shots_per_game'] = skaters['shots'] / skaters['gamesPlayed'].replace(0, np.nan)
        skaters['toi_per_game_min'] = skaters['avgTimeOnIcePerGame'].apply(_sec_to_mmss)

        goalies['saves_per_game'] = goalies['saves'] / goalies['gamesStarted'].replace(0, np.nan)
        goalies['win_pct'] = goalies['wins'] / goalies['gamesPlayed'].replace(0, np.nan)

        return skaters, goalies

    except Exception:
        return _fallback_local()


@st.cache_data(max_entries=2, ttl=1800)
def load_player_game_dataset(season_year: Optional[int] = None) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Loads individual game-by-game player performance records.
    Primary: Hugging Face data/player_box.parquet filtered by season_year.
    Backup: local latest/box/*_box_player.csv.
    """
    def _fallback_local():
        box_dir = get_local_path('latest/box')
        if season_year is not None:
            tf = os.path.join(box_dir, f"{season_year}_box_player.csv")
            files = [tf] if os.path.exists(tf) else []
        else:
            files = sorted(glob.glob(os.path.join(box_dir, '*_box_player.csv')))
        if not files:
            return None, None
        needed_cols = [
            'gameid', 'playerId', 'teamloc', 'abbrev', 'position', 'gameDate',
            'goals', 'assists', 'points', 'plusMinus', 'pim', 'hits',
            'powerPlayGoals', 'sog', 'faceoffWinningPctg', 'blockedShots',
            'shifts', 'giveaways', 'takeaways', 'toi',
            'savePctg', 'goalsAgainst', 'shotsAgainst', 'saves',
            'name.default', 'name.cs', 'name.sk', 'name.fi', 'name.de', 'name.es', 'decision'
        ]
        frames = []
        for f in files:
            try:
                hdr = pd.read_csv(f, nrows=0).columns.tolist()
                uc = [c for c in needed_cols if c in hdr]
                frames.append(pd.read_csv(f, usecols=uc, low_memory=False))
            except Exception:
                pass
        if not frames:
            return None, None
        adf = pd.concat(frames, ignore_index=True)
        locale_cols = ['name.cs', 'name.sk', 'name.fi', 'name.de', 'name.es']
        adf['playerName'] = adf.get('name.default', pd.Series(dtype=str))
        for col in locale_cols:
            if col in adf.columns:
                adf['playerName'] = adf['playerName'].where(adf[col].isna(), adf[col])
        adf['gameDate'] = pd.to_datetime(adf['gameDate'], errors='coerce')
        if season_year is not None:
            adf['season_year'] = int(season_year)
        else:
            m = adf['gameDate'].dt.month
            y = adf['gameDate'].dt.year
            adf['season_year'] = np.where(m >= 10, y, y - 1)
        for col in ['goals', 'assists', 'points', 'plusMinus', 'pim', 'hits', 'powerPlayGoals', 'sog',
                    'faceoffWinningPctg', 'blockedShots', 'shifts', 'giveaways', 'takeaways',
                    'savePctg', 'goalsAgainst', 'shotsAgainst', 'saves']:
            if col in adf.columns:
                adf[col] = pd.to_numeric(adf[col], errors='coerce')

        loc_map = adf[adf['abbrev'].notna()][['gameid', 'teamloc', 'abbrev']].drop_duplicates()
        home_map = loc_map[loc_map['teamloc'] == 'home'].set_index('gameid')['abbrev']
        away_map = loc_map[loc_map['teamloc'] == 'away'].set_index('gameid')['abbrev']
        is_home = adf['teamloc'] == 'home'
        adf['own_team'] = np.where(is_home, adf['gameid'].map(home_map), adf['gameid'].map(away_map))
        adf['opp_team'] = np.where(is_home, adf['gameid'].map(away_map), adf['gameid'].map(home_map))
        sk = adf[adf['position'].isin(['C', 'L', 'R', 'D'])].copy()
        gl = adf[adf['position'] == 'G'].copy()
        return sk, gl

    try:
        df = pd.read_parquet(f"{HF_DATASET_BASE}/data/player_box.parquet")
        if df is None or df.empty:
            return _fallback_local()

        # Season filtering on year-aggregated panel
        df['game_id_str'] = df['game_id'].astype(str)
        df['season_year'] = df['game_id_str'].str[:4].astype(int)
        if season_year is not None:
            df = df[df['season_year'] == int(season_year)].copy()

        # Normalize schema for player views
        df['gameid'] = df['game_id']
        df['playerId'] = df['player_id']
        df['playerName'] = df['player_name']
        df['abbrev'] = df['team']
        df['own_team'] = df['team']
        df['opp_team'] = df.get('opponent', pd.Series(dtype=str))
        df['gameDate'] = pd.to_datetime(df['game_date'], errors='coerce')
        df['plusMinus'] = df.get('plus_minus', np.nan)
        df['powerPlayGoals'] = df.get('powerplay_goals', np.nan)
        df['blockedShots'] = df.get('blocked_shots', np.nan)
        df['faceoffWinningPctg'] = df.get('faceoff_pct', np.nan)
        df['savePctg'] = df.get('save_pct', np.nan)
        df['shotsAgainst'] = df.get('shots_against', np.nan)
        df['goalsAgainst'] = df.get('goals_against', df['shotsAgainst'] - df.get('saves', 0))

        num_cols = [
            'goals', 'assists', 'points', 'plusMinus', 'pim', 'hits',
            'powerPlayGoals', 'sog', 'faceoffWinningPctg', 'blockedShots',
            'shifts', 'giveaways', 'takeaways', 'savePctg', 'goalsAgainst',
            'shotsAgainst', 'saves'
        ]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        skater_games = df[df['position'].isin(['C', 'L', 'R', 'D'])].copy()
        goalie_games = df[df['position'] == 'G'].copy()

        return skater_games, goalie_games

    except Exception:
        return _fallback_local()


# ── 3. ONGOING: Team Season Standings & Advanced KPIs ─────────────────────────

@st.cache_data(max_entries=1, ttl=3600)
def load_team_season_dataset() -> pd.DataFrame:
    """
    Loads multi-season team standings and advanced metrics.
    Primary: Hugging Face measurement/box/team_season.parquet & data/games.parquet.
    Backup: local latest/team/season/*_team_season.csv and latest/box/{year}_box.csv.
    """
    def _fallback_local():
        season_dir = get_local_path("latest/team/season")
        files = glob.glob(f"{season_dir}/*_team_season.csv")
        all_data = []
        for fpath in files:
            year = os.path.basename(fpath)[:4]
            try:
                df_season = pd.read_csv(fpath)
                box_path = get_local_path(f"latest/box/{year}_box.csv")
                if os.path.exists(box_path):
                    df_box = pd.read_csv(box_path)
                    df_for = df_box[['tricode_for', 'tricode_winteam', 'period_ending']].rename(columns={'tricode_for': 'team'})
                    df_against = df_box[['tricode_against', 'tricode_winteam', 'period_ending']].rename(columns={'tricode_against': 'team'})
                    df_teams = pd.concat([df_for, df_against])
                    df_teams['W'] = (df_teams['team'] == df_teams['tricode_winteam']).astype(int)
                    df_teams['L'] = ((df_teams['team'] != df_teams['tricode_winteam']) & (df_teams['period_ending'] == 'REG')).astype(int)
                    df_teams['OTL'] = ((df_teams['team'] != df_teams['tricode_winteam']) & (df_teams['period_ending'].isin(['OT', 'SO']))).astype(int)
                    df_agg = df_teams.groupby('team')[['W', 'L', 'OTL']].sum().reset_index()
                    df_agg['PTS'] = df_agg['W'] * 2 + df_agg['OTL']
                    df_agg['GP'] = df_agg['W'] + df_agg['L'] + df_agg['OTL']
                    df_merged = pd.merge(df_agg, df_season, left_on='team', right_on='team_tri_for', how='inner')
                    df_merged['Season'] = year
                    all_data.append(df_merged)
            except Exception:
                continue
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    try:
        df_team_season = pd.read_parquet(f"{HF_DATASET_BASE}/measurement/box/team_season.parquet")
        df_games = pd.read_parquet(f"{HF_DATASET_BASE}/data/games.parquet")
        if df_team_season is None or df_team_season.empty or df_games is None or df_games.empty:
            return _fallback_local()

        # Calculate W, L, OTL, PTS per team per season from games panel
        df_games['year'] = df_games['game_id'].astype(str).str[:4]
        df_for = df_games[['home_team', 'winning_team', 'period_ending', 'year']].rename(columns={'home_team': 'team'})
        df_against = df_games[['away_team', 'winning_team', 'period_ending', 'year']].rename(columns={'away_team': 'team'})
        df_teams = pd.concat([df_for, df_against], ignore_index=True)
        df_teams['W'] = (df_teams['team'] == df_teams['winning_team']).astype(int)
        df_teams['L'] = ((df_teams['team'] != df_teams['winning_team']) & (df_teams['period_ending'] == 'REG')).astype(int)
        df_teams['OTL'] = ((df_teams['team'] != df_teams['winning_team']) & (df_teams['period_ending'].isin(['OT', 'SO']))).astype(int)
        df_agg = df_teams.groupby(['year', 'team'])[['W', 'L', 'OTL']].sum().reset_index()
        df_agg['PTS'] = df_agg['W'] * 2 + df_agg['OTL']
        df_agg['GP'] = df_agg['W'] + df_agg['L'] + df_agg['OTL']

        df_team_season['year'] = df_team_season['yr_season'].astype(str)
        df_merged = pd.merge(df_agg, df_team_season, left_on=['year', 'team'], right_on=['year', 'team_tri_for'], how='inner')
        df_merged['Season'] = df_merged['year']

        return df_merged

    except Exception:
        return _fallback_local()


# ── 4. ONGOING: Physical Hit & Sequence Analytics ─────────────────────────────

@st.cache_data(max_entries=1, ttl=3600)
def load_hit_analytics_datasets() -> Tuple:
    """
    Loads all physical hit action and sequence datasets.
    Primary: Hugging Face measurement/action/hits/*.parquet.
    Backup: local latest/action/hits/ or app/data/hit_eda/.
    """
    tables = [
        'hit_zone_summary_kpi',
        'tabulated_hit_sequences',
        'hit_goal_window_30s',
        'rink_grid_hit_sequences',
        'team_season_hit_kpi',
        'team_season_spatial_hits',
        'hit_zone_summary',
        'hit_next_events',
        'hit_sequence_transitions',
    ]

    def _fallback_local():
        current_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = get_project_root()
        res = []
        for t in tables:
            filename = f"{t}.csv"
            candidates = [
                os.path.join(repo_root, "latest/action/hits", filename),
                os.path.join(current_dir, "../data/hit_eda", filename),
                os.path.join(repo_root, "dev/decision_science/impact_of_hits/data", filename),
            ]
            loaded = None
            for c in candidates:
                if os.path.exists(c):
                    try:
                        loaded = pd.read_csv(c)
                        break
                    except Exception:
                        pass
            res.append(loaded)

        tl_path = os.path.join(current_dir, "../data/teamlist.csv")
        team_name_map = {}
        if os.path.exists(tl_path):
            try:
                tl = pd.read_csv(tl_path)
                team_name_map = dict(zip(tl["tricode"], tl["tricode"] + " - " + tl["team"]))
            except Exception:
                pass
        team_name_map["ARI"] = "ARI - Arizona Coyotes"

        return (*res[:6], team_name_map, *res[6:])

    try:
        base_url = f"{HF_DATASET_BASE}/measurement/action/hits"
        loaded_tables = {}
        for t in tables:
            try:
                df_t = pd.read_parquet(f"{base_url}/{t}.parquet")
                # Normalize season column to string to prevent int/str UI comparison mismatch
                if "season" in df_t.columns:
                    df_t["season"] = df_t["season"].astype(str)
                loaded_tables[t] = df_t
            except Exception:
                loaded_tables[t] = None

        # If any primary table failed, fallback to local
        if any(loaded_tables[t] is None for t in ['hit_zone_summary_kpi', 'team_season_hit_kpi']):
            return _fallback_local()

        # Build team_name_map using one-off teamlist
        df_tl = load_oneoff_github_csv("app/data/teamlist.csv")
        team_name_map = {}
        if not df_tl.empty and "tricode" in df_tl.columns and "team" in df_tl.columns:
            team_name_map = dict(zip(df_tl["tricode"], df_tl["tricode"] + " - " + df_tl["team"]))
        team_name_map["ARI"] = "ARI - Arizona Coyotes"

        return (
            loaded_tables['hit_zone_summary_kpi'],
            loaded_tables['tabulated_hit_sequences'],
            loaded_tables['hit_goal_window_30s'],
            loaded_tables['rink_grid_hit_sequences'],
            loaded_tables['team_season_hit_kpi'],
            loaded_tables['team_season_spatial_hits'],
            team_name_map,
            loaded_tables['hit_zone_summary'],
            loaded_tables['hit_next_events'],
            loaded_tables['hit_sequence_transitions'],
        )

    except Exception:
        return _fallback_local()


# ── 5. ONE-OFF: Map Geographic & Distance Data ────────────────────────────────

@st.cache_data(max_entries=1, ttl=3600)
def load_map_dataset() -> Optional[pd.DataFrame]:
    """
    Loads arena geographic coordinates and inter-city distance matrices.
    Sources from local repository or official GitHub repository.
    """
    df_loc = load_oneoff_github_csv('app/data/teamlist_locations_distance_meters.csv')
    df_team = load_oneoff_github_csv('app/data/teamlist.csv')

    if df_loc.empty or df_team.empty:
        return None

    try:
        df = df_loc.merge(
            df_team, left_on="tricode_home", right_on="tricode"
        ).rename(columns={"team": "home_team", "city": "home_team_city"}).drop("tricode", axis=1)

        df = df.merge(
            df_team, left_on="tricode_away", right_on="tricode"
        ).rename(columns={"team": "away_team", "city": "away_team_city"}).drop("tricode", axis=1)

        return df
    except Exception:
        return None


# ── 6. ONE-OFF: Amateur Draft Analytics Data ─────────────────────────────────

@st.cache_data(max_entries=1, ttl=3600)
def load_draft_dataset() -> Tuple[bool, Dict[str, pd.DataFrame]]:
    """
    Loads amateur draft lifecycle tracking and exceptional prospect datasets.
    Sources from local repository or official GitHub repository.
    """
    relative_paths = {
        'exceptional_players': 'dev/player/player_draft/draftee_lifecycle/results/multi_exceptional_players.csv',
        'exceptional_seasons': 'dev/player/player_draft/draftee_lifecycle/results/exceptional_seasons_details.csv',
        'merged_stats': 'dev/player/player_draft/data/merged_draft_player_stats.csv',
        'player_stats': 'dev/player/player_draft/data/all_drafted_players_chronicle.csv'
    }
    loaded = {}
    for key, path in relative_paths.items():
        df = load_oneoff_github_csv(path)
        if df.empty:
            return False, {}
        loaded[key] = df

    return True, loaded


# ── 7. ONE-OFF: Shot Blocking Value Analytics Data ────────────────────────────

@st.cache_data(max_entries=1, ttl=3600)
def load_blocking_shots_dataset() -> pd.DataFrame:
    """
    Loads downstream possession and valor outcome tables for shot blocking.
    Sources from local repository or official GitHub repository.
    """
    path = "dev/decision_science/plays_blockingShots/result/table - sequence blocked shot stats.csv"
    return load_oneoff_github_csv(path)
