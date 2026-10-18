# %% 
# Packages

import pandas as pd
import numpy as np
import requests
import time
import datetime
import shutil
import os
from config import get_team_dir, get_player_dir, get_teamlist_path, get_current_season_year, IS_DATABRICKS

TEAM_DIR = get_team_dir()
PLAYER_DIR = get_player_dir()

def safe_get(url, max_retries=5, timeout=15):
    """
    Safely request data with retries, timeout, and backoff for rate limits.
    Returns Response object on HTTP 200, None on 404/400 (e.g. non-existent team season).
    """
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return r
            elif r.status_code in [404, 400]:
                return None
            elif r.status_code == 429:
                wait = 3 * (attempt + 1)
                print(f"Rate limited (429) for {url}, backing off for {wait}s...")
                time.sleep(wait)
        except Exception as e:
            print(f"Network retry {attempt+1}/{max_retries} for {url}: {e}")
            time.sleep(2 * (attempt + 1))
    return None

# Ensure destination directory exists
os.makedirs(PLAYER_DIR, exist_ok=True)

# %% Settings
iter_year = get_current_season_year()
print(f"Iterative season: {iter_year}")
print(f"Target player directory: {PLAYER_DIR}")

# If running on Databricks, seed historical player files from cloned repo if missing in volume
if IS_DATABRICKS:
    base_dir = os.path.dirname(__file__)
    repo_player_dir = os.path.abspath(os.path.join(base_dir, "../../../latest/player"))
    if os.path.exists(repo_player_dir):
        for y in range(2013, iter_year):
            vol_target = f"{PLAYER_DIR}/{y}_player.csv"
            repo_file = os.path.join(repo_player_dir, f"{y}_player.csv")
            if not os.path.exists(vol_target) and os.path.exists(repo_file):
                try:
                    shutil.copy2(repo_file, vol_target)
                    print(f"Seeded historical {y}_player.csv from repo to volume")
                except Exception as e:
                    print(f"Notice: unable to seed {y}_player.csv: {e}")

# Collect missing player stats from 2013 onwards, or honor SEASON_YEAR override
target_year_env = os.environ.get("SEASON_YEAR")
if target_year_env:
    iter_years = [int(target_year_env)]
else:
    iter_years = []
    for y in range(2013, iter_year + 1):
        if not os.path.exists(f"{PLAYER_DIR}/{y}_player.csv"):
            iter_years.append(y)

    if not iter_years:
        iter_years.append(iter_year)

# Load team reference list
teamlist_path = get_teamlist_path()
print(f"Loading teamlist from: {teamlist_path}")
teamcode = pd.read_csv(teamlist_path)

for iter_year in iter_years:
    print(f"Pulling player statistics for season: {iter_year}")
    playerstats = []
    # For each team
    for iter_team in list(teamcode.tricode):
        iter_sesn = str(iter_year) + str(iter_year+1)
        for iter_season_type in [2, 3]: # regular season (2), and playoff (3)
            r = safe_get(f'https://api-web.nhle.com/v1/club-stats/{iter_team}/{iter_sesn}/{iter_season_type}')
            if r is None:
                continue
            try:
                clubstats = r.json()

                if "goalies" in clubstats and len(clubstats["goalies"]) > 0:
                    temp_df = pd.json_normalize(clubstats["goalies"])
                    temp_df['team_tri'] = iter_team
                    temp_df["idx_season"] = iter_year
                    temp_df["idx_season_type"] = iter_season_type
                    temp_df["positionCode"] = "G"
                    playerstats.append(temp_df)

                if "skaters" in clubstats and len(clubstats["skaters"]) > 0:
                    temp_df = pd.json_normalize(clubstats["skaters"])
                    temp_df['team_tri'] = iter_team
                    temp_df["idx_season"] = iter_year
                    temp_df["idx_season_type"] = iter_season_type
                    playerstats.append(temp_df)
            except Exception as e:
                print(f"Error parsing club stats for {iter_team} {iter_sesn} {iter_season_type}: {e}")
        time.sleep(0.3)

    if playerstats:
        df_players = pd.concat(playerstats, ignore_index=True)
        col_remove = list(df_players.filter(regex='firstName'))
        col_remove.extend(list(df_players.filter(regex='lastName')))
        col_remove.extend(['headshot'])
        if 'firstName.default' in col_remove:
            col_remove.remove('firstName.default')
        if 'lastName.default' in col_remove:
            col_remove.remove('lastName.default')

        cols_to_drop = [c for c in col_remove if c in df_players.columns]
        df_players.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        first_cols = [c for c in ['idx_season','team_tri','playerId','firstName.default','lastName.default','positionCode','gamesPlayed'] if c in df_players.columns]
        last_cols = [col for col in df_players.columns if col not in first_cols]

        df_players = df_players[first_cols + last_cols]
        os.makedirs(PLAYER_DIR, exist_ok=True)
        output_path = f"{PLAYER_DIR}/{iter_year}_player.csv"
        df_players.to_csv(output_path, index=False)
        try:
            import config as _cfg
            from function.procs_xg import enrich_season_xg
            base_data_dir = getattr(_cfg, "BASE_DATA_DIR", os.path.dirname(PLAYER_DIR.rstrip("/")) or "./latest")
            enrich_season_xg(iter_year, base_dir=base_data_dir)
        except Exception as xg_err:
            print(f"Notice: xG enrichment skipped for {iter_year}: {xg_err}")
        print(f"Completed player stats for season {iter_year} -> {output_path}")

print("au revoir.")