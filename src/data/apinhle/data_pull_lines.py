# %% 
# Packages

import pandas as pd
import numpy as np
import requests
import time
import datetime
# Functions to process box scores
from function.procs_boxscore import *
from function.procs_playbyplay import *
import os
from config import get_box_dir, get_play_dir, get_current_season_year

BOX_DIR = get_box_dir()
PLAY_DIR = get_play_dir()

# %% Settings

iter_year = get_current_season_year()
print(f"Iterative season: {iter_year}")

# Default to current season for daily ingestion (or set SEASON_YEAR env var for specific season)
target_year = int(os.environ.get("SEASON_YEAR", iter_year))
iter_years = [target_year]

def safe_get(url, retries=5, backoff=3.0):
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 429:
                wait_s = backoff * (2 ** attempt)
                print(f"Rate limited (429) on {url}, sleeping {wait_s:.1f}s...")
                time.sleep(wait_s)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                print(f"Failed to fetch {url} after {retries} attempts: {e}")
                return None
            time.sleep(backoff * (attempt + 1))
    return None

for iter_year in iter_years:

    # Pull all game lists
    try:
        gamecode = pd.read_csv(f"{BOX_DIR}/{iter_year}_box.csv")
    except FileNotFoundError:
        print(f"Skipping {iter_year} because {BOX_DIR}/{iter_year}_box.csv is missing!")
        continue

    # Load the previous game stats, if exist
    try:
        df_playbyplay_exist = pd.read_csv(f"{PLAY_DIR}/{iter_year}_playbyplay_shift.csv", low_memory=False)
        df_playbyplay_exist.columns = [str(c).lower() for c in df_playbyplay_exist.columns]
        idx_exist = True
    except:
        idx_exist = False
        print("No existing game records found, will pull all game records")

    if idx_exist:
        print("Found existing game records, will only pull new game records")
        gameids_exist = df_playbyplay_exist["gameid"].unique()
        gamecode = gamecode.loc[~gamecode["gameid"].isin(gameids_exist), :]
        print(f"Found {len(gamecode)} new game records, will append to the new data")

    if len(gamecode["gameid"]) != 0:
        df_playbyplay = []
        for _, row in gamecode.iterrows():
            res = safe_get(f'https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={row.gameid}')
            if res is None:
                continue
            data = res.get('data', [])
            if not data:
                print(f"Game {row.gameid} has no shift data recorded")
                time.sleep(0.5)
                continue
            iter_shift = pd.DataFrame(data)
            iter_shift.columns = [str(c).lower() for c in iter_shift.columns]
            df_playbyplay.append(iter_shift)
            print(f"Pulled game {row.gameid} shift data ({len(iter_shift)} shifts)")
            time.sleep(0.6)

        if df_playbyplay:
            new_shifts = pd.concat(df_playbyplay, ignore_index=True)
            if idx_exist:
                new_shifts = pd.concat([df_playbyplay_exist, new_shifts], ignore_index=True)
            new_shifts.to_csv(f"{PLAY_DIR}/{iter_year}_playbyplay_shift.csv", index=False)
            print(f"Saved shift data to {PLAY_DIR}/{iter_year}_playbyplay_shift.csv")
    else:
        print(f"All shift records currently up to date for season {iter_year}")

print("au revoir.")
