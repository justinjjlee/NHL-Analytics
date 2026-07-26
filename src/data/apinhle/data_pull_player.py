# %% 
# Packages

import pandas as pd
import numpy as np
import requests
import time
import datetime
# Functions to process box scores
from function.procs_boxscore import *
import os
from config import get_team_dir

TEAM_DIR = get_team_dir()


# %% 
# Get current date
yr_now = datetime.datetime.today().year
mo_now = datetime.datetime.today().month
dy_now = datetime.datetime.today().day

# Select starting year for season to pull.
#   Until the following season starts, always pull the current/past eyar
if (mo_now > 10) | ((mo_now == 10) & (dy_now > 15)): 
    # Season starts on October - start with regular season since pre season games don't have full data
    #   Start the regular season data pull on 10/15
    # Then the season marks starts in the previous calendar year
    iter_year = yr_now
else:
    # iter year starts from previous 
    iter_year = yr_now - 1

print(f"Iterative season: {iter_year}")
from config import get_box_dir

BOX_DIR = get_box_dir()

# Collect missing player stats from 2013 onwards
iter_years = []
for y in range(2013, iter_year + 1):
    if not (os.path.exists(f"{BOX_DIR}/{y}_box_player.csv") or os.path.exists(f"{TEAM_DIR}/{y}_player.csv")):
        iter_years.append(y)

if not iter_years:
    iter_years.append(iter_year)

teamcode = pd.read_csv(f"{TEAM_DIR}/teamlist.csv")

for iter_year in iter_years:
    print(f"Pulling player statistics for season: {iter_year}")
    playerstats = []
    # For each team
    for iter_team in list(teamcode.tricode):
        iter_sesn = str(iter_year) + str(iter_year+1)
        for iter_season_type in [2, 3]: # regular season (2), and playoff (3)
            try:
                r = requests.get(url=f'https://api-web.nhle.com/v1/club-stats/{iter_team}/{iter_sesn}/{iter_season_type}')
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
            except Exception:
                pass
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
        df_players.to_csv(f"{BOX_DIR}/{iter_year}_box_player.csv", index=False)
        print(f"Completed player stats for season {iter_year}")

print("au revoir.")