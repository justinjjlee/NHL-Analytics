# %% 
# Packages

import pandas as pd
import numpy as np
import requests
import time
import datetime
# Functions to process box scores
from function.procs_boxscore import *


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
# Ping and pull data from NHL API
# ---------------------------------------------------
# Team codes for the list pull
teamcode = pd.read_csv("./latest/team/teamlist.csv")

# %%
# Player data 
# ===================================================
# Player data download - season aggregation, up-to-date
playerstats = []
# For each team
for iter_team in list(teamcode.tricode):
    # Pull seasons played by each  team
    r = requests.get(url=f'https://api-web.nhle.com/v1/roster-season/{iter_team}')
    seasons = r.json()

    # For season download: 
    iter_sesn = str(iter_year) + str(iter_year+1)
    for iter_season_type in [2, 3]: # pre season (1), regular season (2), and playoff (3) - if available
        try:
            # Pull club stats, only regular seasons for now
            r = requests.get(url=f'https://api-web.nhle.com/v1/club-stats/{iter_team}/{iter_sesn}/{iter_season_type}')
            clubstats = r.json()

            temp_df = pd.json_normalize(clubstats["goalies"])
            temp_df['team_tri'] = iter_team
            temp_df["idx_season"] = iter_year
            temp_df["idx_season_type"] = iter_season_type
            temp_df["positionCode"] = "G"
            playerstats.append(temp_df)
            temp_df = pd.json_normalize(clubstats["skaters"])
            temp_df['team_tri'] = iter_team
            temp_df["idx_season"] = iter_year
            playerstats.append(temp_df)
        except:
            None
    print(f"{iter_team} completed ...")
    # Pause to play safe with the API
    time.sleep(1)
# %%
# Data concatenation
playerstats = pd.concat(playerstats)
# Aggregate and concatenate
col_remove = list(playerstats.filter(regex='firstName'))
col_remove.extend(list(playerstats.filter(regex='lastName')))
col_remove.extend(['headshot'])
col_remove.remove('firstName.default')
col_remove.remove('lastName.default')

playerstats.drop(columns=col_remove, inplace=True)
# %%
# Column re-order
first_cols = ['idx_season','team_tri','playerId','firstName.default','lastName.default','positionCode','gamesPlayed']
last_cols = [col for col in playerstats.columns if col not in first_cols]

# save sata
playerstats = playerstats[first_cols+last_cols]
playerstats.to_csv(f"./latest/player/{iter_year}_player.csv", index=False)

print("au revoir.")