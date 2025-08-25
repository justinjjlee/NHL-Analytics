# %% Data Pull 

import pandas as pd
import numpy as np
import requests
import time
import datetime
import seaborn as sns
import matplotlib.pyplot as plt

seasons_glory = [20092010, 20122013, 20142015] # Blackhawks' three most recent championships
seasons_ohboy = [20232024, 20242025] # Bedard's first two seasons

idx_season = 2 # pull regular season data
iter_team = "CHI" # Chicago Blackhawks
# %% Pull data
playerstats = []

for iter_sesn in seasons_glory + seasons_ohboy:
    iter_year = str(iter_sesn)[:4] + ' - ' + str(iter_sesn)[4:] # Extract the year from the season code
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
            temp_df["idx_season_type"] = iter_season_type
            temp_df["idx_season"] = iter_year
            playerstats.append(temp_df)
        except:
            None
# %% Clean up data 
chi_players = pd.concat(playerstats)

# any firstNmae and lastName derivation except .default extension is removed
col_remove = list(chi_players.filter(regex='firstName'))
col_remove.extend(list(chi_players.filter(regex='lastName')))
col_remove.extend(['headshot'])
col_remove.remove('firstName.default')
col_remove.remove('lastName.default')
chi_players.drop(columns=col_remove, inplace=True)

# Save data
chi_players.to_csv('./data/chi_players.csv', index=False)

# %%
