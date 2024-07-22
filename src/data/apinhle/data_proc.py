# %% Import context

import pandas as pd
import numpy as np
import requests
import time
# Functions to process box scores
from function.procs_boxscore import *

# %% Run a single locations
iter_year = 2023 # Start with season start


# Load the previous data
#df_player = pd.read_csv(f"./latest/{iter_year}_box_player.csv",
#                        parse_dates = ['gameDate'], 
#                        index_col = 'gameIdx')
df_box_team   = pd.read_csv(f"./latest/{iter_year}_box_team.csv",
                        parse_dates = ['gameDate'], 
                        index_col = 'gameIdx')

# %% Team success measurements
# -----------------------------------------------------
team_season = nhl_dataproc_teamsuccess(iter_year)
summary_season, summary_game = team_season.dataproc(df_box_team)

summary_game.to_csv(f"./latest/{iter_year}_box_team_game.csv")
summary_season.to_csv(f"./latest/{iter_year}_box_team_season.csv")

# %% Multiple data series
# ======================================================
# ======================================================
import json

with open("../../../../settings.json") as f:
    d = json.load(f)
# Connecting to the nhl database csv
db_nhldata = d['credentials']['server_nhl']

iter_year = 2023 # Start with season start
iter_years = np.arange(2011, iter_year + 1) # Pulling all past records

for iter_year in iter_years:
    # Load the previous data
    df_box_team  = pd.read_csv(db_nhldata + f"/box/{iter_year}_box_team.csv",
                            parse_dates = ['gameDate'], 
                            index_col = 'gameIdx')

    # Construct team success measurements, and save the data
    # ---------------------------------------------------
    team_season = nhl_dataproc_teamsuccess(iter_year)
    summary_season, summary_game = team_season.dataproc(df_box_team)

    summary_game.to_csv(db_nhldata + f"/box_game/{iter_year}_box_team_game.csv")
    summary_season.to_csv(db_nhldata + f"/box_season/{iter_year}_box_team_season.csv")