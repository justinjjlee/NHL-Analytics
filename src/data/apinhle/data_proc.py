# %% 
# Packages

import pandas as pd
import numpy as np
import requests
import time
# Functions to process box scores
from function.procs_boxscore import *


# %% 
iter_year = 2023 # Start with season start


# Load the previous data
#df_player = pd.read_csv(f"./latest/{iter_year}_box_player.csv",
#                        parse_dates = ['gameDate'], 
#                        index_col = 'gameIdx')
df_box_team   = pd.read_csv(f"./latest/{iter_year}_box_team.csv",
                        parse_dates = ['gameDate'], 
                        index_col = 'gameIdx')

# %%
# Construct team success measurements, and save the data
# ---------------------------------------------------
team_season = nhl_dataproc_teamsuccess(iter_year)
team_season.dataproc(df_box_team)