# %% 
# Packages

import pandas as pd
import numpy as np
import requests
import time
# Functions to process box scores
from function.procs_boxscore import *
from function.procs_playbyplay import *

# %% 
# Ping and pull data from NHL API
# ---------------------------------------------------
iter_year = 2023 # Start with the current season

# Pull all game lists
gamecode = pd.read_csv(f"./latest/{iter_year}_gamelist.csv")

# %%
# ---------------------------------------------------
# Pull team/game lists of the games for the season
'''
Pulling full season game would take a long time. If some game records were already pulled, 
    I recommend to skip those game records
'''
df_playbyplay = []
df_playerinfo = []
for _, row in gamecode.iterrows():
    # Pull game's play-by-play stat
    r = requests.get(url=f'https://api-web.nhle.com/v1/gamecenter/{row.gameid}/play-by-play')
    
    iter_playbyplay, iter_player = proc_playbyplay_clean(r, row)
    # Append to save
    df_playbyplay.append(iter_playbyplay)
    df_playerinfo.append(iter_player)
    # Pause to play safe with the API
    #time.sleep(1)

# %%
# Save, full data
playbyplay = pd.concat(df_playbyplay)
playbyplay.to_csv(f"./latest/{iter_year}_playbyplay.csv", index=False)

df_playerinfo = pd.concat(df_playerinfo)
df_playerinfo.to_csv(f"./latest/{iter_year}_playbyplay_player.csv", index=False)
