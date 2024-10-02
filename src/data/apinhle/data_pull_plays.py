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

# %% Settings

# Get current date
yr_now = datetime.datetime.today().year
mo_now = datetime.datetime.today().month

# Select starting year for season to pull.
#   Until the following season starts, always pull the current/past eyar
if mo_now < 10: # Season starts on October - start with regular season since pre season games don't have full data
    # Then the season marks starts in the previous calendar year
    iter_year = yr_now - 1
else:
    iter_year = yr_now

# %% 
# Ping and pull data from NHL API
# ---------------------------------------------------

for iter_year in [iter_year]: # or iter_years

    # Pull all game lists
    gamecode = pd.read_csv(f"./latest/box/{iter_year}_box.csv")

    # Load the previous game stats, if exist
    try:
        # If previously pulled data exist
        df_playbyplay_exist = pd.read_csv(f"./latest/play/{iter_year}_playbyplay.csv")
        df_playbyplay_player_exist = pd.read_csv(f"./latest/play/{iter_year}_playbyplay_player.csv")
        # Unique of all existing game records
        gameids_exist = df_playbyplay_exist["gameid"].unique()
        # Remove the existing game records
        # Pick up games with newest data points
        gamecode = gamecode.loc[~gamecode["gameid"].isin(gameids_exist), :]
        # NOTE: This process does not account for any record revisions
        idx_exist = True
    except:
        # New data needed, no need to append the old one
        idx_exist = False

    # ---------------------------------------------------
    # Pull team/game lists of the games for the season
    '''
    Pulling full season game would take a long time. If some game records were already pulled, 
        I recommend to skip those game records
    '''
    if len(gamecode["gameid"]) != 0:
        # At least one records need to be pulled
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
            time.sleep(1)

        # Save, full data
        playbyplay = pd.concat(df_playbyplay)
        df_playerinfo = pd.concat(df_playerinfo)

        if idx_exist:
            # If the old record exists, append the old record
            playbyplay = pd.concat([df_playbyplay_exist, df_playbyplay], axis=0)
            df_playerinfo = pd.concat([df_playbyplay_player_exist, df_playerinfo], axis=0)
        
        # Save data
        playbyplay.to_csv(f"./latest/play/{iter_year}_playbyplay.csv", index=False)
        df_playerinfo.to_csv(f"./latest/play/{iter_year}_playbyplay_player.csv", index=False)
    else:
        # No records need to be pulled
        print("All records currently existing, no need to pull records")
        
print("au revoir.")