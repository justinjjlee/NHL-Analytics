# Databricks notebook source
import pandas as pd
import numpy as np
import requests
import time
import datetime

# Get current date
yr_now = datetime.datetime.today().year
mo_now = datetime.datetime.today().month
dy_now = datetime.datetime.today().day

# COMMAND ----------

# MAGIC %md
# MAGIC Check available game file names

# COMMAND ----------

files = dbutils.fs.ls("/Volumes/nhl-databricks/data/box/")
file_names = [f.name for f in files]
display(pd.DataFrame(file_names, columns=["file_name"]))

# COMMAND ----------

# %% 
# Ping and pull data from NHL API
# ---------------------------------------------------

for iter_file in file_names[13:]: # or iter_years

    iter_year = iter_file[0:4]
    # Pull all game lists
    gamecode = pd.read_csv(f"/Volumes/nhl-databricks/data/box/{iter_file}")

    # Load the previous game stats, if exist
    try:
        # If previously pulled data exist
        df_playbyplay_exist = pd.read_csv(f"./latest/play/{iter_year}_playbyplay_shift.csv")
        # NOTE: This process does not account for any record revisions
        idx_exist = True
    except:
        # New data needed, no need to append the old one
        idx_exist = False
        print("No existing game records found, will pull all game records")

    if idx_exist: # If the current season data exist
        # Don't need to pull all game records
        print("Found existing game records, will only pull new game records")
        # Unique of all existing game records
        gameids_exist = df_playbyplay_exist["gameid"].unique()
        # Remove the existing game records
        # Pick up games with newest data points
        gamecode = gamecode.loc[~gamecode["gameid"].isin(gameids_exist), :]
        print(f"Found {len(gamecode)} new game records, will append to the new data")
    # ---------------------------------------------------
    # Pull team/game lists of the games for the season
    '''
    Pulling full season game would take a long time. If some game records were already pulled, 
        I recommend to skip those game records
    '''
    if len(gamecode["gameid"]) != 0:
        # At least one records need to be pulled
        df_playbyplay = []
        for _, row in gamecode.iterrows():
            # Pull game's play-by-play stat
            r = requests.get(url=f'https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={row.gameid}')
            # Pretty simple data form
            iter_shift = pd.DataFrame(r.json()['data'])
            # Append to save
            df_playbyplay.append(iter_shift)
            print(f"Pulled game {row.gameid} shift data")
            # Pause to play safe with the API
            time.sleep(1)

        # Save, full data
        df_playbyplay = pd.concat(df_playbyplay)

        if idx_exist:
            # If the old record exists, append the old record
            df_playbyplay = pd.concat([df_playbyplay_exist, df_playbyplay], axis=0)
        
        # Save data
        df_playbyplay.to_csv(f"/Volumes/nhl-databricks/data/shift/{iter_year}_playbyplay_shift.csv", index=False)
    else:
        # No records need to be pulled
        print("All records currently existing, no need to pull records")
        
print("au revoir.")
# %%

