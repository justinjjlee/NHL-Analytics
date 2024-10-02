# %%
'''
    Pull odds data provided by api-web.nhle.com

    https://api-web.nhle.com/v1/partner-game/US/now

    The data is only available in real time, thus require constant tracking

    NOTE:
        Depending on when you pull the data, the odds may change
'''

import pandas as pd
import numpy as np
import requests
import time
import datetime

# Get current date
yr_now = datetime.datetime.today().year
mo_now = datetime.datetime.today().month

# Select starting year for season to pull.
#   Until the following season starts, always pull the current/past eyar
if mo_now < 8: # Season starts on October, but pre-season results can be saved. for future reference
    # Then the season marks starts in the previous calendar year
    iter_year = yr_now - 1
else:
    iter_year = yr_now
# %% Pull data 
r = requests.get(url='https://api-web.nhle.com/v1/partner-game/US/now')
data = r.json()

# %% Process data
main_df = pd.json_normalize(data['games'])

# Normalize the nested 'odds' for homeTeam
home_odds_df = pd.json_normalize(
    data['games'], record_path=['homeTeam', 'odds'], 
    meta=[
        'gameId', 'gameType', 'startTimeUTC', 
        ['homeTeam', 'id'], ['homeTeam', 'name', 'default'], 
        ['homeTeam', 'abbrev'], ['homeTeam', 'logo']
    ],
    meta_prefix='homeTeam_', record_prefix='home_odds_'
)

# Keep specific columns
home_odds_df = home_odds_df[["homeTeam_gameId", "home_odds_description", "home_odds_value"]]

# Normalize the nested 'odds' for awayTeam
away_odds_df = pd.json_normalize(
    data['games'], record_path=['awayTeam', 'odds'], 
    meta=[
        'gameId', 'gameType', 'startTimeUTC', 
        ['awayTeam', 'id'], ['awayTeam', 'name', 'default'], 
        ['awayTeam', 'abbrev'], ['awayTeam', 'logo']
    ],
    meta_prefix='awayTeam_', record_prefix='away_odds_'
)
away_odds_df = away_odds_df[["awayTeam_gameId", "away_odds_description", "away_odds_value"]]

# Merge the home and away odds into the main dataframe
final_df = pd.merge(
    main_df, home_odds_df, 
    left_on='gameId', right_on='homeTeam_gameId', 
    suffixes=('', '_home')
    ).merge(
        away_odds_df, 
        left_on='gameId', right_on='awayTeam_gameId', 
        suffixes=('', '_away')
    )

# Drop duplicate columns
final_df = final_df.loc[:,~final_df.columns.duplicated()]
final_df.drop(
    columns=[
        "homeTeam.odds", "awayTeam.odds",
        "homeTeam_gameId", "awayTeam_gameId"
    ], inplace=True
)

# Append additional information
final_df["currentOddsDate"] = data["currentOddsDate"]
final_df["currentOddsDate"] = data["currentOddsDate"]
final_df["lastUpdatedUTC"] = data["lastUpdatedUTC"]
final_df["bettingPartner"] = data["bettingPartner"]["name"]

# %% Save data

try:
    # Load existing real-time tracking file
    exist_df = pd.read_csv(f"./latest/box/{iter_year}_box_odds.csv")
except: 
    # new season starts
    exist_df = final_df
# Append
df = pd.concat([exist_df, final_df], axis= 0).reset_index(drop=True)
# Append the latest available data and replace the existing record
# Sort by gameId and startTimeUTC in descending order and drop duplicates based on gameId, keeping the last (latest) record
df = df\
    .sort_values(by=['gameId', 'lastUpdatedUTC'], ascending=[True, False])\
        .drop_duplicates(subset=['gameId'], keep='first')

# Save the data
df.to_csv(f"./latest/box/{iter_year}_box_odds.csv", index=False)

print("Good bye.")