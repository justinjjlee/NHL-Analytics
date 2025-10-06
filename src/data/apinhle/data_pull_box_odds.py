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
import json 

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


# For each Canada and United States market
for iter_country in ['US', 'CA']:

    r = requests.get(url=f'https://api-web.nhle.com/v1/partner-game/{iter_country}/now')
    data = r.json()

    # %% Process data
    main_df = pd.json_normalize(data['games'])

    # Run if the data is not empty (main_df is not empty)
    if not main_df.empty:
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
        home_odds_df.rename(columns=
            {
                "home_odds_description":"odds_description"
            }, inplace=True
        )

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
        away_odds_df.rename(columns=
            {
                "awayTeam_gameId":"gameId",
                "away_odds_description":"odds_description"
            }, inplace=True
        )

        # Merge the home and away odds into the main dataframe
        final_df = pd.merge(
            main_df, home_odds_df, 
            left_on='gameId', right_on='homeTeam_gameId', 
            suffixes=('', '_home')
            ).merge(
                away_odds_df, 
                on = ['gameId', 'odds_description'], 
                suffixes=('', '_away')
            )

        # Drop duplicate columns
        final_df = final_df.loc[:,~final_df.columns.duplicated()]

        # Append additional information
        final_df["currentOddsDate"] = data["currentOddsDate"]
        final_df["currentOddsDate"] = data["currentOddsDate"]
        final_df["lastUpdatedUTC"] = data["lastUpdatedUTC"]
        final_df["bettingPartner"] = data["bettingPartner"]["name"]

        # Finalize data selection
        final_df = final_df[[
            'gameType', 'gameId', 
            'homeTeam.id', 'homeTeam.abbrev', 
            'awayTeam.id', 'awayTeam.abbrev',
            'odds_description',	
            'home_odds_value', 'away_odds_value',
            'bettingPartner',
            'startTimeUTC', 'lastUpdatedUTC'
        ]]

        # Save data
        # lower-case column name gameId to gameid
        final_df.rename(columns={'gameId':'gameid'}, inplace=True)

        try:
            # Load existing real-time tracking file
            exist_df = pd.read_csv(f"./latest/box/{iter_year}_odds_{iter_country}.csv")
            # Load json file 
            with open(f"./latest/box/{iter_year}_odds_{iter_country}.json", "r") as f:
                exist_json = json.load(f)
        except: 
            # new season starts
            exist_df = final_df
            exist_json = []

        # Append - dataframe
        df = pd.concat([exist_df, final_df], axis= 0).reset_index(drop=True)
        # Append the latest available data and replace the existing record
        # Sort by gameId and startTimeUTC in descending order and drop duplicates based on gameId, keeping the last (latest) record
        df = df\
            .sort_values(by=['gameid', 'lastUpdatedUTC', 'odds_description'], ascending=[True, False, False])\
                .drop_duplicates(subset=['gameid', 'odds_description', 'bettingPartner'], keep='first')

        # Append - json
        for iter_game in data['games']:
            exist_json.append(iter_game)

        # Save the data
        df.to_csv(f"./latest/box/{iter_year}_odds_{iter_country}.csv", index=False)
        # Save the json data as is
        with open(f"./latest/box/{iter_year}_odds_{iter_country}.json", "w") as f:
            json.dump(exist_json, f)

        print(f"Odds data for {iter_country} on {datetime.datetime.today()} saved successfully.")
    else:
        print("No data available at this time.")
print("au revoir.")
