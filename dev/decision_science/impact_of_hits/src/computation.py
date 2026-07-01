# %% Emperical Estimation of Impac of Hits
import pandas as pd
import numpy as np
import os
import requests

# Get the head Github repository (go up four folders)
head_repo = os.path.abspath(os.path.join(os.getcwd(), '../../../..'))

# %% Step 0. Get the Data
# Load the dataset
# Assuming dataset is a CSV, but can be modified to load directly from your format
games = pd.read_csv(head_repo + '/latest/box/2024_box.csv')

# For each of game_id, load the play-by-play data
datadump = []
for game_id in games['gameid']:
    # Pull from the api
    str_api = f'https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play'
    # pull the data from api through request
    tempdata = requests.get(str_api)
    # Extract data
    tempdata = tempdata.json()
    # Convert play-by-play data to a DataFrame
    tempdf = pd.DataFrame(tempdata['plays'])
    # Add the game_id to the DataFrame
    tempdf['game_id'] = game_id
    # add id to the dataframe
    tempdf['id'] = tempdata['id']
    # Add season to the list
    tempdf['season'] = tempdata['season']
    # Add gameType to the list
    tempdf['gameType'] = tempdata['gameType']
    # Add limitedScoring to the list
    tempdf['limitedScoring'] = tempdata['limitedScoring']
    # Add gameDate to the list
    tempdf['gameDate'] = tempdata['gameDate']

    # expand periodDescriptor (json format) as columns
    tempdf = pd.concat([tempdf, tempdf['periodDescriptor'].apply(pd.Series)], axis=1)

    # Repeat with details
    tempdf = pd.concat([tempdf, tempdf['details'].apply(pd.Series)], axis=1)

    # REmove the json columns
    tempdf = tempdf.drop(columns=['periodDescriptor', 'details'])

    # Player information
    templayer = pd.DataFrame(tempdata['rosterSpots'])
    # Their team status, as home or away
    templayer['team'] = ['home' if iter['teamId'] == tempdata['homeTeam']['id'] else 'away' for _,iter in templayer.iterrows()]

    # For these columns, I want to identify if they are home or away team players 
    col_events_player = ['eventOwnerTeamId', 'winningPlayerId', 'hittingPlayerId', 'shootingPlayerId', 'blockingPlayerId', 'committedByPlayerId']
    col_events_team = ['eventOwnerTeam', 'FaceofWinningTeam', 'HittingTeam', 'ShootingTeam', 'BlockingTeam', 'PenaltyTeam']
    for idx,col in enumerate(col_events_player):
        # if the column exists
        if col in tempdf.columns:
            # map the team
            tempdf[col_events_team[idx]] = tempdf[col].map(templayer.set_index('playerId')['team'])
        else:
            tempdf[col_events_team[idx]] = np.nan
    
    # If descKey and duration not already in, create nan
    if 'descKey' not in tempdf.columns:
        tempdf['descKey'] = np.nan
    if 'duration' not in tempdf.columns:
        tempdf['duration'] = np.nan

    # Only keep these columns
    tempdf = tempdf[
        [
            'game_id',  'season', 'gameType', 'gameDate',
            'eventId', 'periodType', 'maxRegulationPeriods', 'sortOrder', 'timeInPeriod', 'timeRemaining', 
            'situationCode', 'typeDescKey',
            'homeTeamDefendingSide', 'xCoord', 'yCoord', 'zoneCode',
            'homeScore', 'awayScore', 
            'homeSOG', 'awaySOG',
            'reason',
            'shotType', 'ShootingTeam', 'BlockingTeam',
            'FaceofWinningTeam',
            'HittingTeam', 
            'PenaltyTeam', 'descKey', 'duration'
        ]
    ]
    # Append to the list
    datadump.append(tempdf)

# Combine all the data into a single DataFrame
data = pd.concat(datadump, ignore_index=True)

# Save the data
data.to_csv(head_repo + '/latest/play/2024_playbyplay.csv', index=False)

# %% Step 1. Data Preprocessing
# Step 1: Calculate score differential and cumulative hits
# Creating a new column for score differential
data['score_differential'] = data['team_score'] - data['opponent_score']

# Step 2: Track the cumulative hits over time
# This is assuming the 'hits' column in the data is the number of hits per event
data['cumulative_hits'] = data.groupby('game_id')['hits'].cumsum()  # 'game_id' should uniquely identify each game

# Step 3: Normalize time remaining within each period if needed
# Here, we're just using time_remaining as continuous, but can normalize if you prefer
# e.g., Normalizing time as percentage of period remaining (optional):
# data['normalized_time'] = data['time_remaining'] / data['period_duration']

# Step 4: Filter data by periods (1st, 2nd, 3rd, overtime)
# Separate the dataset by periods
periods = {
    "period_1": data[data['period'] == 1],
    "period_2": data[data['period'] == 2],
    "period_3": data[data['period'] == 3],
    "overtime": data[data['period'] == 'OT']  # Assuming 'OT' for overtime, adjust as needed
}

# Optional: Verify the transformations
print(periods['period_1'].head())
print(periods['period_2'].head())

# %%
