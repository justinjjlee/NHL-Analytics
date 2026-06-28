# %%
'''
    General note

    For the scope of this analysis, I ignore goalies and only focus on skaters
'''
import pandas as pd
import numpy as np
import requests
import os
import time
# ==============================================
# %% Import Draft List 
'''
    NHL Draft List

    Once download, no need to re-run

'''
# Define list of years, from 2000 to 2025 (latest)
# Get the current date year
import datetime
end_year = datetime.datetime.now().year
years = list(range(2000, end_year + 1))

# Create the data directory if it doesn't exist
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
os.makedirs(data_dir, exist_ok=True)

# Dictionary to store DataFrames for each year
drafted_players_raw = {}
drafted_players_df = {}  # Will store processed DataFrames

# Load existing draft data if it exists
draft_file_path = os.path.join(data_dir, 'all_drafted_players.csv')
if os.path.exists(draft_file_path):
    existing_draft_df = pd.read_csv(draft_file_path)
    existing_draft_years = existing_draft_df['year'].unique().tolist()
    print(f"Loaded existing draft data: {len(existing_draft_df)} players from {len(existing_draft_years)} years.")
else:
    existing_draft_df = pd.DataFrame()
    existing_draft_years = []

# Define the specific columns we want to extract
columns_to_extract = [
    'round', 
    'pickInRound', 
    'overallPick', 
    'teamAbbrev',
    'firstName.default', 
    'lastName.default',
    'positionCode',
    'countryCode',
    'height',
    'weight',
    'amateurLeague',
    'amateurClubName'
]

for year in years:
    if year in existing_draft_years:
        print(f"Draft data for {year} already exists, skipping API call.")
        continue

    # Get the JSON data from the API
    url = f"https://api-web.nhle.com/v1/draft/picks/{year}/all"
    
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()
            drafted_players_raw[year] = data
            
            # Extract the picks list from the response
            picks = data.get('picks', [])
            
            # Create a DataFrame from the picks list
            df = pd.json_normalize(picks)
            
            # Create a more concise DataFrame with only the columns we want
            selected_df = pd.DataFrame()
            selected_df['year'] = [year] * len(df)
            selected_df['round'] = df['round']
            selected_df['pickInRound'] = df['pickInRound']
            selected_df['overallPick'] = df['overallPick']
            selected_df['teamAbbrev'] = df['teamAbbrev']
            selected_df['firstName'] = df['firstName.default']
            selected_df['lastName'] = df['lastName.default']
            selected_df['position'] = df['positionCode']
            selected_df['country'] = df['countryCode']
            selected_df['height'] = df['height']
            selected_df['weight'] = df['weight']
            selected_df['amateurLeague'] = df['amateurLeague']
            selected_df['amateurClubName'] = df['amateurClubName']
            
            # Store the processed DataFrame
            drafted_players_df[year] = selected_df
            
            print(f"Processed {year} draft data: {len(df)} players")
        else:
            print(f"Error {response.status_code} for year {year}: {response.reason}")
    except Exception as e:
        print(f"Error processing year {year}: {str(e)}")

# Create a combined DataFrame with all years
if drafted_players_df:
    new_draft_data = pd.concat(drafted_players_df.values(), ignore_index=True)
    all_drafted_players = pd.concat([existing_draft_df, new_draft_data], ignore_index=True)
    # Save the combined DataFrame to the data directory
    all_drafted_players.to_csv(draft_file_path, index=False)
    print(f"Added {len(new_draft_data)} new players.")
else:
    all_drafted_players = existing_draft_df
    print("No new draft data to add.")
'''
# Save individual year DataFrames if needed
for year, df in drafted_players_df.items():
    df.to_csv(os.path.join(data_dir, f'drafted_players_{year}.csv'), index=False)
'''
print(f"Total players in the combined dataset: {len(all_drafted_players)}")
print(f"Data saved to: {data_dir}")

# ==============================================
# %% Pull the player data off of the draftee
'''
Currently, the best way to pull all players stats of season is to pull stats leader for each season. 
I am using points stats leaders (given I will be using as a proxy for player contribution)

I evaluate seasons from 2006 to 2024
'''

# Create season mark for API: 2006-07 season is written as 20062007
# Define the seasons to pull
seasons = [f"{year}{year + 1}" for year in range(2000, end_year + 1)]

'''
This is the API block to pull the stats for players
https://api-web.nhle.com/v1/skater-stats-leaders/20222023/2?categories=points&limit=-1

2 represents the season type (2 = regular season, 3 = playoffs)
categories=points is the category of stats we want to pull
limit=-1 means we want to pull all players
'''

# Create a dictionary to store the player stats for each season
player_stats = {}
kpi_stats = 'points'  # Key Performance Indicator (KPI) for player stats

stats_file_path = os.path.join(data_dir, 'all_player_stats.csv')
if os.path.exists(stats_file_path):
    existing_stats_df = pd.read_csv(stats_file_path)
    existing_seasons = existing_stats_df['season'].astype(str).unique().tolist()
    print(f"Loaded existing player stats data: {len(existing_stats_df)} records from {len(existing_seasons)} seasons.")
else:
    existing_stats_df = pd.DataFrame()
    existing_seasons = []

# Loop through each season and pull the stats
for season in seasons:
    if str(season) in existing_seasons:
        print(f"Player stats for season {season} already exist, skipping API call.")
        continue
    # Construct the URL for the API request
    url = f"https://api-web.nhle.com/v1/skater-stats-leaders/{season}/2?categories=points&limit=-1"
    
    try:
        # Make the API request
        response = requests.get(url)
        
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()
            
            # Extract the players list from the response
            players = data.get(kpi_stats, [])
            
            # Create a DataFrame from the players list
            df = pd.json_normalize(players)
            
            # Create a more concise DataFrame with only the columns we want
            selected_df = pd.DataFrame()
            
            # Select only the columns we want
            selected_df['id'] = df['id']
            selected_df['firstName'] = df['firstName.default']
            selected_df['lastName'] = df['lastName.default']
            selected_df['teamAbbrev'] = df['teamAbbrev']
            selected_df['position'] = df['position']
            selected_df['points'] = df['value']  # Rename 'value' to 'points'
            selected_df['season'] = season  # Add season information
            
            # Store the DataFrame in the dictionary
            player_stats[season] = selected_df
            
            print(f"Processed {season} player stats: {len(df)} players")
        else:
            print(f"Error {response.status_code} for season {season}: {response.reason}")
    except Exception as e:
        print(f"Error processing season {season}: {str(e)}")

# Create a combined DataFrame with all seasons' player stats
if player_stats:
    new_player_stats = pd.concat(player_stats.values(), ignore_index=True)
    all_player_stats = pd.concat([existing_stats_df, new_player_stats], ignore_index=True)
    # Save the combined DataFrame to the data directory
    all_player_stats.to_csv(stats_file_path, index=False)
    print(f"Added {len(new_player_stats)} new player stats records.")
else:
    all_player_stats = existing_stats_df
    print("No new player stats data to add.")

print(f"Total player stats records: {len(all_player_stats)}")

# ==============================================
# %% Load and match
'''
    Load the two data sources and match based on first and last name
    right: all_player_stats
    left: all_drafted_players
    Join on first and last name
    right join affix should be _stats
    left join affix should be _draft
'''
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
os.makedirs(data_dir, exist_ok=True)
# Load the drafted players data
drafted_players = pd.read_csv(os.path.join(data_dir, 'all_drafted_players.csv'))
# Load the player stats data
player_stats = pd.read_csv(os.path.join(data_dir, 'all_player_stats.csv'))
# Merge the two DataFrames on first and last name
merged_data = pd.merge(
    drafted_players,
    player_stats,
    left_on=['firstName', 'lastName'],
    right_on=['firstName', 'lastName'],
    how='left',
    suffixes=('_draft', '_stats')
)

# id and season are integers
merged_data['id'] = merged_data['id'].astype('Int64')
merged_data['season'] = merged_data['season'].astype('Int64')

# Drop null id or season (no NHL record, for now)
merged_data = merged_data[~(merged_data['id'].isnull() | merged_data['season'].isnull())]

# Save the merged data to a CSV file
merged_data.to_csv(os.path.join(data_dir, 'merged_draft_player_stats.csv'), index=False)
print(f"Merged data saved to: {os.path.join(data_dir, 'merged_draft_player_stats.csv')}")

# For the merged data, how many are not matched with player data
# check the null value of columns from player_stats: null for column id
# count
unmatched = merged_data[merged_data['id'].isnull()]
print(f"Number of unmatched players: {len(unmatched)}")
# ==============================================
# %% Build Player Chronicle
'''
    Once we match player ID,

    We track player performance over their career (NHL and other leagues)

    https://api-web.nhle.com/v1/player/8478402/landing
'''

# Get unique player IDs, but ONLY for 1st round picks to save API time
# since the exceptional seasons analysis only looks at 1st round picks.
first_round_ids = merged_data[merged_data['round'] == 1]['id'].unique()
# exclude null
first_round_ids = first_round_ids[~pd.isnull(first_round_ids)]
# conver to integer
unique_player_ids = first_round_ids.astype(int)

# For each player ID, pull the player stats
player_chronicle = {}

chronicle_file_path = os.path.join(data_dir, 'all_drafted_players_chronicle.csv')
if os.path.exists(chronicle_file_path):
    existing_chronicle_df = pd.read_csv(chronicle_file_path)
    existing_chronicle_ids = existing_chronicle_df['id'].unique().tolist()
    print(f"Loaded existing player chronicles: {len(existing_chronicle_df)} records for {len(existing_chronicle_ids)} players.")
else:
    existing_chronicle_df = pd.DataFrame()
    existing_chronicle_ids = []

for player_id in unique_player_ids:
    if player_id in existing_chronicle_ids:
        continue

    # Construct the URL for the API request
    url = f"https://api-web.nhle.com/v1/player/{player_id}/landing"
    
    try:
        # Make the API request
        response = requests.get(url)
        
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()
            
            # Extract the player stats from the response
            player_data = data.get('seasonTotals', [])
            
            # Create a DataFrame from the player data
            df = pd.json_normalize(player_data)
            # Add player ID
            df['id'] = player_id
            # Store the DataFrame in the dictionary
            player_chronicle[player_id] = df
            
            print(f"Processed player ID {player_id}: {len(df)} records")
        else:
            print(f"Error {response.status_code} for player ID {player_id}: {response.reason}")
    except Exception as e:
        print(f"Error processing player ID {player_id}: {str(e)}")
    
    # Print % processed
    if len(player_chronicle) % 10 == 0:
        print(f"Processed {len(player_chronicle)} player IDs out of {len(unique_player_ids)}")

    # Wait a short duration to provide a buffer for the API
    time.sleep(0.05)

if player_chronicle:
    new_chronicle_df = pd.concat(player_chronicle.values(), ignore_index=True)
    
    # Some clean up needing. 
    # for teamname (teamName or teamNameCommon) keep teamName.default and drop others
    cols_to_drop = [
        'teamCommonName.cs', 'teamCommonName.de', 'teamCommonName.sk',
        'teamCommonName.sv', 'teamName.cs', 'teamName.de', 'teamName.fi',
        'teamName.sk', 'teamName.sv','teamName.fr', 'teamPlaceNameWithPreposition.fr',
        'teamCommonName.default', 'teamCommonName.es',
        'teamCommonName.fi', 'teamCommonName.fr',
        'teamPlaceNameWithPreposition.default'
    ]
    
    existing_drop_cols = [c for c in cols_to_drop if c in new_chronicle_df.columns]
    if existing_drop_cols:
        new_chronicle_df = new_chronicle_df.drop(columns=existing_drop_cols)
        
    all_player_chronicle = pd.concat([existing_chronicle_df, new_chronicle_df], ignore_index=True)
    # Save the combined DataFrame to the data directory
    all_player_chronicle.to_csv(chronicle_file_path, index=False)
    print(f"Added {len(new_chronicle_df)} new player chronicle records.")
else:
    all_player_chronicle = existing_chronicle_df
    print("No new player chronicle data to add.")
# %%
