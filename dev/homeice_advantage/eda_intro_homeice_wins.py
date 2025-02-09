# %% EDA of home ice advantage - The Intro
'''
    Introductory analysis on how the home ice advantages would impact team performance.
'''

# %% Import packages and data
import pandas as pd
import matplotlib.pyplot as plt
import os 

# Set Directory - currently @ this script file
dir_cwd = os.getcwd()
dir_git = os.path.dirname(os.path.dirname(dir_cwd))

# Import distance between the stadium
df_distance = pd.read_csv(
    dir_git + '/latest/team/teamlist_locations_distance_meters.csv'
)

# Team performance, the lastest (2024 - 2025 season at the time of this first script)
df_performance = pd.read_csv(
    dir_git + '/latest/box/2024_box.csv'
)

# FOR LOCAL USE ONLY
# Add all other older data from 2021 to 2023
for iter in range(2021, 2024):
    df_performance = pd.concat(
        [
            df_performance,
            pd.read_csv(
                dir_git + f'/latest/box/{iter}_box.csv'
            )
        ]
    )
# %% Data Preparation
'''
    We want to know the performance of away team when traveling
'''
tab_dist = pd.merge(
    df_performance, df_distance,
    left_on=['tricode_for', 'tricode_against'],
    right_on=['tricode_home', 'tricode_away'],
    how='left'
)

tab_dist['idx_awaywin'] = tab_dist['tricode_winteam'] == tab_dist['tricode_away']
# Create quantiles and label them
tab_dist['distance_quantile'] = pd.qcut(tab_dist['geo_distance_haversine'], q=4, labels=[1, 2, 3, 4])

''' 
    Additional data points
        - Consecutive traveling game
            - The opponent location and travel distance
            - Duration of the away games played
'''

tab_dist.set_index('gameid', inplace=True)

# %% Calculate sequence of home/away games
#

# For each away team, count their consecutive away games
aggdf = []
for iter_teams in tab_dist.tricode_for.unique():
    # filter games that the team is playing, either home or away
    tab_games = tab_dist[(tab_dist.tricode_for == iter_teams) | (tab_dist.tricode_against == iter_teams)].copy()
    tab_games.reset_index(inplace=True)
    # Sort by day of the game
    tab_games.sort_values(by='date', inplace=True)
    # For each date, calculate the number of consecutive games that team is appearing as away team
    tab_games['is_away'] = tab_games['tricode_away'] == iter_teams
    # cumulate sum of is_away
    tab_games['away_consecutive'] = tab_games.groupby('tricode_away').cumcount() + 1
    # Tracking total distance travel for away game
    tab_games['away_distance'] = 0.0

    # Need to start correcting,
    # If the team is playing at home, the conesecutive value is zero
    tab_games.loc[tab_games['tricode_for'] == iter_teams, 'away_consecutive'] = 0
    # check each row, and if away_consecutive is same as previous row, subtrack own and add 1 for all values below
    #tab_games['away_consecutive'] = tab_games['away_consecutive'].diff().fillna(0)

    for index, values in tab_games.iterrows():
        # If the preivous row value 'away_conesecutive' is zero and current row is 1 or more,
        # then the current row should be reset to 1 and all the subsequent rows with non-zero away_consecutive value
        #   subtracted by the current row value
        if index > 0 and tab_games.loc[index-1, 'is_away'] == False and values['away_consecutive'] > 0:
            tab_games.loc[index:, 'away_consecutive'] -= tab_games.loc[index, 'away_consecutive']

        # If away game and its first of the sequence, then define away distance to be distance traveled
        if values['is_away'] and values['away_consecutive'] == 1:
            tab_games.loc[index, 'away_distance'] = values['geo_distance_haversine']
        # If away game sequence is greater than 1, then add the distance traveled to the previous game
        elif index > 0 and values['is_away'] and values['away_consecutive'] > 1:
            tab_games.loc[index, 'away_distance'] = values['geo_distance_haversine'] + tab_games.loc[index-1, 'away_distance']
    
    # Polish the metric
    # Only keep the values with away games
    #   This way, no worry for duplication due to home/away duplication pull
    tab_games = tab_games[tab_games['is_away']]
    # Add 1 to transform as count not index
    tab_games['away_consecutive'] += 1    
    # append in aggdf
    aggdf.append(tab_games)

df_consequa = pd.concat(aggdf).set_index('gameid')
df_consequa = df_consequa[['away_consecutive', 'away_distance']]

# Join with original file
tab_dist = tab_dist.join(df_consequa)

# %% EDA: Validation if the consecutive away games are calculated correctly
'''
    Validation
    DET in Jan/Feb 2025 Did four-game road sweep for the first time since March 1996
'''
# Select only DET away game since January of 2025
eda_validation_DET2025 = tab_dist[
    (tab_dist['tricode_against'] == 'DET') & 
    (tab_dist['date'] >= '2025-01-30') & 
    (tab_dist['date'] <= '2025-02-05')
]
eda_validation_DET2025

# %% EDA: Plot the outcome trend of how teams have faired in consecuitive away games
eda_distance_traveled_teams = tab_dist.groupby([ 'away_consecutive', 'tricode_away'])\
    .agg(
        {
            'idx_awaywin':'mean',
            'period_ending':'count'
        }
    ).reset_index()

# Remove any games over 5 consecutive games (all-star or 4-nation faceoff breaks)
eda_distance_traveled_teams = eda_distance_traveled_teams[eda_distance_traveled_teams['away_consecutive'] < 5]

# Create index for away_consecutive three or greater as three
eda_distance_traveled_teams['away_consecutive_indx'] = eda_distance_traveled_teams['away_consecutive'].apply(lambda x: 4 if x >= 4 else x)

# Set figure and axis
fig, ax = plt.subplots(figsize=(8, 4.5))

# Create the box plot with a specified axis
eda_distance_traveled_teams.boxplot(column='idx_awaywin', by='away_consecutive_indx', ax=ax)

# Set x-axis labels
ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(['1', '2', '3', '4'])
ax.set_xlabel('n-th Consecutive Away Games')

# Set y-axis label and format tick labels as percentages
ax.set_ylabel('Win Percentage')
ax.set_yticks(ax.get_yticks())  # Ensure ticks are set before formatting
ax.set_yticklabels([f'{int(x * 100)}%' for x in ax.get_yticks()])

# Set title and remove supertitle
ax.set_title('Consecutive Away Games (2021 - 2024 Seasons)')
fig.suptitle('')

# Remove gridlines
ax.grid(False)

# Save figure with proper DPI and size
plt.savefig(dir_git + '/dev/homeice_advantage/output/eda_awayteam_winfromawayseries.png', dpi=400, bbox_inches='tight')

# Show the plot (optional)
plt.show()

# %% EDA: If sequential travel distance impacts the team win probability

# Divide away_distance into quantiles of 10
tab_dist['away_distance_quantile'] = pd.qcut(tab_dist['away_distance'], q=10, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# Group by the quantile and calculate the mean win percentage
eda_distance_traveled_winp = tab_dist.groupby('away_distance_quantile')\
    .agg(
        {
            'idx_awaywin':'mean',
            'period_ending':'count'
        }
    ).reset_index()

# Plot scatter plot of win percentage against distance traveled
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.scatter(eda_distance_traveled_winp['away_distance_quantile'], eda_distance_traveled_winp['idx_awaywin'], s=eda_distance_traveled_winp['period_ending'] * 0.5, alpha=0.5)
# Draw 50% line
ax.axhline(y=0.5, color='gray', linestyle='--')
# Change y axis label to be percentage with % sign
ax.set_yticks(ax.get_yticks())
ax.set_yticklabels([f'{int(x * 100)}%' for x in ax.get_yticks()])
# Set x-axis labels
ax.set_xlabel('Single/Consecutive Distance Traveled (Quantile)')
# Set y-axis label
ax.set_ylabel('Win Percentage')
# Set title and remove supertitle
ax.set_title('Away Games by Distance Traveled (2021 - 2024 Seasons)')
fig.suptitle('')
# Save figure with proper DPI and size
plt.savefig(dir_git + '/dev/homeice_advantage/output/eda_awayteam_winfromawaydistance.png', dpi=400, bbox_inches='tight')
# %%
