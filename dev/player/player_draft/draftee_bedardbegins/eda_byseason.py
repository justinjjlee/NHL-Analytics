# %% Data Pull 

import pandas as pd
import numpy as np
import requests
import time
import datetime
import seaborn as sns
import matplotlib.pyplot as plt

# Save data
chi_players = pd.read_csv('./data/chi_players.csv')
# %%
# For each season, rank top 10 players point generation 
#   Create dataframe of last name, games played, and points 
#   data is in season-level, so no need to aggregate
skaters = chi_players.query('positionCode != "G"')
# only regular season
skaters = skaters[skaters['idx_season_type'] == 2]
skaters = skaters.groupby(['idx_season'])[['lastName.default', 'gamesPlayed', 'avgTimeOnIcePerGame', 'plusMinus', 'points']].apply(
    lambda x: x.sort_values('points', ascending=False).head(6)[['lastName.default', 'gamesPlayed', 'points']]
)
# Season index fix
skaters['Season'] = skaters.index.get_level_values(0)
# Calculate points per game
skaters['PPG'] = skaters['points'] / skaters['gamesPlayed']
# Create column of rank number players by points per game per season
skaters['Rank'] = skaters.groupby('Season')['PPG'].rank(method='first', ascending=False).astype(int)
# Sort the dataframe by Season first, then Rank
skaters = skaters.sort_values(['Season', 'Rank'])

# Using seaborn plot, by season as category, ranking in order, plot y-axis points per game
plt.figure(figsize=(12, 6))
sns.barplot(data=skaters, x='Rank', y='PPG', hue='Season', dodge=True)
plt.title('Top 6 Players by Points Per Game')
plt.xlabel('Rank Order')
plt.ylabel('Points Per Game')
#plt.xticks(rotation=45)
plt.legend(title='Season')
plt.tight_layout()

# Add player names as labels along the bars (vertical alignment)
for i in range(skaters.shape[0]):
    # Calculate the x position based on rank and season offset for dodge=True
    seasons = skaters['Season'].unique()
    season_idx = list(seasons).index(skaters['Season'].iloc[i])
    n_seasons = len(seasons)
    offset = (season_idx - n_seasons/2 + 0.5) * (0.8/n_seasons)
    x_pos = skaters['Rank'].iloc[i] - 1 + offset
    
    # Position text vertically along the bar, right-aligned to the tip
    plt.text(x_pos, skaters['PPG'].iloc[i] / 2, skaters['lastName.default'].iloc[i],
             ha='center', va='center', fontsize=9, rotation=90, 
             color='white', fontweight='bold')

# Add some padding to the top of the plot to make room for labels
plt.ylim(0, skaters['PPG'].max() * 1.15)
#plt.show()
# # Save image
plt.savefig('./results/chi_cupteam_ppg.png', dpi=450, bbox_inches='tight')

# %% Repeat above but for plusMinus
skaters = chi_players.query('positionCode != "G"')
# only regular season
skaters = skaters[skaters['idx_season_type'] == 2]
skaters = skaters.groupby(['idx_season'])[['lastName.default', 'gamesPlayed', 'plusMinus']].apply(
    lambda x: x.sort_values('plusMinus', ascending=False).head(3)[['lastName.default', 'gamesPlayed', 'plusMinus']]
)
# Repeat above but bottom 6 players and combine
skaters_bottom = chi_players.query('positionCode != "G"')
skaters_bottom = skaters_bottom.groupby(['idx_season'])[['lastName.default', 'gamesPlayed', 'plusMinus']].apply(
    lambda x: x.sort_values('plusMinus', ascending=True).head(6)[['lastName.default', 'gamesPlayed', 'plusMinus']]
)
# Combine top and bottom players
skaters = pd.concat([skaters, skaters_bottom])

# Season index fix
skaters['Season'] = skaters.index.get_level_values(0)
# Create column of rank number players by points per game per season
skaters['Rank'] = skaters.groupby('Season')['plusMinus'].rank(method='first', ascending=False).astype(int)
# Sort the dataframe by Season first, then Rank
skaters = skaters.sort_values(['Season', 'Rank'])

plt.figure(figsize=(12, 6))
sns.barplot(data=skaters, x='Rank', y='plusMinus', hue='Season', dodge=True)
plt.title('Top 3 and Bottom 6 Players by Plus/Minus')
#plt.xlabel('Rank Or')
plt.ylabel('Season Plus/Minus')
plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
#plt.xticks(rotation=45)
plt.legend(title='Season')
plt.tight_layout()

# Add player names as labels along the bars (vertical alignment)
for i in range(skaters.shape[0]):
    # Calculate the x position based on rank and season offset for dodge=True
    seasons = skaters['Season'].unique()
    season_idx = list(seasons).index(skaters['Season'].iloc[i])
    n_seasons = len(seasons)
    offset = (season_idx - n_seasons/2 + 0.5) * (0.8/n_seasons)
    x_pos = skaters['Rank'].iloc[i] - 1 + offset
    
    # If value less than 10 then different color and positioning
    if np.abs(skaters['plusMinus'].iloc[i]) < 10:
        iter_color = 'black'
        iter_align = 'bottom' if skaters['plusMinus'].iloc[i] > 0 else 'top'
    else:
        iter_color = 'white'
        iter_align = 'center'
    # Position text vertically along the bar, right-aligned to the tip
    plt.text(x_pos, skaters['plusMinus'].iloc[i] / 2, skaters['lastName.default'].iloc[i],
            ha='center', va=iter_align, fontsize=9, rotation=90, 
            color=iter_color, fontweight='bold')

# Add some padding to the top of the plot to make room for labels
plt.ylim(skaters['plusMinus'].min() * 1.15, skaters['plusMinus'].max() * 1.15)
#plt.show()

# Save image
plt.savefig('./results/chi_cupteam_plusminus.png', dpi=450, bbox_inches='tight')

# %% Goalie stats
goalies = chi_players.query('positionCode == "G"')
# only regular season
goalies = goalies[goalies['idx_season_type'] == 2]
# Played at least 10 games
goalies = goalies[goalies['gamesPlayed'] >= 10]
goalies = goalies.groupby(['idx_season'])[['lastName.default', 'gamesPlayed', 'goalsAgainstAverage', 'savePercentage']].apply(
    lambda x: x.sort_values('goalsAgainstAverage', ascending=False)[['lastName.default', 'gamesPlayed', 'goalsAgainstAverage', 'savePercentage']]
)
# Season index fix
goalies['Season'] = goalies.index.get_level_values(0)
# Create column of rank number players by points per game per season
goalies['Rank'] = goalies.groupby('Season')['goalsAgainstAverage'].rank(method='first', ascending=False).astype(int)
# Sort the dataframe by Season first, then Rank
goalies = goalies.sort_values(['Season', 'Rank'])

# Create plots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot Goals Against Average - always use hue for consistency
sns.barplot(data=goalies, x='lastName.default', y='goalsAgainstAverage', hue='Season', ax=axes[0])
axes[0].set_title('Goals Against')
axes[0].set_xlabel('Goalies')
axes[0].set_ylabel('Season Average')

# Plot Save Percentage 
sns.barplot(data=goalies, x='lastName.default', y='savePercentage', hue='Season', ax=axes[1])
axes[1].set_title('Save Percentage')
axes[1].set_xlabel('Goalies')
axes[1].set_ylabel('Season Average')
axes[1].set_ylim(0.8, 1)  # Save percentage is between 0 and 1
axes[1].legend(title='Season', loc='upper left')

# Rotate x-axis labels for better readability
for ax in axes:
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()

# Save the figure
plt.savefig('./results/chi_cupteam_goalies.png', dpi=450, bbox_inches='tight')

# %%
