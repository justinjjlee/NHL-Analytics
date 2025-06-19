# %% Exploratory Data Analysis
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
os.makedirs(data_dir, exist_ok=True)
# %% Pull data
# Load cleaned pull data
dfplayer = pd.read_csv(os.path.join(data_dir, 'merged_draft_player_stats.csv'))
# Just get player information based on id, get unique rows
dfplayer_info = dfplayer[
    [
        'id', 'firstName', 'lastName', 
        'position_draft', 'teamAbbrev_draft', 'round', 'pickInRound', 'overallPick', 
        'year',
    ]
].drop_duplicates()
# Player chronical data
dfchrono = pd.read_csv(os.path.join(data_dir, 'all_drafted_players_chronicle.csv'))
# Left join the rest of the player information, based on id and season
dfchrono = dfchrono.merge(dfplayer_info, how='left', left_on=['id'], right_on=['id'])

# Join two df based on 
# %% Create attributes
# For each player id, count teamAbbrev_stats unique count
# Filter out players drafted on the first round
dfplayer_first = dfchrono[
    (dfchrono['round'] == 1) &
    (dfchrono['leagueAbbrev'] == 'NHL')
]
# Limit to top 10 players
#dfplayer_first = dfplayer_first[dfplayer_first['overallPick'] <= 10]

# Only track regular season
dfplayer_first = dfplayer_first[
    (dfplayer_first['gameTypeId'] == 2) 
]
# only track with those with NHL season: season is not null
dfplayer_first = dfplayer_first[dfplayer_first['season'].notnull()]
# Define year of season by first four character of season
dfplayer_first['year_game'] = dfplayer_first['season'].astype(str).str[:4].astype(int)
# calculate year difference between current year and year drafted
dfplayer_first['year_sincedraft'] = dfplayer_first['year_game'] - dfplayer_first['year']
# Per player row number, to get years in NHL, in ascending season order
#   THis way of counting gets us to avoid players moving from team to team in a season to be duplicated
dfplayer_first['year_inNHL'] = dfplayer_first.groupby('id')['season'].rank(method='dense').astype(int)

# Drop negative year_sincedraft
dfplayer_first = dfplayer_first[dfplayer_first['year_sincedraft'] >= 0]
# Group year_sincedraft in buckets (a) first 3 years, (b) 4-6 years, (c) 7-9 years, (d) 10+ years
def bucket_year_sincedraft(year):
    if year <= 3:
        return '0-3'
    elif year <= 6:
        return '4-6'
    elif year <= 9:
        return '7-9'
    else:
        return '10+'
dfplayer_first['year_sincedraft_bucket'] = dfplayer_first['year_sincedraft'].apply(bucket_year_sincedraft)
# Bucket for 1st rounder draft players
def bucket_overallPick(overallPick):
    if overallPick <= 5:
        return '1-5'
    elif overallPick <= 10:
        return '6-10'
    else:
        return '11+'
dfplayer_first['overallPick_bucket'] = dfplayer_first['overallPick'].apply(bucket_overallPick)

# Some additional calculations
# calculate points per game
dfplayer_first['pointspergame'] = dfplayer_first['points'] / dfplayer_first['gamesPlayed']

# %% EDA: Pie chart of player positions (position_draft)
# Create a pie chart of player positions
position_counts = dfplayer_first['position_draft'].value_counts()
# Combine multiple positions (forwards) to be general
position_counts = position_counts.rename({
    'F': 'F (multiple)',
    'C/RW': 'F (multiple)',
    'C/LW': 'F (multiple)',
    'LW/RW': 'F (multiple)',
    'RW': 'RW',
    'LW': 'LW',
    'C': 'C'
})
# After renaming, aggregate the counts for combined categories
position_counts = position_counts.groupby(position_counts.index).sum()

idx_minyear = dfplayer_first['year'].min()
idx_maxyear = dfplayer_first['year'].max()

plt.figure(figsize=(8, 8))

# Explode the pie chart slightly to separate wedges and prevent label overlap
explode = [0.05] * len(position_counts)

# Create the pie chart with autopct to handle percentage calculations
plt.pie(
    position_counts,
    labels=position_counts.index,
    autopct=lambda pct: f"{pct:.1f}%",
    startangle=90,
    colors=sns.color_palette("pastel"),
    wedgeprops=dict(edgecolor='w'),
    textprops={'fontsize': 18},
    explode=explode,
    labeldistance=1.1
)

plt.title(
    f'Positions of 1st Round Draft Picks ({idx_minyear}-{idx_maxyear})', 
    fontsize=18)
plt.axis('equal')
plt.tight_layout()
plt.savefig(
    './results/player_positions_pie_chart.png',
    dpi=300,
    bbox_inches='tight'
)

# %% EDA: Create 2x2 subplot comparing points and points-per-game metrics

# Create a 1x2 grid of subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

# Custom function to calculate quantile-based error bars
def quantile_errorbars(y, data=None, **kwargs):
    # Only use y values that are not NaN
    y = y[~np.isnan(y)]
    
    # If there's no data or too little data, return zeros
    if len(y) < 2:
        return np.zeros(2)
    
    # Calculate 2.5% and 97.5% quantiles
    q025 = np.quantile(y, 0.025)
    q975 = np.quantile(y, 0.975)
    
    # Return the distance from the median to these quantiles
    median = np.median(y)
    
    # Ensure we don't return negative error bars
    lower_err = max(0, median - q025)
    upper_err = max(0, q975 - median)
    
    return [lower_err, upper_err]

# PLOT 1: Aggregated view of points over NHL career (left)
# First plot the overall trend
sns.lineplot(
    data=dfplayer_first, 
    x="year_inNHL", 
    y="points",
    estimator=np.mean,
    errorbar=quantile_errorbars,
    err_kws={"alpha": 0.15},
    color="darkred",
    label="All 1st Round Picks",
    ax=axes[0]
)

# Then overlay the breakdown by draft position
sns.lineplot(
    data=dfplayer_first, 
    x="year_inNHL", 
    y="points", 
    hue="overallPick_bucket",
    estimator=np.mean,
    errorbar=None,  # Remove error bars for the breakdown to avoid clutter
    palette="viridis",
    linewidth=2,
    ax=axes[0]
)

axes[0].set_title('Points per Season', fontsize=16)
axes[0].set_xlabel('Year(s) in NHL', fontsize=12)
axes[0].set_ylabel('Points per Season', fontsize=12)
axes[0].set_xlim(1, 15)
axes[0].grid(alpha=0.3)
axes[0].legend(title='Draft Position', loc='upper right')

# PLOT 2: Points-per-game breakdown (right)
# First plot the overall trend
sns.lineplot(
    data=dfplayer_first, 
    x="year_inNHL", 
    y="pointspergame",
    estimator=np.mean,
    errorbar=quantile_errorbars,
    err_kws={"alpha": 0.15},
    color="darkred",
    label="All 1st Round Picks", 
    ax=axes[1]
)

# Then overlay the breakdown by draft position
sns.lineplot(
    data=dfplayer_first, 
    x="year_inNHL", 
    y="pointspergame", 
    hue="overallPick_bucket",
    estimator=np.mean,
    errorbar=None,  # Remove error bars for the breakdown to avoid clutter
    palette="viridis",
    linewidth=2,
    ax=axes[1]
)

axes[1].set_title('Points-per-Game per Season', fontsize=16)
axes[1].set_xlabel('Year(s) in NHL', fontsize=12)
axes[1].set_ylabel('Points per Game', fontsize=12)
axes[1].set_xlim(1, 15)
axes[1].grid(alpha=0.3)
axes[1].legend(title='Draft Position', loc='upper right')

# Add an overall title
fig.suptitle(
    'Points per Season Over the Careers of NHL First-Round Picks (2006-2023)', 
    fontsize=18
)

# Add explanation text
plt.figtext(0.5, -0.05, 
    "Solid blue/red lines show mean for all 1st round picks with 95% quantile range.\n"
    "Colored lines show breakdown by draft position (picks 1-5, 6-10, and 11+).", 
    ha="center", fontsize=12, 
    bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5}
)

# Adjust layout and spacing
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
# Save the figure
plt.savefig(
    './results/first_round_career_progression.png', 
    dpi=300, 
    bbox_inches='tight'
)

# %% EDA: Exceptional players with multiple high-performance seasons

# Calculate the 95th percentile threshold for points in each NHL year
percentile_thresholds = dfplayer_first.groupby('year_inNHL')['points'].quantile(0.95)
print("95th percentile thresholds by NHL year:")
print(percentile_thresholds)

# Create a new column indicating if a player exceeded the 95th percentile in that season
dfplayer_first['exceeded_95th'] = False
for year in dfplayer_first['year_inNHL'].unique():
    threshold = percentile_thresholds[year]
    dfplayer_first.loc[(dfplayer_first['year_inNHL'] == year) & 
                        (dfplayer_first['points'] > threshold), 'exceeded_95th'] = True

# Count how many seasons each player exceeded the 95th percentile
exceptional_seasons_count = dfplayer_first[dfplayer_first['exceeded_95th']].groupby(
    ['id', 'firstName', 'lastName', 'overallPick']
).size().reset_index(name='exceptional_seasons')

# Filter for players with more than one exceptional season
multi_exceptional_players = exceptional_seasons_count[exceptional_seasons_count['exceptional_seasons'] > 1]

# Sort by number of exceptional seasons, then by overall pick
multi_exceptional_players = multi_exceptional_players.sort_values(
    by=['exceptional_seasons', 'overallPick'], 
    ascending=[False, True]
)

# Display the results
print(f"\nPlayers with multiple seasons exceeding the 95th percentile (n={len(multi_exceptional_players)}):")
print(multi_exceptional_players)

# Get more details about these exceptional performances
# Un-comment if i only want to include players with multiple exceptional seasons
exceptional_details = dfplayer_first[
    (dfplayer_first['exceeded_95th']) 
    #& (dfplayer_first['id'].isin(multi_exceptional_players['id']))
].sort_values(['lastName', 'firstName', 'year_inNHL'])

# Select relevant columns for the detailed view
exceptional_details_display = exceptional_details[[
    'firstName', 'lastName', 'year_inNHL', 'season', 'overallPick',
    'points', 'goals', 'assists', 'gamesPlayed', 'pointspergame'
]]

print("\nDetailed view of exceptional seasons:")
print(exceptional_details_display)

# Create a visualization of these exceptional performers
plt.figure(figsize=(12, 8))

# Get top 10 players by number of exceptional seasons
top_players = multi_exceptional_players.head(15)

# Plot their performances
for _, player in top_players.iterrows():
    player_data = dfplayer_first[
        (dfplayer_first['id'] == player['id']) &
        (dfplayer_first['year_inNHL'] <= 15)  # Limit to first 15 years for clarity
    ]
    
    # Plot player's career trajectory
    plt.plot(
        player_data['year_inNHL'], 
        player_data['points'],
        marker='o',
        linewidth=2,
        label=f"{player['firstName']} {player['lastName']} (#{player['overallPick']})"
    )
    
    # Highlight exceptional seasons with larger markers
    exceptional_seasons = player_data[player_data['exceeded_95th']]
    if not exceptional_seasons.empty:
        plt.scatter(
            exceptional_seasons['year_inNHL'],
            exceptional_seasons['points'],
            s=100,
            edgecolor='black',
            zorder=10
        )

# Add 95th percentile threshold line
for year, threshold in percentile_thresholds.items():
    if year <= 15:  # Limit to first 15 years
        plt.plot([year-0.2, year+0.2], [threshold, threshold], 'r--', alpha=0.6)

plt.title('Top Players with Multiple Exceptional Seasons\n(Points > 95th percentile)', fontsize=16)
plt.xlabel('Years in NHL', fontsize=12)
plt.ylabel('Points', fontsize=12)
plt.grid(alpha=0.3)
plt.xlim(0.5, 15.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Save the visualization
plt.savefig('./results/multi_exceptional_performers.png', dpi=300, bbox_inches='tight')
plt.show()

# Save the data for further analysis
multi_exceptional_players.to_csv('./results/multi_exceptional_players.csv', index=False)
exceptional_details_display.to_csv('./results/exceptional_seasons_details.csv', index=False)

# %%
