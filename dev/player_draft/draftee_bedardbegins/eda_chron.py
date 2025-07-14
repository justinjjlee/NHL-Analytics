# %% Exploratory Data Analysis
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
os.makedirs(data_dir, exist_ok=True)
# %% Pull data
# Load cleaned pull data
dfplayer = pd.read_csv(os.path.join(data_dir, 'merged_draft_player_stats.csv'))
# Just get player information based on id, get unique rows
dfplayer_info = dfplayer[
    [
        'id', 'firstName', 'lastName', 'position_draft', 'teamAbbrev_draft', 'round', 'pickInRound', 'overallPick', 'year',
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
dfplayer_first = dfplayer[dfplayer['round'] == 1]
# Limit to top 10 players
#dfplayer_first = dfplayer_first[dfplayer_first['overallPick'] <= 10]

# only track with those with NHL season: season is not null
dfplayer_first = dfplayer_first[dfplayer_first['season'].notnull()]
# Define year of season by first four character of season
dfplayer_first['year_game'] = dfplayer_first['season'].astype(str).str[:4].astype(int)
# calculate year difference between current year and year drafted
dfplayer_first['year_sincedraft'] = dfplayer_first['year_game'] - dfplayer_first['year']
# Per player row number, to get years in NHL, in ascending season order
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

# %% EDA: Exceptions
# Who are the exceptions?
'''
    Those beyond standard deviation in the first two years of their NHL career
'''
# First year, and over 50 points per season (0 and 1 year)
eda_exceptions = dfplayer_first[dfplayer_first['year_sincedraft'] <= 2]
eda_exceptions = eda_exceptions[
    (
        (dfplayer_first['year_inNHL'] == 1) & (dfplayer_first['points'] > 50)
    ) |
    (
        (dfplayer_first['year_inNHL'] == 2) & (dfplayer_first['points'] > 60)
    )
]
# get unique list based on draft counts and firstName lastName and Points with years
eda_exceptions = eda_exceptions[['id']].drop_duplicates()
# Track their career trejectory
eda_exceptions = eda_exceptions.merge(
    dfplayer_first,
    how='left',
    left_on=['id'],
    right_on=['id']
)

# Get Chicago players data
chicago_players = eda_exceptions[
    eda_exceptions['lastName'].isin(['Kane', 'Toews', 'Bedard']) &
    eda_exceptions['teamAbbrev_draft'].isin(['CHI'])
]
# Max at 14th year
chicago_players = chicago_players[chicago_players['year_inNHL'] <= 14]

# Create custom hue labels that include draft year
chicago_players['player_label'] = chicago_players['lastName'] + ' (' + chicago_players['year'].astype(str) + ')'

# Mark years with stanley cup winning season
#2009–10, 2012–13, 2014–15
chicago_players['stanley_cup'] = 0
chicago_players.loc[chicago_players['year_game'] == 2009, 'stanley_cup'] = 1
chicago_players.loc[chicago_players['year_game'] == 2012, 'stanley_cup'] = 1
chicago_players.loc[chicago_players['year_game'] == 2014, 'stanley_cup'] = 1

# %%
'''
Compare the Chicago greats (?) in the list
    Patrick Kane
    Jonathan Toews
    Connor Bedard
'''
plt.figure(figsize=(8, 4))
sns.lineplot(
    data=chicago_players, 
    x="year_inNHL", 
    y="points", 
    hue="player_label",
    palette="cool",
    linewidth=6,
)

sns.lineplot(
    data=dfplayer_first, 
    x="year_inNHL", 
    y="points", 
    ci="sd",  # Use standard deviation
    errorbar=("se", 2),# Set multiplier to 2 for 2 standard deviations
    palette="CMRmap",
    err_kws={"alpha": 0.2},
    label='1st Rounder (All Players)',
)
plt.xlim(0, 14)

# Add Stanley Cup markers
stanley_cup_data = chicago_players[chicago_players['stanley_cup'] == 1]
plt.scatter(
    stanley_cup_data['year_inNHL'], 
    stanley_cup_data['points'],
    s=250,  # Size of the marker
    marker='*',  # Star marker
    color='gold',  # Gold color for Stanley Cup
    edgecolor='black',
    linewidth=1.5,
    zorder=10,  # Ensure markers are on top
    label='Stanley Cup Win'  # Add to legend
)
plt.title('Chicago Blackhawks First-round Draftees: Kane, Toews, and Bedard')
plt.xlabel('Year(s) Since The Entry Draft')
plt.ylabel('Points per Season (Regular)')
plt.legend(
    title='Player (Draft Year)', 
    loc='lower right'
)

# Repeate but for those draft team same as playing team
#dfplayer_first_same = dfplayer_first[dfplayer_first['teamAbbrev_draft'] == dfplayer_first['lastName']]

# Save image
plt.savefig('./results/chi_exceptional_comparison.png', dpi=450, bbox_inches='tight')

# %% Player roster for Chicago Blackhawks - Stanley Cup Winning Season
'''
Pull player roster and production 

https://api-web.nhle.com/v1/club-stats/CHI/20232024/2
'''

# List of season column with stanley cup = 1
stanley_cup_season = chicago_players[
    (chicago_players['teamAbbrev_draft'] == 'CHI') & 
    (chicago_players['stanley_cup'] == 1)
]['season'].unique().tolist()

# API request for each season
import requests
import json

def get_roster_data(team_abbrev, season, type=2):
    url = f"https://api-web.nhle.com/v1/club-stats/{team_abbrev}/{season}/{type}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to retrieve data for {team_abbrev} in {season}")
        return None
    
# Loop through each season and get the data
roster_data = []
for season in stanley_cup_season:
    data = get_roster_data('CHI', season)
    if data:
        roster_data.append(data)

# %%
# Pull skater information, and convert to dataframe
'''
Extract detailed player statistics from each season's roster data
including games played, scoring stats, time on ice, and more
'''

roster_df = pd.DataFrame()
for season_data in roster_data:
    for player in season_data['skaters']:
        player_info = {
            'season': season_data['season'],
            'player_id': player['playerId'],
            'firstName': player['firstName']['default'],
            'lastName': player['lastName']['default'],
            'position': player['positionCode'],
            'gamesPlayed': player.get('gamesPlayed', 0),
            'goals': player.get('goals', 0),
            'assists': player.get('assists', 0),
            'points': player.get('points', 0),
            'plusMinus': player.get('plusMinus', 0),
            'penaltyMinutes': player.get('penaltyMinutes', 0),
            'powerPlayGoals': player.get('powerPlayGoals', 0),
            'shorthandedGoals': player.get('shorthandedGoals', 0),
            'gameWinningGoals': player.get('gameWinningGoals', 0),
            'overtimeGoals': player.get('overtimeGoals', 0),
            'shots': player.get('shots', 0),
            'shootingPctg': player.get('shootingPctg', 0),
            'avgTimeOnIcePerGame': player.get('avgTimeOnIcePerGame', 0),
            'avgShiftsPerGame': player.get('avgShiftsPerGame', 0),
            'faceoffWinPctg': player.get('faceoffWinPctg', 0)
        }
        roster_df = pd.concat([roster_df, pd.DataFrame([player_info])], ignore_index=True)

# Save data as csv in same path
roster_df.to_csv('./data/chicago_blackhawks_stanleycup20_roster.csv', index=False)
# %% EDA: It's not just about one player

# Create a dataframe with the top 10 points generators for each season
chicago_stanleycup_top5skaters = pd.DataFrame()

# Process each Stanley Cup winning season
for season in stanley_cup_season:
    # Filter data for this season
    season_data = roster_df[roster_df['season'] == str(season)]
    
    # Sort by points in descending order and take top 10
    top5 = season_data.sort_values(by='points', ascending=False).head(5)

    # Add season label and rank within season
    season_str = str(season)
    top5['season_label'] = f"{season_str[:4]}-{season_str[4:8]}"
    top5['rank'] = range(1, 6)
    
    # Add to final dataframe
    chicago_stanleycup_top5skaters = pd.concat([chicago_stanleycup_top5skaters, top5], ignore_index=True)

# %% EDA: Simplified Top 5 points production visualization with pattern for one-season players

# Define colors for players (using your specified colors)
player_colors = ['#000000', '#CE0E2D', '#FF671D', '#CF8A00', '#FFD100', '#00833E', '#00205b', '#99d9d9']

# Create a dictionary mapping unique players to colors
all_players = chicago_stanleycup_top5skaters['lastName'].unique()
player_color_map = {}
for i, player in enumerate(all_players):
    player_color_map[player] = player_colors[i % len(player_colors)]  # Use modulo to handle more players than colors

# Identify players who appear in exactly one season (instead of not in all three)
season_counts = chicago_stanleycup_top5skaters.groupby('lastName')['season_label'].nunique()
one_season_players = set(season_counts[season_counts == 1].index)

print(f"Players in exactly one season: {one_season_players}")

# Create the catplot
g = sns.catplot(
    data=chicago_stanleycup_top5skaters, 
    kind="bar",
    x="rank", 
    y="points", 
    col="season_label",
    height=4, 
    aspect=.5,
    palette="viridis",  # This will be overridden with our custom colors
    hue="rank",
    legend=False
)

# Define a single hatch pattern for all one-season players
single_hatch_pattern = '///'

# Add player names as labels and customize bar colors
for i, ax in enumerate(g.axes.flatten()):
    season_label = list(g.col_names)[i]
    season_data = chicago_stanleycup_top5skaters[chicago_stanleycup_top5skaters['season_label'] == season_label]
    
    # Customize each bar's color and add labels
    for p, bar in zip(season_data.itertuples(), ax.patches):
        # Set bar color based on player's last name
        bar.set_color(player_color_map[p.lastName])
        
        # Add hatch pattern if player appears in exactly one season
        if p.lastName in one_season_players:
            bar.set_hatch(single_hatch_pattern)
            # Set edge color to black for better pattern visibility
            bar.set_edgecolor('black')
        
        # Get the player's first initial and last name
        first_initial = p.firstName[0]
        player_name = f"{first_initial}. {p.lastName}"
        
        # Position for the text (centered on the bar)
        height = bar.get_height()
        x_pos = bar.get_x() + bar.get_width()/2
        
        # Determine text color based on bar color
        # Use black text for the fifth color (#FFD100) and last color (#99d9d9)
        if player_color_map[p.lastName] == '#FFD100' or player_color_map[p.lastName] == '#99d9d9':
            text_color = 'black'
        else:
            text_color = 'white'
        
        # Add player name on the bar with appropriate text color
        ax.text(x_pos, height/2, player_name, 
                ha='center', va='center', 
                color=text_color, fontsize=8, 
                fontweight='bold', rotation=90)
        
        # Add point value above the bar
        ax.text(x_pos, height + 2, f"{int(p.points)}", 
                ha='center', va='bottom', 
                color='black', fontsize=8)

# Set better titles and labels
g.set_axis_labels("Top 5 Skaters", "Points per (Regular) Season")
g.set_titles("{col_name}")
g.figure.suptitle("Point Production of Chicago Blackhawks Stanley Cup Teams", fontsize=14)
g.figure.subplots_adjust(top=0.85)
'''
# Create a simplified legend showing player colors and the single pattern
plt.figure(figsize=(10, 3))
legend_ax = plt.gca()
legend_ax.axis('off')

# Create legend handles for all players
legend_handles = []
legend_labels = []

# Regular players (no hatch)
multi_season_players = set(all_players) - one_season_players
for player in sorted(multi_season_players):
    color = player_color_map[player]
    legend_handles.append(Patch(facecolor=color, label=player))
    legend_labels.append(player + " (multiple seasons)")

# One-season players (with hatch)
for player in sorted(one_season_players):
    color = player_color_map[player]
    patch = Patch(facecolor=color, hatch=single_hatch_pattern, edgecolor='black', label=player)
    legend_handles.append(patch)
    legend_labels.append(player + " (one season only)")

# Add the legend
legend_ax.legend(
    legend_handles, legend_labels, 
    loc='center', 
    ncol=min(4, len(legend_handles)),
    title="Players (Appearance Frequency)"
)
'''
plt.tight_layout()

# Show the main plot
plt.figure(g.figure.number)
# Save image
plt.savefig('./results/chi_cupteam_topskaters.png', dpi=450, bbox_inches='tight')


# %%
