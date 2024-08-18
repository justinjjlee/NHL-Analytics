'''
EDA Analysis for blocking shots:
* Creating data structure
* Formatting data file for data visualization and analysis
'''
# %% Data import and process
import duckdb
import os
import json
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

with open("../../../../settings.json") as f:
    d = json.load(f)
# Connecting to the nhl database csv
db_nhldata = d['credentials']['server_nhl']
# %%
# Save the working directory for results savings
str_cwd = os.getcwd()
# Get to top repository directory
os.chdir("../../../")
# Import functions - object to generate play-by-play data relations
from src.data.apinhle.proc.sequence_dataproc_general import nhl_sequence

# %% Data sequence

dbnhl = nhl_sequence(db_nhldata)

# Initial database creation
dbnhl.database_generate()
# Creating features
dbnhl.database_DataEngineering()

# Print existing columns for the table created
dbnhl.conn.execute("DESCRIBE dfseq").fetchall()

# %% Data specific for this analysis
'''
Notes:
    Blocked shot measured on offense and defense in reverse, such that the offense and defense posture
        Need to be reversed based on the team marked blocking
'''
dbnhl.conn.execute(
    """
        DROP TABLE IF EXISTS dfblock;
        CREATE TABLE dfblock AS 
        SELECT *
            -- 
        
            -- Creating the column of posture the following event
            , CASE 
                -- For blocker team, need to switch
                WHEN  ("blocker_team" = "blocker_team_lag1") 
                    AND ("typeDescKey" = 'blocked-shot')
                    THEN 'Defense'
                WHEN  ("blocker_team" != "blocker_team_lag1") 
                    AND ("typeDescKey" = 'blocked-shot')
                    THEN 'Offense'
                WHEN  ("details.eventOwnerTeam" = "blocker_team_lag1") 
                    AND ("details.zoneCode" = 'D')
                    THEN 'Defense'
                WHEN ("details.eventOwnerTeam" = "blocker_team_lag1") 
                    AND ("details.zoneCode" = 'O')
                    THEN 'Offense'
                WHEN ("details.eventOwnerTeam" != "blocker_team_lag1") 
                    AND ("details.zoneCode" = 'D')
                    THEN 'Offense'
                WHEN ("details.eventOwnerTeam" != "blocker_team_lag1") 
                    AND ("details.zoneCode" = 'O')
                    THEN 'Defense'
                ELSE 'Neutral'
            END AS "blocker_team_lag1_now"
        FROM dfseq
    """
)

# %% EDA: Average activities by team per game
ts_shots = dbnhl.conn.execute(
    """
        SELECT
            idx_season, seasonIdx, 
            CASE 
                WHEN seasonIdx = 2 THEN 'Regular'
                ELSE 'Playoff'
            END AS Season,
            CAST(idx_season/10000 AS INT) AS Year, 
            team_tri,
            AVG(missedshot) AS missedshot,
            AVG(blockedshot) AS blockedshot,
            AVG(shotongoal) AS shotongoal,
            AVG(goal) AS goal
        FROM (
            SELECT 
                idx_season, seasonIdx, gameDate, "details.eventOwnerTeam" AS team_tri,
                SUM(CASE WHEN typeDescKey = 'missed-shot' THEN 1 ELSE 0 END) AS missedshot,
                SUM(CASE WHEN typeDescKey = 'blocked-shot' THEN 1 ELSE 0 END) AS blockedshot,
                SUM(CASE WHEN typeDescKey = 'shot-on-goal' THEN 1 ELSE 0 END) AS shotongoal,
                SUM(CASE WHEN typeDescKey = 'goal' THEN 1 ELSE 0 END) AS goal
            FROM dfseq
            GROUP BY idx_season, seasonIdx, gameDate, "details.eventOwnerTeam"
        )
        GROUP BY idx_season, seasonIdx, team_tri
    """
).df()

# pass custom palette:
fig, axes = plt.subplots(2,2, figsize=(10, 7), dpi=300)
fig.suptitle("Average activities by team per game", fontsize=30)

sns.lineplot(
    ax = axes[0,0],
    x='Year', y='shotongoal', hue='Season', 
    palette=['b','r'],
    data=ts_shots
)
axes[0,0].set_ylabel("Shots per game by team")
axes[0,0].set_title(label = "Shots on goal")

sns.lineplot(
    ax = axes[0,1],
    x='Year', y='missedshot', hue='Season', 
    palette=['b','r'],
    data=ts_shots
)
axes[0,1].set_ylabel("Shots per game by team")
axes[0,1].set_title(label = "Missed shots")

sns.lineplot(
    ax = axes[1,0],
    x='Year', y='blockedshot', hue='Season', 
    palette=['b','r'],
    data=ts_shots
)
axes[1,0].set_ylabel("Shots per game by team")
axes[1,0].set_title(label = "Blocked shots")

sns.lineplot(
    ax = axes[1,1],
    x='Year', y='goal', hue='Season', 
    palette=['b','r'],
    data=ts_shots
)
axes[1,1].set_ylabel("Goals per game by team")
axes[1,1].set_title(label = "Goals")

plt.tight_layout()
fig.savefig(str_cwd + "/result/figure - shot activities by season.png")

# %% Headline Stats: Shot type proportion
eda_totalcount = dbnhl.conn.execute(
    """
        SELECT
            typeDescKey AS "Shot Events",
            --COALESCE(typeDescKey, 'Total') AS "Shot Events",
            COUNT(*) AS "Accounts"
        FROM dfseq
        WHERE 
            gameDate > '2022-10-01'
            AND "typeDescKey" IS NOT NULL
            AND "blocker_team_lag1" IS NOT NULL
            AND typeDescKey IN (
                'missed-shot',
                'blocked-shot',
                'shot-on-goal',
                'goal'
            )
        GROUP BY typeDescKey 
        -- GROUP BY ROLLUP(typeDescKey)
        ORDER BY Accounts ASC
    """
).df()
eda_totalcount["Share"] = eda_totalcount["Accounts"]/eda_totalcount["Accounts"].sum()*100

# Create the plot
fig, ax = plt.subplots(figsize=(10, 2))

# Create a cumulative sum for the 'bottom' parameter in the bar plot
eda_totalcount['Cumulative'] = eda_totalcount['Share'].cumsum() - eda_totalcount['Share']

# Plot each category as a separate layer in the stacked bar chart
for _, row in eda_totalcount.iterrows():
    sns.barplot(
        y=[''], x=[row['Share']], label=row['Shot Events'],
        left=row['Cumulative'], color=sns.color_palette()[row.name % len(sns.color_palette())]
    )

# Adding percentage labels
cumulative_sum = 0
for _, row in eda_totalcount.iterrows():
    ax.text(cumulative_sum + row['Share'] / 2, 0, f'{row["Share"]:.1f}%', ha='left', va='center', color='white')
    cumulative_sum += row['Share']

# Customize the plot
ax.set_ylabel('Percentage')
ax.set_title('Scoring attempts')
ax.legend(title='Scoring attepts')
plt.xticks([])
plt.tight_layout()

plt.savefig(str_cwd + '/result/figure - eda - scoring attempts', dpi=300)
eda_totalcount.to_csv(str_cwd + '/result/figure - eda - scoring attempts.csv', index=False)

# %% EDA: sequence of team posture after blocked shots
tempdf = dfblock.groupby(['blocker_team_lag1_now']).seasonIdx.count().reset_index()

mark_defense = tempdf.loc[tempdf.blocker_team_lag1_now == "D", 'seasonIdx'].iloc[0]
mark_neutral = tempdf.loc[tempdf.blocker_team_lag1_now == "N", 'seasonIdx'].iloc[0]
mark_offense = tempdf.loc[tempdf.blocker_team_lag1_now == "O", 'seasonIdx'].iloc[0]

# Define the periods and percentages
period1 = "Last Event: \n Block shot"
period2 = "Current"
percentages = [mark_defense, mark_neutral, mark_offense]  # Example percentages
percentages = [round(iter, 1) for iter in percentages/sum(percentages)*100]
label_string = ["Defense", "Neutral/Stoppage", "Offense"]

# Define the coordinates
x1, y1 = 0, 0  # Dot for period 1
x2, y2 = 1.3, 0  # Center of the horizontal tree for period 2

# Create the figure and axis
fig, ax = plt.subplots(figsize=(6,2))

# Plot the dot for period 1
ax.plot(x1, y1, 'o', label=period1)
ax.plot(x2, y2, 'o', label=period1, linewidth=1)

# Plot the line connecting period 1 to period 2
ax.plot([x1, x2], [y1, y2], '--k')

# Plot the horizontal tree with three legs for period 2
leg_length = 1
half_leg_length = leg_length / 2
offsets = [-0.5, 0, 0.5]  # Vertical offsets for the legs

for i, offset in enumerate(offsets):
    if offset == 0:
        # Plot the middle line
        ax.plot([x2, x2 + leg_length], [y2, y2 + offset], '-k', linewidth=1)
        ax.annotate('', xy=(x2 + leg_length + 0.1, y2 + offset), xytext=(x2, y2),
                    arrowprops=dict(facecolor='black', edgecolor='black', shrink=0.05, width=1, headwidth=8)
                    )
        ax.text(x2 + leg_length + 0.1, y2 + offset, f"{percentages[i]}% {label_string[i]}", fontsize=12, va='center')
    else:
        # Plot the stepped line for top and bottom
        ax.plot([x2, x2 + half_leg_length], [y2, y2], '-k', linewidth=1)  # First horizontal segment
        ax.plot([x2 + half_leg_length, x2 + half_leg_length], [y2, y2 + offset], '-k', linewidth=1)  # Vertical segment
        ax.plot([x2 + half_leg_length, x2 + leg_length/2], [y2 + offset, y2 + offset], '-k', linewidth=1)  # Second horizontal segment
        ax.annotate('', xy=(x2 + leg_length, y2 + offset), xytext=(x2 + half_leg_length, y2 + offset),
                    arrowprops=dict(facecolor='black', edgecolor='black', shrink=0.05, width=1, headwidth=8)
                )
        ax.text(x2 + leg_length + 0.1, y2 + offset, f"{percentages[i]}% {label_string[i]}", fontsize=12, va='center')

# Annotate the periods
ax.text(x1, y1 - 0.5, period1, fontsize=12, ha='center')
ax.text(x2, y2 - 0.3, period2, fontsize=12, ha='center')

# Set axis limits
ax.set_xlim(-1, 4)
ax.set_ylim(-0.5, 1)

# Remove axis for better visualization
ax.axis('off')

plt.title("Team's posture after blocking shot")
plt.tight_layout()

# Show the plot
plt.savefig(str_cwd + '/result/figure - eda - sequence of posture after blocking shot.png', dpi=600)