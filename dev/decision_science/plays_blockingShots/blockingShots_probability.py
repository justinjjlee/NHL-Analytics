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
from matplotlib.ticker import FuncFormatter
plt.style.use('ggplot')

# Define a function to convert proportion to percentage
def to_percent(y, position):
    # Convert the proportion to percentage
    s = str(round(100 * y,0))
    
    # Format the percentage string with one decimal place
    if plt.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'
# Color pallet to be used
palettes = ['#073b4c', '#06d6a0', '#e63946']

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

# %% Data pull for sequence activities tracker
df_sequence = dbnhl.conn.execute(
    """
        SELECT 
            seasonIdx
            , gameid
            , "periodDescriptor.number" AS period
            , "details.eventOwnerTeam" AS team_owner
            , "details.zoneCode" AS posture_current
            , typeDescKey
            , typeDescKey_lag1
            , penalty_team
            , eventOwnerTeam_sequence
        FROM dfblock
        WHERE 
            gameDate > '2022-10-01'
            AND "typeDescKey_lag1" IN (
                'missed-shot',
                'blocked-shot',
                'shot-on-goal'
            )
    """
).df()

# %% Goal or not right now? given the past event?

df_sequence['idx_goal'] = [1 if iter == 'goal' else 0 for iter in df_sequence.typeDescKey]

# Tabulate probability
tabp = df_sequence \
    .groupby(['typeDescKey_lag1', 'gameid', 'eventOwnerTeam_sequence']) \
    .agg(
        {
            "idx_goal":"sum",
            "seasonIdx":"count"
        }
    )

tabp["p_game"] = tabp['idx_goal']/tabp['seasonIdx']
tabp.reset_index(inplace=True)

# Create category
#tabp["category"] = tabp.eventOwnerTeam_sequence + ": " + tabp.typeDescKey_lag1
tabp["agent"] = [
    "Defending team" if iter == "Own action" else "Shot-attempt team" 
    for iter in tabp["eventOwnerTeam_sequence"]
]
tabp.rename(
    columns = {
        "typeDescKey_lag1":"category"
    }, inplace=True
)
# %%

fig, ax = plt.subplots(figsize=(10,5), nrows=1, ncols=2)
fig.suptitle("Probability of goal score right after shot events")

for cnt, iter in enumerate(["Shot-attempt team", "Defending team"]):
    sns.kdeplot(
        data=tabp.loc[tabp.agent == iter, :], 
        x="p_game", hue="category", 
        palette= palettes, fill=True, alpha=.4,
        ax=ax[cnt]
    )
    ax[cnt].set_xlim([0, 0.25])
    ax[cnt].set_ylim([0, 25])
    ax[cnt].set_title("Previously " + iter.lower())

    ax[cnt].set_xlabel("Likelihood of goal score")

plt.tight_layout()
# Show the plot
plt.savefig(str_cwd + '/result/figure - eda - 1d likelihood of scoring after shot attempts.png', dpi=600)
# %% EDA: Distribution plot - contour

fig, ax = plt.subplots(figsize=(10,5), nrows=1, ncols=2)
fig.suptitle("Probability of goal score right after shot events")

for cnt, iter in enumerate(["Shot-attempt team", "Defending team"]):
    sns.kdeplot(
        data=tabp.loc[(tabp.agent == iter)&(tabp.p_game !=0), :], 
        y="p_game", x="seasonIdx", hue="category", 
        palette= palettes, #fill=True, alpha=.5,
        ax=ax[cnt]
    )
    ax[0].set_ylabel("Likelihood of goal score")
    ax[1].set_ylabel("")
    ax[cnt].set_xlabel("Shot attempt counts")
    ax[cnt].set_ylim([0, 0.25])
    ax[cnt].set_xlim([0, 60])

    ax[cnt].yaxis.set_major_formatter(FuncFormatter(to_percent))

    ax[cnt].set_title("Previously " + iter.lower())


plt.tight_layout()
# Show the plot
plt.savefig(str_cwd + '/result/figure - eda - 2d likelihood of scoring after shot attempts.png', dpi=600)

# %% Saving data for hypothesis testing
tabp.to_csv(str_cwd + '/result/tabulated - sequence.csv', index=False)