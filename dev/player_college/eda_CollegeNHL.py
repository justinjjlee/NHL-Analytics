# %%
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so

cwd = os.getcwd()
cwd_git = os.path.dirname(os.path.dirname(os.path.dirname(cwd)))

import sys
sys.path.append(cwd_git)

import credentials
from credentials import cwd_data

# For the first time, run the data processing
#%run -i 'player_college_matching.py'
# %%
dfin = pd.read_csv(cwd_data + "nhl/data_gamelvl/data/processed/2022cum_player_collegematch.csv")
# Ignore Goalie
dfin = dfin.loc[dfin.position != "G"]
# Season to start at 2017
dfin = dfin.loc[dfin.season >= 2017]
# For plotting purpose
dfin["Draft Level"] = ["1st" if iter == "1st" else "FA" if iter=="FA" else "Lower" for iter in dfin.draft]
# fixing the draft season 
dfin["draft"] = ["7th" if iter == "9th" else iter for iter in dfin.draft]
# %%
# Top 10 production college? need college column
tab_count = dfin.groupby('college').id_player.nunique().reset_index()
templot = tab_count.sort_values(by='id_player',ascending=False).head(8)

# Include the time-on-ice (median) contribution
# Exclude goalie here (which would be ~ 60 mins)
tempdf = dfin.loc[dfin.gamemin < 30,:]
tab_count = tempdf.groupby('college').gamemin.median().reset_index()
templot = pd.merge(templot,tab_count, on='college', how='left')

# %%
# Observing rookie debut
rookie = dfin.loc[dfin.rookie==True, :]\
    .groupby(['season','college'])\
    .id_player.count().reset_index()
rookie = rookie.loc[rookie.college.isin(templot.college)]

tab_rookie = rookie.pivot_table(index="season", values="id_player", columns="college").reset_index()

tab_rookie.plot(x='season', kind='bar', stacked=True,
        title='Rookie Debut by College (Top 8)')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
plt.xticks(rotation=0)

plt.savefig("./output/debut.png",  bbox_inches='tight', dpi=300)
# %%
fig, ax = plt.subplots()
plt.bar(templot.college, templot.id_player)

ax.bar_label(ax.containers[0], label_type='edge')

plt.title("Number of College Players in NHL (2017-2023) - Top 8")
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.savefig("./output/college.png",  bbox_inches='tight', dpi=300)
plt.show()
# %%
# Exploring years in college and draft status
tempdf = dfin.groupby(['id_player'])\
    .agg(
        {
            "college":"max",
            "college_yrs":"max",
            "draft":"max"
        }
    )
plotdf = tempdf.groupby(["college_yrs", "draft"]).college.count().reset_index()
plotdf.sort_values(by = ["draft", "college_yrs","college"], inplace=True)

# %%
p = (so.Plot(plotdf, x="draft", y="college", color="college_yrs")
    .add(so.Bar(), so.Stack())
)
p = p.label(
    x="Draft Status", 
    y="Number of Players", 
    color= "Years in\nCollege",
    title="NHL Players' College Tenure (2017-2023)"
)
p.save("./output/college_draftnyrs.png",  bbox_inches='tight', dpi=300)

# %%
fig, axes = plt.subplots(1, 2)
fig.suptitle('Average Game Time Played')
sns.kdeplot(ax=axes[0], data=dfin.loc[dfin.rookie==False], x="gamemin", 
            hue="college_yrs", common_norm=False)
axes[0].set_title("Rookie Season")
sns.kdeplot(ax=axes[1], data=dfin.loc[dfin.rookie==True], x="gamemin", 
            hue="college_yrs", common_norm=False)
axes[1].set_title("non-Rookie Season")

fig.savefig("./output/college_yrs_timeonice.png",  bbox_inches='tight', dpi=300)

# %%
idx_order = sorted(dfin["Draft Level"].unique())
fig, axes = plt.subplots(1, 2)
fig.suptitle('Goals Scored')
sns.kdeplot(ax=axes[0], data=dfin.loc[dfin.rookie==False], x="goals", 
            hue="Draft Level", hue_order=idx_order, common_norm=False)
axes[0].set_title("Rookie Season")
sns.kdeplot(ax=axes[1], data=dfin.loc[dfin.rookie==True], x="goals", 
            hue="Draft Level", hue_order=idx_order, common_norm=False)
axes[1].set_title("non-Rookie Season")

fig.savefig("./output/college_draft_goals.png",  bbox_inches='tight', dpi=300)

# %%
idx_order = sorted(dfin["Draft Level"].unique())
fig, axes = plt.subplots(1, 2)
fig.suptitle('Average Game Time Played')
sns.kdeplot(ax=axes[0], data=dfin.loc[dfin.rookie==False], x="gamemin", 
            hue="Draft Level", hue_order=idx_order, common_norm=False)
axes[0].set_title("Rookie Season")
sns.kdeplot(ax=axes[1], data=dfin.loc[dfin.rookie==True], x="gamemin", 
            hue="Draft Level", hue_order=idx_order, common_norm=False)
axes[1].set_title("non-Rookie Season")

fig.savefig("./output/college_draft_timeonice.png",  bbox_inches='tight', dpi=300)
# %%
idx_order = sorted(dfin["Draft Level"].unique())
fig, axes = plt.subplots(1, 2)
fig.suptitle('Season Plus/Minus')
sns.kdeplot(ax=axes[0], data=dfin.loc[dfin.rookie==False], x="plusminus", 
            hue="Draft Level", hue_order=idx_order, common_norm=False)
axes[0].set_title("Rookie Season")
sns.kdeplot(ax=axes[1], data=dfin.loc[dfin.rookie==True], x="plusminus", 
            hue="Draft Level", hue_order=idx_order, common_norm=False)
axes[1].set_title("non-Rookie Season")

fig.savefig("./output/college_draft_plusminus.png",  bbox_inches='tight', dpi=300)

# %% Player stats rank
player_stats_lt = dfin.groupby(["fullName", "draft", "college"])\
    .agg(
        {
            "position":"last",
            "rookie":"sum",
            "team_triCode": "last",
            "season":"count",
            "gamemin":"mean",
            "plusminus":"mean",
            "goals":"mean",
            "assists":"mean",
            "takeaways":"mean",
            "giveaways":"mean"
        }
    ).sort_values("goals", ascending=False)

# Count only for those who started their career in 2017
player_stats_lt = player_stats_lt.loc[player_stats_lt.rookie == 1, :].head(20)

player_stats_lt["gamemin"] = player_stats_lt["gamemin"].astype(int)
player_stats_lt["plusminus"] = player_stats_lt["plusminus"].round(1)
player_stats_lt["goals"] = player_stats_lt["goals"].round(1)
player_stats_lt["assists"] = player_stats_lt["assists"].round(1)
player_stats_lt["takeaways"] = player_stats_lt["takeaways"].round(1)
player_stats_lt["giveaways"] = player_stats_lt["giveaways"].round(1)

player_stats_lt.drop(columns="rookie", inplace=True)

player_stats_lt.reset_index(inplace=True)

player_stats_lt.columns = ["Player", "Draft", "College", "Position", "Last Team", "Seasons", "Game Minutes", "Plus-Minus", "Goals", "Assists", "Takeaways", "Giveaways"]

player_stats_lt.to_csv("./output/df_players_top_college.csv", index=False)

player_stats_lt