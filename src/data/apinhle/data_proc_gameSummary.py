# %%
import pandas as pd
import numpy as np
import requests
import os
import time
import datetime

# Functions to process box scores
from function.procs_boxscore import *

# Latest data setting
yr_now = datetime.datetime.today().year
mo_now = datetime.datetime.today().month

# Select starting year for season to pull.
#   Until the following season starts, always pull the current/past eyar
if mo_now < 8: # Season starts on October
    # Then the season marks starts in the previous calendar year
    iter_year = yr_now - 1
else:
    iter_year = yr_now

# %% Team success measurements
# -----------------------------------------------------
# Load the current data
try: # If the data exist,
    df_box_team   = pd.read_csv(
        f"./latest/box/{iter_year}_box_team.csv",
        parse_dates = ['gameDate'], 
        index_col = 'gameIdx'
    )

    team_season = nhl_dataproc_teamsuccess(iter_year)
    df_kpi, summary_game = team_season.dataproc(df_box_team)

    summary_game.to_csv(f"./latest/box/{iter_year}_box_gameStats.csv")
    df_kpi.to_csv(f"./latest/team/season/{iter_year}_team_season.csv")

    print("Team - season-level statistics compute completed")
except: # Data does not exist
    print("The data currently does not exist. Exit the process safely.")

# %% Plot: Generate statistics for team success measurements
'''
# Vector of min
x_min = np.min(df_kpi['rpe'])
x_max = np.median(df_kpi['rpe'])
y_min = np.min(df_kpi['wp'])
y_max = np.median(df_kpi['wp'])
x_max_super = np.max(df_kpi['rpe'])
y_max_super = np.max(df_kpi['wp'])

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedFormatter, FixedLocator
from adjustText import adjust_text

# Plot configuration
fig = plt.figure(figsize=(5,4), dpi=400)
ax = plt.gca()
# Line plot for cutoff annotation
plt.annotate('Equilibrium', xy=(0.53, 0.6),
             xycoords='data',
             xytext=(0.59, 0.48),
             textcoords='data',
             color='red',
             arrowprops=dict(arrowstyle= '<|-|>',
                             color='red',
                             lw=3.5,
                             ls='--')
           )
# Point out the cutoff regions
emph1 = plt.Rectangle((x_min, y_min), (x_max-x_min), (y_max-y_min), facecolor="black", alpha = 0.1, ec = 'k', lw=2)
emph2 = plt.Rectangle((0.6, 0.6), (x_max_super-0.6), (y_max_super-0.6), facecolor="red", alpha = 0.2, ec = 'r', lw=2)
ax.add_patch(emph1)
ax.add_patch(emph2)
# Plot the data
plt.scatter(df_kpi['rpe'], df_kpi['wp'], c = df_kpi['pairwise_win'])
# Color bar annotation
cbar = plt.colorbar()
cbar.set_label('Strength of record: Pairwise Rank', rotation = 270, labelpad = 20)
# Labeling
plt.xlabel(f"Pythagorean Expectation: {iter_year}-{iter_year-2000+1} Regular Season")
plt.ylabel("Win percentage")
plt.title("Pythagorean Expectation & Strenght of Record")
# Adding team label annotation
iter_teams = []
for i, txt in enumerate(df_kpi.index):
    iter_teams.append(
        ax.annotate(
            txt, (df_kpi['rpe'][i], df_kpi['wp'][i])
        )
    )
# Adjust text location
adjust_text(iter_teams)
#plt.xlim([0.3, 0.8])
#plt.ylim([0.6, 1.2])

# Save figure
plt.savefig(f"./latest/team/season/plot_PEnSoR_{iter_year}.png",  bbox_inches='tight', dpi=400)
'''

print("au revoir.")