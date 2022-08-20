# sub process to evaluate win streak by season

# For plotting, only look at top 15 teams
dfplot = dfteams.sort_values(by = 'idx_WinStreak', ascending = False).head(15)
dfplot = dfplot[['idx_WinStreak', 'idx_WinStreak_expected', 
                'idx_HotStreak', 'ratio', 'ratio_viz']]

fig, ax = plt.subplots(figsize = (12,7))

def addlabels(x,y,z):
    for i in range(len(x)):
        plt.text(i, y[i], z[i], ha = 'center', fontsize = 12)

ax.bar(dfplot.index, height = dfplot['idx_WinStreak'], \
        label = "Win Streak", color = '#FFCB05')
temp_plot = dfplot[dfplot.idx_WinStreak == dfplot.idx_WinStreak_expected]
ax.plot(temp_plot.index, temp_plot['idx_WinStreak_expected'], \
         'v', markersize=12, c = '#00274C', label = 'Win Streak: All expected (Season RPI)')
temp_plot = dfplot[dfplot.idx_WinStreak != dfplot.idx_WinStreak_expected]
ax.plot(temp_plot.index, temp_plot['idx_WinStreak_expected'], \
         '^', markersize=12, c = 'green', label = 'Win Streak: with unexpected (Season RPI)')

# plot hot streak
ax.bar(dfplot.index, height = dfplot['idx_HotStreak'], \
        label = "Hot Streak", color = '#e10600')
addlabels(dfplot.index, dfplot.idx_HotStreak + 0.5, dfplot.ratio_viz)

plt.ylabel('Wins point generated')
plt.title(f"{iter_yr}-{iter_yr+1} Regular Season Win Streak Breakdown")
ax.legend()

plt.savefig(str_dirc + f"/report/Team_Win_Streaks/{iter_yr}.png", \
            facecolor='white', edgecolor='white')