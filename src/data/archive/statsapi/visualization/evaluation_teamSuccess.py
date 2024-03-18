# Process to evaluate team season succcess, 
#       over all available play-level data (2011-current)
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedFormatter, FixedLocator
plt.rcParams.update({'font.size': 16})

# Directories - locate your data location from the scraping exercise
# https://github.com/justinjoliver/NHL-Analytics/tree/main/src/data
str_dirc = '__DATA DIR__';

# import relevant functions and import process data
str_dir_sourceCode = "__SOURE DIR__"
exec(open(str_dir_sourceCode + "/measurement_teamSuccess.py").read())

# Last year season progress
idx_yr_last = 2022
list_years = np.arange(2011, idx_yr_last)

for iter_yr in list_years:
    # Pull the latest data for process
    df_box = pd.read_csv(str_dirc + f"/cleaned/{iter_yr}_02_box.csv", 
                        parse_dates = ['gameDate'], 
                        index_col = 'gameIdx')
    # Unpack data
    df_box_exnt = df_unpack(df_box)
    #df_box_exnt.columns
    # Calculate metrics
    df_box_exnt = df_metrics(df_box_exnt)

    # Aggregate up for team-level statistics
    dfteams = df_box_exnt.groupby(['team_tri_for'])\
        .agg(
            {
                "fenwick_lvl":"sum",
                "corsi_lvl":"sum",
                "rpe":"last",
                "rwin":"last",
                "rgame":"last"
            }
        )
    dfteams.sort_values(by = ["rwin","fenwick_lvl", "corsi_lvl"], inplace = True)
    # Join with season-pythagorean expectation
    #dfteams = dfteams.join(df_pe_season)

    # Normalize KPIs for measurement comparison
    dfteams['wp'] = dfteams.rwin / dfteams.rgame
    dfteams['kpi_corsi'] = kpinorm(dfteams.corsi_lvl)
    dfteams['kpi_fenwick'] = kpinorm(dfteams.fenwick_lvl)
    dfteams['kpi_pe'] = kpinorm(dfteams.rpe)

    # Coefficient estimates from 2010-2020, calculate fitted win percentage

    beta_pyth_old = 0.00809623 * 100
    alpah_pyth_old = 0.042901
    x_starting = 0.3
    x_end = x_starting + 0.45
    y_starting = (x_starting) * beta_pyth_old + alpah_pyth_old
    y_end = (x_end) * beta_pyth_old + alpah_pyth_old
    dfteams['rwin_fittedPE'] = (dfteams['rpe'] * beta_pyth_old + alpah_pyth_old) * 82


    # drop the first game, 
    df_box_exnt.loc[df_box_exnt.gameDate_del_against.notnull(), :].head(3)

    time_games_all = np.sort(df_box_exnt.gameDate_for.unique()); # by date existing

    # Capture the last date.
    iter_time = time_games_all[-1];

    # NOTE: while it may be exciting to compute the measure in real time, it takes
    #   some time to compute it all. I focus on looking at upto the latest available
    #   data

    

    # Head to head statistics
    dff_h2h_com = df_gen_h2h_common(df_box_exnt, iter_time)

    # Game pairwise - RPI statistics
    dfteams = dfteams.join(pairwise_h2h(df_box_exnt, dff_h2h_com))


    # More head-to-head statistics
    dff_h2h_com_win = dff_h2h_com\
        .groupby(['team_tri_for_ow', 'team_tri_for_oow']) \
        .agg(
            {
                'game_won_h2h_ow':'sum', 
                'game_played_h2h_ow':'sum',
                'game_won_h2h_oow':'sum', 
                'game_played_h2h_oow':'sum'
            }
        )
    # for each opponent, compare who has winning record as common opponent
    #   I use win ratio
    dff_h2h_com_win['h2h_win'] = (\
        (dff_h2h_com_win.game_won_h2h_ow / dff_h2h_com_win.game_played_h2h_ow) > \
        (dff_h2h_com_win.game_won_h2h_oow / dff_h2h_com_win.game_played_h2h_oow)
    ) * 1
    # For each team, count how many opponents the team has winning record against 
    rank_pairwise_win = dff_h2h_com_win.groupby(['team_tri_for_ow']) \
                        .agg({'h2h_win':'sum'}) \
                        .sort_values(by = 'h2h_win', ascending = False)
    rank_pairwise_win.columns = ['pairwise_win']

    # Here the rank can even be more continuous (how much they win instead of count in games)

    # Join with the existing team stats
    dfteams = dfteams.join(rank_pairwise_win)
    # Normalize
    dfteams['kpi_pairwise']  = kpinorm(dfteams.pairwise_win)
    dfteams['kpi_rpi']  = kpinorm(dfteams.rpi)
    dfteams['kpi_wp']  = kpinorm(dfteams.wp)

    dfteams['Final'] = dfteams\
        .sort_values(by = 'kpi_wp', ascending = False) \
        .reset_index().index + 1
    dfteams['Final']  = kpinorm(dfteams.Final)

    # Data append 
    # Add season statistics for and against
    df_box_exnt = df_box_exnt.merge(dfteams.reset_index(), 
                    on = ['team_tri_for'], suffixes=('', '_for')) \
                .merge(dfteams.reset_index(), 
                        left_on = ['team_tri_against'], 
                        right_on = ['team_tri_for'],
                        suffixes=('', '_against'))
    # Check wins against better opponent
    # Is the team-for facing eventual better opponent?
    #   Using pairwise here - alternatively can use RPI
    # ...........................................................................
    # Observing win streaks
    # Is opponent better than average?
    df_box_exnt['idx_OppoGood'] = df_box_exnt.rpi_against > \
                                    np.quantile(df_box_exnt.rpi_against, 0.75) 
    # Playing against worse team, thus leading to expected outcome?
    df_box_exnt['idx_OppoBetter'] = df_box_exnt.pairwise_win_against < \
                                    df_box_exnt.pairwise_win
    df_box_exnt['idx_OppoBetter'] = df_box_exnt.rpi_against < \
                                    df_box_exnt.rpi
    df_box_exnt['win_for_better'] = df_box_exnt.win_for * df_box_exnt.idx_OppoGood
    df_box_exnt['win_for_expected'] = df_box_exnt.win_for * df_box_exnt.idx_OppoBetter

    pivot_teamcumwin = df_box_exnt \
        .groupby(['team_tri_for', 'rgame']) \
        .agg(
            win = ("win_for", "sum"),
            win_better = ("win_for_better", "sum"),
            win_expected = ("win_for_expected", "sum")
        ) \
        .reset_index()
    for iter_lag in np.arange(1,3):
        pivot_teamcumwin[f'win_lag{iter_lag}'] = pivot_teamcumwin \
            .groupby(['team_tri_for']).win.shift(iter_lag)
        pivot_teamcumwin[f'win_better_lag{iter_lag}'] = pivot_teamcumwin \
            .groupby(['team_tri_for']).win_better.shift(iter_lag)
        pivot_teamcumwin[f'win_expected_lag{iter_lag}'] = pivot_teamcumwin \
            .groupby(['team_tri_for']).win_expected.shift(iter_lag)
    
    # Identify all win streak
    pivot_teamcumwin['idx_WinStreak'] = 0
    pivot_teamcumwin.loc[(np.sum(pivot_teamcumwin[['win','win_lag1', 'win_lag2']], axis = 1) == 3), 'idx_WinStreak'] = 1
    # Identify win streak with games all expected to win
    pivot_teamcumwin['idx_WinStreak_expected'] = 0
    pivot_teamcumwin.loc[\
        (np.sum(pivot_teamcumwin[['win_expected','win_expected_lag1', 'win_expected_lag2']], axis = 1) >= 2) \
        & pivot_teamcumwin['idx_WinStreak'] == 1 \
        , 'idx_WinStreak_expected'] = 1
    # Identify win streak against good team: 75%tile and above RPI
    pivot_teamcumwin['idx_HotStreak'] = 0
    pivot_teamcumwin.loc[(pivot_teamcumwin.idx_WinStreak == 1) & \
                        (np.sum(pivot_teamcumwin[['win_better', 'win_better_lag1', 'win_better_lag2']], axis = 1) >= 1), \
                            'idx_HotStreak'] = 1

    df_streak = pivot_teamcumwin[['team_tri_for', 
                              'idx_WinStreak', 
                              'idx_WinStreak_expected', 
                              'idx_HotStreak']] \
            .groupby('team_tri_for').sum()
    df_streak['ratio'] = df_streak.idx_HotStreak / df_streak.idx_WinStreak
    df_streak['ratio_viz'] = np.int0(df_streak['ratio']*100)
    df_streak['ratio_viz'] = [str(iter) + '%' for iter in df_streak['ratio_viz']]
    # Sort value for future ease of use

    # Finally join with data
    dfteams = dfteams.join(df_streak)
    # ===========================================================================
    # Some plotting
    exec(open(str_dir_sourceCode + "/evaluation_winStreak_plot.py").read())
    # ...........................................................................
    # output data
    #   (a) game-level stats
    df_box_exnt['yr_season'] = iter_yr
    df_box_exnt.to_csv(str_dirc + f"/processed/{iter_yr}_02_box.csv")
    #   (b) team-level stats for each season
    dfteams['yr_season'] = iter_yr
    dfteams.to_csv(str_dirc + f"/processed/{iter_yr}_02_team.csv")
    #   (c) team-level head-to-head statistics
    dff_h2h_com_win['yr_season'] = iter_yr