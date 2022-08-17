# Team success measurement calculations
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedFormatter, FixedLocator

# Data pull
def df_unpack(df_box):
    # Collect the team names
    str_team_names = list(df_box.team_tri.unique())

    # Create column of date differneces. 
    # make sure the dates are sorted first, then shift by each team.
    df_box.sort_values(by = ['gameDate'], inplace = True)
    df_box.loc[:, 'gameDate_prev'] = df_box.groupby(['team_tri'])['gameDate'].shift(1)
    df_box.loc[:, 'gameDate_del'] = df_box.loc[:, 'gameDate'] - df_box.loc[:, 'gameDate_prev']
    #df_box.drop('gameDate_prev', axis = 1, inplace = True)

    # get indicator for current and next game: these will be used to match game for predictor of the next game
    df_box['gameIdx_now'] = df_box.index
    df_box['gameIdx_next'] = df_box.groupby(['team_tri'])['gameIdx_now'].shift(-1)
    #df_box.drop('gameIdx_now', axis = 1, inplace = True)

    # For the team, match the opponent team info.
    def oppo_match(df, str_team_to_eval):
      # The function is used to match specific opponenets, without matching all teams
      df_box_team = df_box.loc[df_box['team_tri'] == str_team_to_eval, :]
      df_box_team_excl = df_box.loc[df_box['team_tri'] != str_team_to_eval, :]

      # join the opponent information
      df_box_team = df_box_team.join(df_box_team_excl, 
                                    on = 'gameIdx',
                                    how = 'left', 
                                    lsuffix = '_for',
                                    rsuffix = '_against')
      return df_box_team

    # first team to run
    iter_team = str_team_names[0]
    df_box_exnt = oppo_match(df_box, iter_team)
    # then match rest of the teams
    for iter_team in str_team_names[1:]:
      df_box_exnt = pd.concat([df_box_exnt, oppo_match(df_box, iter_team)])
    
    return df_box_exnt

# Measurement calculations
def df_metrics(df_box_exnt):
    # Coris and Fenwick
    df_box_exnt['fenwick_for'] = df_box_exnt.shots_for + df_box_exnt.shots_missed_for
    df_box_exnt['corsi_for'] = df_box_exnt.fenwick_for + df_box_exnt.shots_blocked_for
    df_box_exnt['fenwick_against'] = df_box_exnt.shots_against + df_box_exnt.shots_missed_against
    df_box_exnt['corsi_against'] = df_box_exnt.fenwick_against + df_box_exnt.shots_blocked_against

    # Not much meaning for calculating game-level stat
    df_box_exnt['corsi_lvl'] = df_box_exnt.corsi_for - df_box_exnt.corsi_against
    df_box_exnt['corsi_pct'] = df_box_exnt.corsi_for / (df_box_exnt.corsi_for + df_box_exnt.corsi_against)

    df_box_exnt['fenwick_lvl'] = df_box_exnt.fenwick_for - df_box_exnt.fenwick_against

    # Cumulative sum measure as season progress
    #   Make sure to have the data points sorted by dates.

    # Basic stats
    df_box_exnt['rshots_for'] = df_box_exnt.groupby(['team_tri_for']).shots_for.cumsum()
    df_box_exnt['rshots_missed_for'] = df_box_exnt.groupby(['team_tri_for']).shots_missed_for.cumsum()
    df_box_exnt['rshots_blocked_for'] = df_box_exnt.groupby(['team_tri_for']).shots_blocked_for.cumsum()

    df_box_exnt['rshots_against'] = df_box_exnt.groupby(['team_tri_for']).shots_against.cumsum()
    df_box_exnt['rshots_missed_against'] = df_box_exnt.groupby(['team_tri_for']).shots_missed_against.cumsum()
    df_box_exnt['rshots_blocked_against'] = df_box_exnt.groupby(['team_tri_for']).shots_blocked_against.cumsum()
    # Goal productions
    df_box_exnt['rgoals_for'] = df_box_exnt.groupby(['team_tri_for']).goals_for.cumsum()
    df_box_exnt['rgoals_against'] = df_box_exnt.groupby(['team_tri_for']).goals_against.cumsum()
    
    # Rolling pythagorean expectation
    df_box_exnt['rpe'] = calc_pe(df_box_exnt['rgoals_for'], df_box_exnt['rgoals_against'])
    # Rolling pythagorean expectation, but just last x-number of games
    n_ma = 10 # last 10 games
    # Cumulative rolling calculations 
    temp_gf = df_box_exnt.groupby(['team_tri_for']).rgoals_for.shift(n_ma)
    temp_ga = df_box_exnt.groupby(['team_tri_for']).rgoals_against.shift(n_ma)
    # Remove previous cumulative performance
    temp_gf = df_box_exnt['rgoals_for'] - temp_gf
    temp_ga = df_box_exnt['rgoals_against'] - temp_ga
    df_box_exnt[f'rpe_{n_ma}'] = calc_pe(temp_gf, temp_ga)

    # Rolling win performance
    # Include cumulative win count and games played
    df_box_exnt["rwin"] = df_box_exnt.groupby(['team_tri_for']).win_for.cumsum()
    df_box_exnt["rgame"] = df_box_exnt.groupby(['team_tri_for']).win_for.cumcount() + 1
    # note that the count in python zero-indexes.
    # Calculate winning percentage of the team in each iterations, in proportion
    df_box_exnt["rwp"] = df_box_exnt["rwin"] / df_box_exnt["rgame"] 
    
    # Measruement stats
    df_box_exnt['rfenwick_for'] = df_box_exnt.groupby(['team_tri_for']).fenwick_for.cumsum()
    df_box_exnt['rfenwick_against'] = df_box_exnt.groupby(['team_tri_for']).fenwick_against.cumsum()
    df_box_exnt['rfenwick_lvl'] = df_box_exnt.rfenwick_for - df_box_exnt.rfenwick_against

    df_box_exnt['rcorsi_for'] = df_box_exnt.groupby(['team_tri_for']).corsi_for.cumsum()
    df_box_exnt['rcorsi_against'] = df_box_exnt.groupby(['team_tri_for']).corsi_against.cumsum()

    df_box_exnt['rcorsi_lvl'] = df_box_exnt.rcorsi_for - df_box_exnt.rcorsi_against
    df_box_exnt['rcorsi_pct'] = df_box_exnt.rcorsi_for / (df_box_exnt.rcorsi_for + df_box_exnt.rcorsi_against)

    # append to get opponent's statistics
    #temp = df_box_exnt[['team_tri','gameIdx_now','fenwick_against', 
    #                    'fenwick_lvl', 'corsi_for', 'corsi_against', 
    #                    'corsi_lvl', 'corsi_pct']]

    df_box_exnt = df_box_exnt.merge(df_box_exnt, how = 'left', 
                                    left_on = ['team_tri_against', 
                                               'gameIdx_now_against'], 
                                    right_on = ['team_tri_for', 'gameIdx_now_for'], 
                                    suffixes = ('', '_oppo'))
    df_box_exnt.set_index('gameIdx_now_for', inplace = True)
    # remove duplicated column with merged opponent data
    df_box_exnt = df_box_exnt.loc[:,~df_box_exnt.columns.duplicated()]
    
    return df_box_exnt

# pythagorean expectations
def calc_pe(x,y):
    return (x**2)/(x**2 + y**2)

# Functions to normalize [0,1]
def kpinorm(x):
    return (x - np.min(x))/(np.max(x) - np.min(x))

# Estimation of head-to-head metrics
def df_gen_h2h(dffunc):
    # Generate head-to-head data frame
    # Need to calculate the team level stats separately 
    dff_team = dffunc.groupby(['team_tri_for']).agg(
        game_played = ("rgame","max"), 
        game_won = ("rwin","max")
    )

    dff_oppo = dffunc.groupby(['team_tri_against']).agg(
        game_opponent_played = ("rgame_oppo","max"), 
        game_opponent_won = ("rwin_oppo","max")
    )

    dff_h2h = dffunc.groupby(['team_tri_for', 'team_tri_against']).agg(
        # Start with accumulated value to get team-level records\
        # Need to think about this carefully - I am counting how many games 
        #     played up to the point they played the team,
        # if played the beginning of the year, would not be counted well.

        # Get head-to-head information based on cumulative each game
        game_played_h2h = ("win_for", "count"),
        game_won_h2h = ("win_for","sum"),
        play_goal_h2h = ("goals_for", "sum"),
        play_goal_against_h2h = ("goals_against", "sum"),
        play_shot_h2h = ("shots_for", "sum")
        # Other stats, like score for/against
    )
    dff_h2h.reset_index(inplace = True)

    # Join all three to create head to head record pair with all teams 
    dff_h2h = dff_h2h.join(dff_team, on = 'team_tri_for')
    dff_h2h = dff_h2h.join(dff_oppo, on = 'team_tri_against')

    return dff_h2h


# Function to generate head to head record for a given time period
def df_gen_h2h_common(dffunc_origin, iter_time):
    # Setting,
    # Columns for head-to-head records
    temp_str_col = ['team_tri_for', 'team_tri_against', 'game_played_h2h', 
                    'game_won_h2h', 'play_goal_h2h', 'play_goal_against_h2h', 
                    'play_shot_h2h'];

    # pull all games until the date of the game
    tempdf = dffunc_origin.loc[dffunc_origin.gameDate_for <= iter_time, : ]

    # (A) Go through each team to calculate head to head estimation
    dff_h2h = df_gen_h2h(tempdf)

    # ------------------------------------------------------------------------
    # Go through each of own team
    for idx_team, iter_team in enumerate(tempdf.team_tri_for.unique()):
        # pull the data of own, with relevant column

        # pull data of own:
        temp_df = dff_h2h.loc[dff_h2h.team_tri_for == iter_team, temp_str_col].add_suffix('_ow')
        # pull data of opponent
        team_opponents = temp_df.team_tri_against_ow.unique()

        # For each of own team, go through each of opponent
        for idx_opponent, iter_opponent in enumerate(team_opponents):
            # Get head-to-head record of opponent
            temp_df_oppo = dff_h2h.loc[(dff_h2h.team_tri_for != iter_team) & 
                                      (dff_h2h.team_tri_for.isin([iter_opponent])),
                                      :].add_suffix('_oow')
            # Get the list of opponent's opponents
            team_opponents_oppo = list(temp_df_oppo.team_tri_against_oow.unique())
            # Drop the two team compared 
            team_opponents_oppo.remove(iter_team)

            # Get the list of common opponents
            team_opponents_common = list(set(team_opponents) & set(team_opponents_oppo))

            # For own and opponents, get the summary statisics of those teams only
            temp_df_own = temp_df[temp_df.team_tri_against_ow.isin(team_opponents_common)]
            temp_df_oppo = temp_df_oppo[temp_df_oppo.team_tri_against_oow.isin(team_opponents_common)]

            # Match records with common opponents
            temp_h2h_common = temp_df_own.merge(temp_df_oppo, 
                                            left_on = 'team_tri_against_ow', 
                                            right_on = 'team_tri_against_oow', how='inner')
            if (idx_team == 0) & (idx_opponent == 0): 
                # For the very first iteration
                h2h_common = temp_h2h_common
            else:
                h2h_common = pd.concat([h2h_common, temp_h2h_common]).reset_index(drop = True)
    # Record date last available
    h2h_common["gameDate_last"] = iter_time

    return h2h_common

# Pairwise RPI calculation
def pairwise_h2h(df_box_exnt):
    # own success
    df_own = df_box_exnt \
        .groupby(['team_tri_for']) \
        .agg(
            {
                "rwin":"last",
                "rgame":"last"
            }
        ).reset_index()
    df_own["wp_own"] = df_own.rwin / df_own.rgame
    df_own.set_index("team_tri_for", inplace = True)

    # Opponents' success, accumulated
    df_ow = df_box_exnt \
        .groupby(['team_tri_for', 'team_tri_against']) \
        .agg(
            {
                "rwin_oppo":"last",
                "rgame_oppo":"last",
            }
        ).reset_index().groupby(['team_tri_for']) \
        .agg(
            {
                "rwin_oppo":"sum",
                "rgame_oppo":"sum",
            }
        ).reset_index()
    df_ow["wp_ow"] = df_ow.rwin_oppo / df_ow.rgame_oppo
    df_ow.set_index("team_tri_for", inplace = True)

    # Opponents common opponents success
    idx_h2h_pivot = 'team_tri_for_ow'
    df_oow = dff_h2h_com.groupby([idx_h2h_pivot]) \
        .agg(
            {
                'game_won_h2h_ow':'sum', 
                'game_played_h2h_ow':'sum'
            }
        ).reset_index()
    df_oow["wp_oow"] = df_oow.game_won_h2h_ow / df_oow.game_played_h2h_ow
    df_oow.set_index(idx_h2h_pivot, inplace = True)

    df_own = df_own.join(df_ow)
    df_own = df_own.join(df_oow)

    df_own['rpi'] = (df_own.wp_own * 0.25) \
                    + (df_own.wp_ow * 0.5) \
                    + (df_own.wp_oow * 0.25) 
    df_own.drop(columns = ['rgame', 'rwin', 
                            'rwin_oppo', 'rgame_oppo',
                            'game_won_h2h_ow', 'game_played_h2h_ow'
                        ], 
                inplace = True)
    df_own.sort_values(by = 'rpi')
    return df_own