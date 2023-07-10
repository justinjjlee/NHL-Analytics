import pickle as pickle
import requests
import numpy as np 
import pandas as pd
import operator

import sys
# Add github location of github file location (keep it as local file)

# import custom functions written for the data processing
from NHL_gameData_process_functions import *
from NHL_gameData_process_sequence import *

import os
dir_data = os.getcwd() # Current working directory of the server

# Object to pull data from NHL official API
class nhl_api_datapull:
    def __init__(self, year, dir_data):
        self.year = year

    def datacall(self, postseason=False):     
        if postseason:
            # By default don't include
            ssn = ['02']
        else: 
            ssn = ['02', '03']; # Regular season and playoff, respectively
        
        max_game_ID = 3000; # just to be conservative in case there are data not accounted
        for season_type in ssn:
            game_data = []
            for i in range(0,max_game_ID):
                r = requests.get(url='http://statsapi.web.nhl.com/api/v1/game/'
                    + self.year + season_type +str(i).zfill(4)+'/feed/live')
                data = r.json()
                game_data.append(data)
            # Save as pickle file
            with open(dir_data + '/raw/' + self.year + '_' + season_type + 'FullDataset.pkl', 'wb') as f:
                pickle.dump(game_data, f, pickle.HIGHEST_PROTOCOL)
            # Data pull done, let me know.
            print('Done! - year ' + self.year + ' and season type: ' + season_type)
        # no object to return
        return None

class nhl_dataproc_baseline:
    def __init__(self, year, dir_data):
        # Year and season to process
        self.str_year = year
        self.dir_data = dir_data

    def dataproc(self, idx_seasontype='02'):
        # idx_season 02 is regular, 03 postseason

        # Define some data location source and 
        str_dirc = self.dir_data + '/raw/' # Raw data location
        str_data = self.str_year + '_' + idx_seasontype + 'FullDataset.pkl' # name data file to pull
        str_dirc_save = self.dir_data + '/cleaned/' # cleaned data

        with open(str_dirc + str_data, 'rb') as f:
            game_data = pickle.load(f)
        dat = pd.DataFrame(game_data)

        # Process data
        # only including with those of data points
        dat_data = dat.dropna(subset = ["gameData"]).reset_index();
        # Total computing game
        num_tot_games = dat_data.shape[0] - 1

        #data pull and enumerate - initialize entries
        df_game, df_player, df_box = process_game(dat_data.iloc[0, :]['liveData'], dat_data.iloc[0, :]['gameData'])

        for index, row  in dat_data.iterrows(): # The first game has been recorded
            # Skip the first observation, already have it 
            if index == 0:
                None
            else:
                # Check to see if the data is valid
                #   There are posponed games that still have placeholders:
                if row['liveData']['boxscore']['officials'] == []:
                    # If there is no information on the officials, 
                    #   we assume there was no games played
                    continue
                # move on to the next row
                else:
                    None
                # Process the data
                df1, df2, df3 = process_game(row['liveData'], row['gameData'])

                # Append the necessary data
                df_game = df_game.append(df1);
                df_player = df_player.append(df2);
                df_box = df_box.append(df3);
                #print(f"Row {index} of {num_tot_games} completed", end='\r')
                #progressBar(index, num_tot_games, barLength = 20)

        # Save the data
        df_game.to_csv(str_dirc_save + str_data[:7] + '_game.csv', index = False)
        df_player.to_csv(str_dirc_save +  str_data[:7] + '_player.csv', index = False)
        df_box.to_csv(str_dirc_save + str_data[:7] + '_box.csv', index = True)

        return df_game, df_player, df_box
    

# Team success measurement
class nhl_dataproc_teamsuccess:
    def __init__(self, year, dir_data):
        self.str_year = year
        self.dir_data = dir_data

    def dataproc(self, idx_seasontype='02'):
        # idx_season 02 is regular, 03 postseason

        # inital name directiory of data 
        str_dirc = self.dir_data + f"/cleaned/{self.str_year}"
        str_dirc_save = self.dir_data + f"/processed/{self.str_year}"

        # Pull the latest data for process
        df_box = pd.read_csv(str_dirc + f"_{idx_seasontype}_box.csv", 
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
        #exec(open(str_dir_sourceCode + "/evaluation_winStreak_plot.py").read())
        # ...........................................................................
        # output data
        #   (a) game-level stats
        df_box_exnt['yr_season'] = self.str_year
        df_box_exnt.to_csv(str_dirc_save + "_02_box.csv")
        #   (b) team-level stats for each season
        dfteams['yr_season'] = self.str_year
        dfteams.to_csv(str_dirc_save + "_02_team.csv")

# Data sequence creation
#   NOTE: This process use parallel process. Make sure to adjust your available process
class nhl_dataproc_sequence:
    def __init__(self, year, dir_data):
        # Year and season to process
        self.str_year = year
        self.dir_data = dir_data

    def dataproc(self, idx_seasontype='02'):
        # inital name directiory of data 
        str_dirc = self.dir_data + f"/cleaned/{self.str_year}"
        str_dirc_save = self.dir_data + f"/processed/{self.str_year}"

        # Pull processed data
        df = pd.read_csv(str_dirc+f"/_{idx_seasontype}_game.csv")

        # ==================== Data processing step ====================
        # Convert time to python-datetime format
        df["periodTime"] = [datetime.strptime(iterconvert,'%M:%S') for iterconvert in df.periodTime]
        # Calculating total seconds require python datetime format to be timedelta
        df["periodTime"] = df["periodTime"] - datetime.strptime("00:00",'%M:%S')
        # Convert the period information to be in seconds
        df["periodTime"] = df.periodTime.dt.total_seconds()

        # Identify games to be evaluated
        list_gameix = df.gameIdx.unique()

        # ==================== Extract features ====================
        # For each year of data process, 
        # Parallel process the data
        outdf_stack = Parallel(n_jobs=num_worker)(delayed(process_sequence)(iter_game, df) for iter_game in list_gameix)
        # Combine dataframe
        df_test = pd.concat(outdf_stack)

        # ==================== Save data ====================
        df_test.to_csv(str_dirc_save+f"/_{idx_seasontype}_possession_sequence.csv", index=False)