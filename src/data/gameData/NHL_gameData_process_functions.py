#import pickle as pickle
# If you get error of pickle protocol 5
import pickle as pickle
import numpy as np 
import pandas as pd
import operator

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedFormatter, FixedLocator

def plr_process_info(gameData_dict):

    tst_game = gameData_dict['players']

    # Search the player ID for rows
    # Using items() + list comprehension
    # Substring Key match in dictionary
    str_player_ids = [key for key, val in tst_game.items() if "ID" in key]

    df_player = []
    for iter, val in enumerate(str_player_ids):
        if iter == 0:
            df_player = pd.json_normalize(tst_game[val])
        else:
            df_player = df_player.append(pd.json_normalize(tst_game[val]))
    df_player = df_player.reset_index().drop(['index', 'link', 'firstName', 
                                                'lastName', 'currentTeam.id', 
                                                'currentTeam.link', 
                                                'primaryPosition.code'], 1)
    df_player.rename(columns = {"id":"id_player", "currentTeam.name":"team_name", 
                                "currentTeam.triCode":"team_triCode", 
                                "primaryPosition.name":"player_pos", 
                                "primaryPosition.abbreviation":"player_pos_abv", 
                                "primaryPosition.type":"player_pos_type"}, 
                    inplace = True)
    return df_player
  
def plr_process_liveData_dfclean(liveData_player):
    # To differentiate and have columns cleanup easier, go through by layer
    #   OUTCOME: columns to be consistent
    # Parse player's info in order
    # (1) Player information
    #['id', 'fullName', 'shootsCatches', 'rosterStatus']
    df_temp_plr = pd.json_normalize(liveData_player['person'])[['id', 'fullName']].rename(columns = {'id':'id_player'})

    # (2) Player position
    df_temp_plr_log = pd.json_normalize(liveData_player['position'])[['abbreviation']].rename(columns = {'abbreviation':'player_pos_abv'})
    df_temp_plr = pd.concat([df_temp_plr, df_temp_plr_log], axis = 1)

    # (3) Player statistics, based on being a goalie or not
    if df_temp_plr['player_pos_abv'][0] == 'G':
        df_temp_stats = pd.json_normalize(liveData_player['stats']['goalieStats'])
        df_temp_plr = pd.concat([df_temp_plr, df_temp_stats], axis = 1)
    elif df_temp_plr['player_pos_abv'][0] == 'N/A':
        None
    else: #others are skaters
        df_temp_stats = pd.json_normalize(liveData_player['stats']['skaterStats'])
        df_temp_plr = pd.concat([df_temp_plr, df_temp_stats], axis = 1)
    # Return dataframe
    return df_temp_plr

def plr_process_liveData(livdata_dict):
    # Initialize space to save data pull
    df_player = [];

    # For each team in the data
    for iter_team in ['away', 'home']:
        # Parse player-level information for each team
        tst_game = livdata_dict['boxscore']['teams'][iter_team]['players']

        # Find IDs of the team
        str_player_ids = [key for key, val in tst_game.items() if "ID" in key]

        # For each team
        for iter, val in enumerate(str_player_ids):
            if ((iter == 0) & (iter_team == 'away')): # The first iteration
                df_player = plr_process_liveData_dfclean(tst_game[val])
            else:
                df_player = df_player.append(plr_process_liveData_dfclean(tst_game[val]))

    return df_player.reset_index().drop(['index'], axis = 1)

# To process player-level information, for those with events.
def element_sparce(element, idx_element):
    if any(pd.isnull(element)):
        return element
    else:
        df_tps = pd.json_normalize(element)
        #df_temp = df_tps[['playerType', 'seasonTotal', 'player.fullName']]
        #df_temp = df_temp.rename(columns = {'playerType':'plays', 'seasonTotal':'count_total', 'player.fullName':'name'})
        df_temp = df_tps[['playerType', 'player.fullName']]
        df_temp = df_temp.rename(columns = {'playerType':'plays', 'player.fullName':'name'})
        idx_assist = 1
        for idx, iter in enumerate(df_temp.plays):
            # Rename with Assist 1 and 2
            if df_temp.plays[idx] == 'Assist':
                df_temp.plays[idx] = 'Assist_' + str(idx_assist)
                idx_assist += 1
        #  df_temp.pivot_table('name', [], 'plays', aggfunc='first')
        df_temp_df = df_temp.set_index(['plays']).unstack()
        df_temp_df.index = df_temp_df.index.map('_'.join)
        df_res = pd.DataFrame(data = [list(df_temp_df)], columns = df_temp_df.index)

    df_res['index'] = idx_element
    return df_res

def trial_rinkside(dict_live):
    df_rink = pd.json_normalize(dict_live['linescore']['teams']['home'])
    df_rink['team_location'] = 'home';
    df_rink_away = pd.json_normalize(dict_live['linescore']['teams']['away'])
    df_rink_away['team_location'] = 'away'
    df_rink = pd.concat([df_rink, df_rink_away]).reset_index()
    df_rink.columns = df_rink.columns.str.replace("team.", "")
    df_rink = df_rink[["goals", "shotsOnGoal", "triCode", "location"]]
    df_rink.rename(columns = {'goals':'goals_game', 
                            'shotsOnGoal':'shotsOnGoal_game'}, inplace = True)

    df_rink_period = pd.json_normalize(dict_live['linescore']['periods'])

    #str_ary_base = ['periodType', 'startTime', 'endTime', 'num', 'ordinalNum']
    str_ary_base = ['periodType', 'num', 'ordinalNum']
    str_ary_base_home = str_ary_base + ['home.goals', 'home.shotsOnGoal', 'home.rinkSide']
    str_ary_base_away = str_ary_base + ['away.goals', 'away.shotsOnGoal', 'away.rinkSide']

    df_rink_period_home = df_rink_period[str_ary_base_home]

    df_rink_period_home.loc[:, 'location'] = 'home'
    df_rink_period_home.columns = df_rink_period_home.columns.str.replace("home.", "")

    # Process for the away team
    df_rink_period_away = df_rink_period[str_ary_base_away]
    df_rink_period_away.loc[:, 'location'] = 'away'
    df_rink_period_away.columns = df_rink_period_away.columns.str.replace("away.", "")

    df_rink_period = pd.concat([df_rink_period_home, df_rink_period_away])
    df_rink_period.rename(columns = {'goals':'goals_period', 
                                    'shotsOnGoal':'shotsOnGoal_period'}, 
                        inplace = True)

    # Merge it all for having location information with period and game-level information
    df_rink = df_rink_period.merge(df_rink, how = 'left', on = "location")

    # Change the location column name for team
    df_rink.rename(columns = {'location':'team_staff', 
                            'rinkSide':'rinkSide_team'}, 
                    inplace = True)

    df_rink = df_rink[['ordinalNum', 'goals_period', 'goals_game', 
                        'shotsOnGoal_period', 'shotsOnGoal_game', 
                        'rinkSide_team', 'team_staff', 'triCode']]

    return df_rink

def process_game(dict_live, dict_game):

    # Pull the live game data
    parced_df = pd.json_normalize(dict_live, record_path=['plays', 'allPlays'])
    # Correct names of columns
    parced_df.columns = parced_df.columns.str.replace("result.", "").str.replace("about.", "").str.replace("team.", "").str.replace(".", "_")

    # For those with actions, need to process those with more information in nested dictionaries
    val = parced_df.players[map(operator.not_, pd.isnull(parced_df.players))]

    # Go through the events with the actions
    df_res = []
    for idx, iter in enumerate(val):
        # There are cases where the type of players are missing from the data
        # skip for now
        #   If both players are "unknown", this process throws out error
        #       Single "unknown" creates column "name_Unknown"
        try:
            df_tst = element_sparce(iter, val.index[idx])
            if idx == 0:
                df_res = df_tst
            else:
                df_res = pd.concat([df_res, df_tst], ignore_index = True)
        except:
            None
    df_res.set_index('index', inplace = True)
    parced_df = pd.concat([parced_df, df_res], axis = 1);

    # ========== Game-level information ==========================================
    # Match the period and game information for reference

    # get the side of the rink among other information
    try: # fetching rink information in summary dictionary
        df_rink = trial_rinkside(dict_live)
        parced_df = parced_df.merge(df_rink, how = 'left', 
                                on = ['ordinalNum', 'triCode'])
        # Get rink-side information
        parced_df['rinkSide_play'] = ['left' if (iter < 0) else 
                                    'right' if (iter > 0) else 
                                    'center' for iter in parced_df.coordinates_x];
        parced_df['rinkSide_play_relative'] = ['own' if (val == parced_df.rinkSide_team[idx]) else
                                            'center' if (val == 'center') else
                                            'opponent' for (idx, val) in enumerate(parced_df.rinkSide_play)]

    except: # if there is no information
        None

    # Pulling player-level information for the game
    df_player_info = plr_process_info(dict_game)
    df_player_game = plr_process_liveData(dict_live)
    df_player = df_player_info.merge(df_player_game, how = 'left', 
                                    on = ['id_player', 
                                        'fullName', 
                                        'player_pos_abv'])

    # get home vs. away team label
    str_awayteam = dict_game['teams']['away']['triCode']
    str_hometeam = dict_game['teams']['home']['triCode']
    parced_df['team_staff'] = np.nan
    parced_df['team_staff'][parced_df.triCode == str_awayteam] = 'away'
    parced_df['team_staff'][parced_df.triCode == str_hometeam] = 'home'

    # Wrapping up & create unique identifier for the game
    df_gameinfo = parced_df.groupby(['triCode', 'team_staff'])['dateTime'].max().reset_index().sort_values('team_staff')

    # Pull team statistics - box score
    df_boxscore = pd.json_normalize(dict_live['boxscore']['teams']['away']['teamStats']['teamSkaterStats'])
    df_boxscore['team_tri'] = dict_live['boxscore']['teams']['away']['team']['triCode']
    df_boxscore['team_vis'] = 'away'
    # Dame for the home team
    dftemp = pd.json_normalize(dict_live['boxscore']['teams']['home']['teamStats']['teamSkaterStats'])
    dftemp['team_tri'] = dict_live['boxscore']['teams']['home']['team']['triCode']
    dftemp['team_vis'] = 'home'
    # Combine the two information
    df_boxscore = df_boxscore.append(dftemp).reset_index(drop = True)

    # ................... Add additional information from play-level data ........
    # Set index of team names to join with other columns
    df_boxscore = df_boxscore.set_index('team_tri')

    # Collect additional information from the play-level data
    parced_df_action = parced_df.dropna(subset = ["triCode"])
    boxscore_detail = parced_df_action.groupby(['triCode', 'eventTypeId']).event.count()

    # Aggregation of events to get event counts that are not traced in game-level info
    df_addition = pd.DataFrame(boxscore_detail).reset_index().pivot(index = "triCode", columns = "eventTypeId", values = 'event')
    # join the two data
    df_boxscore = df_boxscore.join(df_addition); # Join the two
    # Select columns needed
    # There will be games without the statistics, go through the list and make sure to create zero for measures not in.
    temp_collist = ['goals', 'shots', 'BLOCKED_SHOT', 'MISSED_SHOT','TAKEAWAY', 'GIVEAWAY', 'HIT', 'FACEOFF','powerPlayOpportunities','powerPlayGoals','PENALTY','pim','team_vis'];
    for iter in temp_collist:
        if iter not in df_boxscore.columns:
            df_boxscore[iter] = 0
        else:
            None # do nothing

    df_boxscore = df_boxscore[temp_collist]

    # Select and rename columns
    df_boxscore.rename(columns={'BLOCKED_SHOT':'shots_blocked', 'MISSED_SHOT':'shots_missed','TAKEAWAY':'poss_takeaway', 'GIVEAWAY':'poss_giveaway', 'HIT':'hits', 'FACEOFF':'poss_faceoff',
            'powerPlayOpportunities':'pp_count','powerPlayGoals':'pp_goals','PENALTY':'pp_penalty','pim':'pp_pmi'}, inplace = True)
    # Finalize the dataframe 
    df_boxscore.reset_index()

    # FIll missing values as zero, such as penalty information
    df_boxscore.pp_penalty.fillna(0, inplace = True)

    # Print game results column, who won.
    df_boxscore['win'] = 0
    df_gr = df_boxscore[['goals', 'team_vis']]
    if df_gr.goals[0] > df_gr.goals[1]: # away team wins
        df_boxscore.loc[df_gr.index[0], 'win'] = 1
    elif df_gr.goals[0] < df_gr.goals[1]:
        df_boxscore.loc[df_gr.index[1], 'win'] = 1
    else: # tie, with shootout - split the game
        df_boxscore['win'] = 0.5
    # Note that the point system in NHL is different from the official point syste.
    #   I standardize the measure such that each games after regulation is treated equally.

    # Date of the game is measured at the start of the first period of the game.
    #   Only exclude date time - first 9 string entries
    temp_str_date = min(parced_df.loc[parced_df.eventTypeId == 'PERIOD_READY', 'dateTime'])[0:10]
    # gamecode - home team, away team, date of the game
    temp_str_code = df_gameinfo.triCode[1] + '_' + df_gameinfo.triCode[0] + '_' + temp_str_date;

    # Add indicators
    parced_df['gameIdx'] = temp_str_code
    df_player['gameIdx'] = temp_str_code
    df_boxscore['gameIdx'] = temp_str_code

    # Add date
    parced_df['gameDate'] = temp_str_date
    df_player['gameDate'] = temp_str_date
    df_boxscore['gameDate'] = temp_str_date

    # Create column to identify regular season / post season
    # Merge left of the action level data with the above, using
    #   Ordinal number (period information) as well as team tricode
    temp_season_type = dict_game['game']['type']
    parced_df['seasonIdx'] = temp_season_type
    df_player['seasonIdx'] = temp_season_type
    df_boxscore['seasonIdx'] = temp_season_type

    return parced_df, df_player, df_boxscore
  
def data_process(dat):
# Get the data source
    dict_live = dat.liveData[map(operator.not_, pd.isnull(dat.liveData))].reset_index(drop=True)
    dict_game = dat.gameData[map(operator.not_, pd.isnull(dat.liveData))].reset_index(drop=True)

    df_game = []
    df_player = []

    # Go through each game
    for idx, iter in enumerate(dict_game):
        df_game_temp, df_player_temp, df_boxscore_temp = process_game(dict_live[idx], dict_game[idx])
        if idx == 0:
            df_game  = df_game_temp
            df_player = df_player_temp
            df_boxscore = df_boxscore_temp
        else:
            try:
                df_game = pd.concat([df_game, df_game_temp], axis = 0, ignore_index = True)
                df_player = pd.concat([df_player, df_player_temp], axis = 0, ignore_index = True)
                df_boxscore = pd.concat([df_boxscore, df_boxscore_temp], axis = 0, ignore_index = True)
            except: # there are empty entries
                None
    return df_game, df_player, df_boxscore

# Team success measurements ----------------------------------

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
def pairwise_h2h(df_box_exnt, dff_h2h_com):
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