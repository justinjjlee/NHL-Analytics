import pickle5 as pickle
# If you get error of pickle protocol 5
import numpy as np 
import pandas as pd
import operator

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
    # No need for this data points, except one
    #df_temp_plr_log = pd.json_normalize(liveData_player['position'])[['abbreviation', 'code', 'name', 'type']]
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
    df_tst = element_sparce(iter, val.index[idx])
    if idx == 0:
        df_res = df_tst
    else:
        df_res = pd.concat([df_res, df_tst], ignore_index = True)
    df_res.set_index('index', inplace = True)
    parced_df = pd.concat([parced_df, df_res], axis = 1);
    # one game done.

    # ========== Game-level information ==========================================
    # Match the period and game information for reference
    # COPY PASTE THE WORK BELOW AFTER CHANGING THE REFERENCE>>>>>>>>
    tst_game = dat.liveData[2]; # later put in for loop
    df_rink = pd.json_normalize(tst_game['linescore']['teams']['home'])
    df_rink['team_location'] = 'home';
    df_rink_away = pd.json_normalize(tst_game['linescore']['teams']['away'])
    df_rink_away['team_location'] = 'away'
    df_rink = pd.concat([df_rink, df_rink_away]).reset_index()
    df_rink.columns = df_rink.columns.str.replace("team.", "")
    df_rink = df_rink[["goals", "shotsOnGoal", "triCode", "location"]]
    df_rink.rename(columns = {'goals':'goals_game', 
                            'shotsOnGoal':'shotsOnGoal_game'}, inplace = True)

    df_rink_period = pd.json_normalize(tst_game['linescore']['periods'])

    str_ary_base = ['periodType', 'startTime', 'endTime', 'num', 'ordinalNum']
    str_ary_base_home = str_ary_base + ['home.goals', 'home.shotsOnGoal', 'home.rinkSide']
    str_ary_base_away = str_ary_base + ['away.goals', 'away.shotsOnGoal', 'away.rinkSide']

    df_rink_period_home = df_rink_period[str_ary_base_home]

    df_rink_period_home['location'] = 'home'
    df_rink_period_home.columns = df_rink_period_home.columns.str.replace("home.", "")
    print('errormark')
    # Process for the away team
    df_rink_period_away = df_rink_period[str_ary_base_away]
    df_rink_period_away['location'] = 'away'
    df_rink_period_away.columns = df_rink_period_away.columns.str.replace("away.", "")

    df_rink_period = pd.concat([df_rink_period_home, df_rink_period_away])
    df_rink_period.rename(columns = {'goals':'goals_period', 
                                    'shotsOnGoal':'shotsOnGoal_period'}, 
                        inplace = True)
    print('errormark')
    # Merge it all for having location information with period and game-level information
    df_rink = df_rink_period.merge(df_rink, how = 'left', on = "location")

    # Change the location column name for team
    df_rink.rename(columns = {'location':'team_staff', 
                            'rinkSide':'rinkSide_team'}, 
                    inplace = True)

    df_rink = df_rink[['ordinalNum', 'goals_period', 'goals_game', 
                        'shotsOnGoal_period', 'shotsOnGoal_game', 
                        'rinkSide_team', 'team_staff', 'triCode']]

    parced_df = parced_df.merge(df_rink, how = 'left', 
                                on = ['ordinalNum', 'triCode'])

    # Get rink-side information
    parced_df['rinkSide_play'] = ['left' if (iter < 0) else 
                                'right' if (iter > 0) else 
                                'center' for iter in parced_df.coordinates_x];
    parced_df['rinkSide_play_relative'] = ['own' if (val == parced_df.rinkSide_team[idx]) else
                                            'center' if (val == 'center') else
                                            'opponent' for (idx, val) in enumerate(parced_df.rinkSide_play)]

    # Pulling player-level information for the game
    df_player_info = plr_process_info(dict_game)
    df_player_game = plr_process_liveData(dict_live)
    df_player = df_player_info.merge(df_player_game, how = 'left', 
                                    on = ['id_player', 
                                            'fullName', 
                                            'player_pos_abv'])

    # Wrapping up
    # Create unique identifier for the game
    #    Create game index using vals
    df_gameinfo = parced_df.groupby(['triCode', 'team_staff'])['dateTime'].max().reset_index()
    df_gameinfo.sort_values('team_staff', inplace = True)

    temp_str_date = df_gameinfo.dateTime[1]
    temp_str_date = temp_str_date[0:10]
    # gamecode - home team, away team, date of the game
    temp_str_code = df_gameinfo.triCode[1] + '_' + df_gameinfo.triCode[0] + '_' + temp_str_date;

    # Add indicators
    parced_df['gameIdx'] = temp_str_code
    df_player['gameIdx'] = temp_str_code

    # Create column to identify regular season / post season
    # Merge left of the action level data with the above, using
    #   Ordinal number (period information) as well as team tricode
    temp_season_type = dict_game['game']['type']
    parced_df['seasonIdx'] = temp_season_type
    df_player['seasonIdx'] = temp_season_type

    return parced_df, df_player