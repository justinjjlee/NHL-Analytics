# Game sequence level microdata processing
import os
import pandas as pd
import numpy as np
import glob
from datetime import datetime
#import seaborn as sns
# get available processor count
import multiprocessing as mp
num_worker = int(mp.cpu_count()/5*4)

from joblib import Parallel, delayed

def process_sequence(iter_game, df):
    '''
    Generates dataframe of sequence 
    '''
    # For each game, process each data
    # For each game of the dataframe 
    df_iter = df[df.gameIdx == iter_game]

    # Files to be saved
    df_agent = []

    # Pull faceoffs 
    list_faceoff = df_iter.query(
        'eventTypeId == "FACEOFF"'
    )

    # Create offense-based metrics
    # Home team is marked wtih the event code
    idx_home = df_iter.eventCode.iloc[0][0:3]
    # Away team code get
    teamcode = [iter_game[0:3], iter_game[4:7]] # Get teh list of team names
    [idx_away] = [iter for iter in teamcode if iter != idx_home]
    # Incorrect way
    #idx_home = iter_game[0:3]
    #idx_away = iter_game[4:7]
    
    # For each faceoff
    for iter_faceoff, idx_faceoff in enumerate(list_faceoff.index):
        # Get index of the following faceoff
        try:
            idx_faceoff_next = list_faceoff.index[iter_faceoff + 1]
            idx_faceoff_sequence = np.arange(idx_faceoff, idx_faceoff_next)
        except: # If the last faceoff of the game, go all the way to the last entry
            idx_faceoff_sequence = np.arange(idx_faceoff, df_iter.index.max())

        # --------------------------------------------------------------------------------------------------------
        # Get the measurements
        # Selecting the series based on the faceoff
        iter_decision = df.loc[(df.gameIdx == iter_game) & (df.index.isin(idx_faceoff_sequence)), :]

        # Series settings
        agent_period = iter_decision.period.iloc[0] # Period number
        # Time - series lengh
        agent_serieslength = np.max(iter_decision.periodTime) - np.min(iter_decision.periodTime)
        # Zone start location
        #   python returns as list
        [agent_zonestartloc] = iter_decision[iter_decision.event == 'Faceoff'].rinkSide_play_relative
        
        # Create elemnts
        #   Based on the home team
        #   Faceoff win: 1 if home team won, 0 if visiting team won
        agent_faceoff = 1 if iter_decision.iloc[0].team_staff == 'home' else 0
        #   Home-agent type when faceoff: defense or offense
        #       1 if opponent (defense) 0 ()
        agent_faceoff_posture = 'offense' if iter_decision.iloc[0].rinkSide_play_relative == 'opponent' else 'defense'
        # How many zones used for the action
        #   If more than one, it means that the zone exit happened
        #   0 if only one zone, 1 otherwise (zone changed)
        agent_zonestay = 0 if len(np.unique(iter_decision[iter_decision.eventTypeId != 'STOP'].rinkSide_play)) == 1 else 1

        # Sequence where goal happened 
        agent_goal = 0
        #   1 if goal exist, 0 otherwise
        #       Sequence where goal made by home (1) or away (-1) or no goal exist
        # Tracking goal information
        agent_goal_type = "none"
        agent_goal_fin  = 0
        # non-even goal (6-5 with goalie pulled)
        #   powerplay goal or Empty net goal (maybe its coded out)
        # strength_name: [nan, 'Even', 'Power Play', 'Short Handed']

        if ("GOAL" in list(iter_decision.eventTypeId)): # If a goal happened in the sequence
            # Identify event record of goal: There should be one only
            iter_goalseries = iter_decision.loc[iter_decision.eventTypeId == 'GOAL', :]

            # (_) Get information on who made the goal
            if (iter_goalseries['team_staff'].iloc[0] == 'home'):
                agent_goal = 1
            else:
                agent_goal = -1
            
            # (_) Get the goal type
            agent_goal_type = iter_goalseries['strength_name'].iloc[0]

            # (_) Information if the goal is game-winning
            agent_goal_fin  = 1 if (iter_goalseries['gameWinningGoal'].iloc[0] == True) else 0
        else: #no action needed
            None

        ## Team - offense perspective
        
        # Team identity of the offense
        #   Default is NEUT given they are starting at neutral location
        agent_offense_id = "NEUT"
        if (agent_zonestartloc == "own"):
            agent_offense_id = idx_away
        elif (agent_zonestartloc == "opponent"):
            agent_offense_id = idx_home

        # Has offense won the faceoff, default is 0 (i.e. neurtral zone start)
        agent_offense_faceoff_won = 0
        if agent_offense_id == idx_home:
            if agent_faceoff == 1:
                agent_offense_faceoff_won = 1
            else:
                None
        else: #(df_agent.idx_offense[iter] == df_agent.idx_away[iter]):
            # Else if the away team is the offensive zone team
            if agent_faceoff == 0:
                agent_offense_faceoff_won = 1
            else:
                None

        # SOmething to go along with the possession duration:
        #   Shot count for the period (subtract the cumulative count)

        # --------------------------------------------------------------------------------------------------------
        # Create dataframe to save 
        tempdf_agent = pd.DataFrame(
            data = {
                'gameIdx':iter_decision.gameIdx.iloc[0],
                'team_home' : idx_home,
                'team_opponent' : idx_away,
                'period': agent_period,
                'time_length': agent_serieslength,
                'zone_faceoff_home': agent_zonestartloc,
                'zone_faceoff_offense':agent_offense_id,
                'faceoff_won_home': agent_faceoff,
                'faceoff_won_offense':agent_offense_faceoff_won,
                'posture': agent_faceoff_posture,
                'zone_change': agent_zonestay,
                'goal': agent_goal,
                'goal_strengh':agent_goal_type,
                'goal_finalwin':agent_goal_fin
            }, index = [idx_faceoff]
        )

        # Save the agent work
        if len(df_agent) == 0: # The first element
            df_agent = tempdf_agent
        else: # append the existing
            df_agent = pd.concat([df_agent, tempdf_agent])
    
    return df_agent