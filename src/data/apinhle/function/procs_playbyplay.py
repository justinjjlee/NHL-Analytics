import pandas as pd
import numpy as np
import requests
import time

def proc_playbyplay_clean(r_iter, row_iter):
    # Cleean proc, process for each game and players associated

    # Organize as json
    data = r_iter.json()
    # Data for play-by-play
    df_plays = pd.json_normalize(data['plays'])

    # Basic set up
    df_plays['idx_season'] = data['season']
    df_plays['seasonIdx'] = data['gameType']
    df_plays['gameDate'] = data['gameDate']
    df_plays['gameid'] = row_iter.gameid
    # The event team to abbreviation
    df_plays["details.eventOwnerTeam"] = \
        [
            data['homeTeam']['abbrev'] 
                if data['homeTeam']['id'] == iter 
                else data['awayTeam']['abbrev'] 
                for iter in df_plays["details.eventOwnerTeamId"]
        ]
    col_to_collect = [   
        # Season and game detail
        'gameid',
        'idx_season',
        'seasonIdx',
        'gameDate',
        # Event time detail
        'eventId', 
        'timeInPeriod', 
        'timeRemaining', 
        'periodDescriptor.number',
        'periodDescriptor.periodType',
        # Event label
        'typeDescKey', 
        'details.reason',
        'details.eventOwnerTeam',
        # Zone detail
        'details.xCoord',
        'details.yCoord', 
        'details.zoneCode', 
        'homeTeamDefendingSide', 
        # Faceoff detail
        'details.losingPlayerId', 
        'details.winningPlayerId', 
        # Shots and blocks
        'details.shotType',
        'details.awaySOG',
        'details.homeSOG', 
        'details.shootingPlayerId',
        'details.blockingPlayerId',
        'details.goalieInNetId', 
        # Scoring detail
        'details.scoringPlayerId',
        'details.assist1PlayerId',
        'details.assist2PlayerId',
        # Hits
        'details.hittingPlayerId', 
        'details.hitteePlayerId', 
        # Penalty
        'details.descKey', 
        'details.duration',
        'details.committedByPlayerId', 
        'details.drawnByPlayerId'
    ]
    # Sometimes the column does not exist 
    try:
        # Select columns needed, if all exist
        df_plays = df_plays[col_to_collect]
    except:
        # If not, then create those columns and fill with NaN
        for col in col_to_collect:
            if col not in df_plays.columns:
                df_plays[col] = np.nan  # Create the column with NaN values

        # Select columns needed
        df_plays = df_plays[col_to_collect]
    
    # Process player information
    df_player_name = pd.json_normalize(data['rosterSpots'])

    # The event team to abbreviation
    df_player_name["team_tri"] = \
        [
            data['homeTeam']['abbrev'] 
                if data['homeTeam']['id'] == iter 
                else data['awayTeam']['abbrev'] 
                for iter in df_player_name["teamId"]
        ]
    # Label roaster date
    df_player_name['gameid'] = row_iter.gameid

    # Select relevant data
    df_player_name = df_player_name[
        [
            "gameid",
            "headshot",
            "team_tri",
            "playerId",
            "sweaterNumber",
            "positionCode",
            "firstName.default",
            "lastName.default",
        ]
    ]

    # In progress... 
    #   Pull player stats and aggregate for each game
    
    return df_plays, df_player_name
