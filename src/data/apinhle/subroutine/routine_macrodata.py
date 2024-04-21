# Routine to pull and update the latest 
#   Game-level team- and player-level statistics

import pandas as pd
import numpy as np

# Functions to process box scores
from subroutine.procs_boxscore import *

# Team codes for the list pull
teamcode = pd.read_csv("teamlist.csv")

# ---------------------------------------------------
# Pull team/game lists of the games for the season

# Varibles to be in the loop
df = []

for iter_team in teamcode.tricode:
    iter_sesn = str(iter_year) + str(iter_year+1)

    r = requests.get(url='https://api-web.nhle.com/v1/club-schedule-season/'
                    + iter_team + "/" + iter_sesn)
    data = r.json()
    data = pd.json_normalize(data['games'])
    # Filter out columns and rows

    # For now, remove all future games
    data = data.loc[data.gameState == "OFF"]

    # Append to save
    df.append(data)

# Convert to dataframe to get the list
games = pd.concat(df).sort_values(by='startTimeUTC')
#len(games)
# Drop duplicated records (double-counted if pulling all teams)
games = games.drop_duplicates(subset="id").reset_index(drop=True)
#len(games)
# Drop unnecessary columns
drop_columns = [
    'tvBroadcasts', 'gameCenterLink', 'venue.default',
    'awayTeam.placeName.default', 'awayTeam.logo', 'awayTeam.darkLogo', 
    'awayTeam.awaySplitSquad', 'homeTeam.placeName.default',
    'homeTeam.logo', 'homeTeam.darkLogo',
    'homeTeam.homeSplitSquad', 'homeTeam.hotelLink', 'homeTeam.hotelDesc',
    'winningGoalie.firstInitial.default', 'winningGoalie.lastName.default',
    'winningGoalScorer.firstInitial.default',
    'winningGoalScorer.lastName.default', 'venue.es', 'venue.fr',
    'awayTeam.airlineLink', 'awayTeam.airlineDesc',
    'winningGoalie.lastName.cs', 'winningGoalie.lastName.sk',
    'winningGoalScorer.lastName.cs', 'winningGoalScorer.lastName.fi',
    'winningGoalScorer.lastName.sk', 'threeMinRecap',
    'awayTeam.placeName.fr', 'awayTeam.hotelLink', 'awayTeam.hotelDesc',
    'homeTeam.placeName.fr', 'threeMinRecapFr', 'winningGoalie.lastName.fi',
    'homeTeam.airlineLink', 'homeTeam.airlineDesc', 'ticketsLink',
    'awayTeam.radioLink', 'homeTeam.radioLink', 'awayTeam.promoLink',
    'awayTeam.promoDesc', 'specialEvent.default',
    'winningGoalScorer.lastName.de', 'winningGoalScorer.lastName.es',
    'winningGoalScorer.lastName.sv', 'homeTeam.promoLink',
    'homeTeam.promoDesc', 'specialEvent.fr', 'specialEventLogo'
]
for iter in drop_columns:
    try:
        games.drop(columns=iter, inplace=True)
    except:
        None

columns_select = [
    'id',
    'gameDate',
    'startTimeUTC',
    'homeTeam.abbrev',
    'homeTeam.id',
    'awayTeam.abbrev',
    'awayTeam.id',
    'homeTeam.score',
    'awayTeam.score',
    'gameOutcome.lastPeriodType'
]
game_list = games[columns_select]
game_list.columns = [
    "gameid",
    "date", "time_start",
    "tricode_for", "id_for",
    "tricode_against", "id_against",
    "metric_score_for", "metric_score_against",
    "period_ending"
]

# Create list of team abbreviation and id for matching
reference_team = game_list.groupby(['tricode_for']).id_for.min().reset_index()
reference_team.columns = ['tricode', 'id']

# Create winning team tri-code columns
for count, row in game_list.iterrows():
    cond_iter = row["metric_score_for"] > row["metric_score_against"]
    if cond_iter:
        game_list.loc[count, "tricode_winteam"] = row["tricode_for"]
    else:
        game_list.loc[count, "tricode_winteam"] = row["tricode_against"]

for count, row in games.iterrows():
    cond_iter = row["homeTeam.score"] > row["awayTeam.score"]
    if cond_iter:
        games.loc[count, "tricode_winteam"] = row["homeTeam.abbrev"]
    else:
        games.loc[count, "tricode_winteam"] = row["awayTeam.abbrev"]
        
# ---------------------------------------------------
# Pull game-level statistics
#   In order to save the API pull time, I only need to pull records
#   Not currently pulled and saved
game_list_last = pd.read_csv(f"./latest/{iter_year}_gamelist.csv")
# Pick up games with newest data points
inx_gamesnodata = game_list.loc[~game_list["gameid"].isin(game_list_last["gameid"]), "gameid"]

# Pull data
# For each game id, pull play-by-play
df_box_player = []
df_box_team   = []

for iter_game in inx_gamesnodata: #game_list["gameid"]: # if pulling all
    # Maybe wait a minute?

    # Call data
    r = requests.get(url='https://api-web.nhle.com/v1/gamecenter/'
                        + str(iter_game) + "/boxscore")
    data = r.json()

    team_stat_empty = []
    # Get game stats
    gamedate = data['gameDate']

    # Process data
    for iter_team in ['awayTeam','homeTeam']:
        teamloc = iter_team[0:4]
        # Get the team box
        temp_team = pd.json_normalize(data[iter_team])
        temp_team['gameid'] = iter_game
        temp_team['teamloc'] = teamloc
        # Save team information
        teamid  = temp_team['id']
        teamtri = temp_team['abbrev']

        # Append the team stats
        team_stat_empty.append(temp_team)

        # Player-specific stats
        #   3/29/2024 - API structure is changing so frequently, and currently I don't find a use for ti
        #       Commenting out until the next season when the data will be useful and API structure more consistent
        #for iter_pos in ['forwards','defense','goalies']:
            #tempdata = data['playerByGameStats'][iter_team][iter_pos]
            #tempdata = pd.json_normalize(tempdata)
            #tempdata['teamid'] = teamid
            #tempdata['teamtri'] = teamtri
            #tempdata['gameid'] = iter_game
            #tempdata['teamloc'] = teamloc
            #tempdata['gameDate'] = data['gameDate']
            #tempdata['seasonIdx'] = data['gameType']
            #df_box_player.append(tempdata)

    # Data concat to be dataframe
    team_stat = pd.concat(team_stat_empty).reset_index(drop=True)
    team_stat = team_stat[['abbrev', 'gameid', 'teamloc', 'score']]

    # Who won the game
    team_stat['win'] = 0
    if team_stat.loc[0, "score"] > team_stat.loc[1, "score"]:
        # Home team won
        team_stat.loc[0, 'win'] = 1
    else:
        # Away team won
        team_stat.loc[1, 'win'] = 1

    # Add game info
    team_stat['gameIdx'] = team_stat.loc[0, 'abbrev'] + '_' + team_stat.loc[1, 'abbrev'] + '_' + data['gameDate']

    team_stat['seasonIdx'] = data['gameType']
    team_stat['gameDate'] = data['gameDate']
    team_stat['gameEnd'] = data['gameOutcome']['lastPeriodType']

    # Attach the box score
    team_boxstats = pd.json_normalize(data['summary']['teamGameStats'])
    team_boxstats.set_index('category', inplace=True)
    team_boxstats = team_boxstats.transpose().reset_index(drop=True)
    # Join the data
    team_stat = team_stat.join(team_boxstats)

    # Save data
    df_box_team.append(team_stat)

#df_box_player = pd.concat(df_box_player)
df_box_team = pd.concat(df_box_team)

# Reset index
df_box_team.set_index("gameIdx", inplace=True)

# Rename columns
df_box_team.rename(
    columns = {
        "abbrev":"team_tri",
        "score": "goals",
        "sog":"shots",
        "blockedShots":"shots_blocked",
        "giveaways":"poss_giveaway",
        "takeaways":"poss_takeaway"
    }, inplace = True
)

df_box_team['gameDate'] = pd.to_datetime(df_box_team['gameDate'])
'''
df_box_player.drop(
    columns=[
        'name.cs','name.fi', 'name.sk', 'name.de', 'name.sv', 'name.es',
        'name.default'
    ],
    inplace = True
)
'''
# Load the previous data
#df_player = pd.read_csv(f"./latest/{iter_year}_box_player.csv",
#                        parse_dates = ['gameDate'], 
#                        index_col = 'gameIdx')
df_team   = pd.read_csv(f"./latest/{iter_year}_box_team.csv",
                        parse_dates = ['gameDate'], 
                        index_col = 'gameIdx')
# Attach the past data
#df_box_player = pd.concat([df_player, df_box_player], ignore_index=True)
df_box_team   = pd.concat([df_team, df_box_team])

# ---------------------------------------------------
# Save data
# Box scores for each game for each team
#df_box_player.to_csv(f"./latest/{iter_year}_box_player.csv")
df_box_team.to_csv(f"./latest/{iter_year}_box_team.csv")

# Save 'games" and "game_list"
games.to_csv(f"./latest/{iter_year}_gamelist_raw.csv", index=False)
game_list.to_csv(f"./latest/{iter_year}_gamelist.csv", index=False)
# ---------------------------------------------------
# Construct team success measurements, and save the data
team_season = nhl_dataproc_teamsuccess(iter_year)
team_season.dataproc(df_box_team)

# ===================================================
# Player data =======================================
# ===================================================
# Player data download - season aggregation, up-to-date
playerstats = []
# For each team
for iter_team in list(teamcode.tricode):
    # Pull seasons played by each  team
    r = requests.get(url=f'https://api-web.nhle.com/v1/roster-season/{iter_team}')
    seasons = r.json()

    # For season download: 
    iter_sesn = str(iter_year) + str(iter_year+1)
        
    try:
        # Pull club stats, only regular seasons for now
        r = requests.get(url=f'https://api-web.nhle.com/v1/club-stats/{iter_team}/{iter_sesn}/2')
        clubstats = r.json()

        temp_df = pd.json_normalize(clubstats["goalies"])
        temp_df['team_tri'] = iter_team
        temp_df["idx_season"] = iter_year
        temp_df["positionCode"] = "G"
        playerstats.append(temp_df)
        temp_df = pd.json_normalize(clubstats["skaters"])
        temp_df['team_tri'] = iter_team
        temp_df["idx_season"] = iter_year
        playerstats.append(temp_df)
    except:
        None
    # Pause to play safe with the API
    time.sleep(3)
    
# Data concatenation
playerstats = pd.concat(playerstats)
# Aggregate and concatenate
col_remove = list(playerstats.filter(regex='firstName'))
col_remove.extend(list(playerstats.filter(regex='lastName')))
col_remove.extend(['headshot'])
col_remove.remove('firstName.default')
col_remove.remove('lastName.default')

playerstats.drop(columns=col_remove, inplace=True)

# Column re-order
first_cols = ['idx_season','team_tri','playerId','firstName.default','lastName.default','positionCode','gamesPlayed']
last_cols = [col for col in playerstats.columns if col not in first_cols]
# save ata
playerstats = playerstats[first_cols+last_cols]
playerstats.to_csv(f"./latest/player_{iter_year}_2.csv", index=False)