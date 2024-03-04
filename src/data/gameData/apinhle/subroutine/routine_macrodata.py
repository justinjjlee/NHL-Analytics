# Routine to pull and update the latest 
#   Game-level team- and player-level statistics


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
games.drop(columns=[
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
], inplace=True)

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
        df_box_team.append(temp_team)
        for iter_pos in ['forwards','defense','goalies']:
            tempdata = data['boxscore']['playerByGameStats'][iter_team][iter_pos]
            tempdata = pd.json_normalize(tempdata)
            tempdata['teamid'] = teamid
            tempdata['teamtri'] = teamtri
            tempdata['gameid'] = iter_game
            tempdata['teamloc'] = teamloc
            df_box_player.append(tempdata)

df_box_player = pd.concat(df_box_player)
df_box_team = pd.concat(df_box_team)

# Drop unnecesary columns
df_box_team.drop(columns=['logo', 'name.default', 'name.fr'], inplace=True)
df_box_player.drop(
    columns=[
        'name.cs','name.fi', 'name.sk', 'name.de', 'name.sv', 'name.es',
        'name.default'
    ],
    inplace = True
)

# Load the previous data
df_player = pd.read_csv(f"./latest/{iter_year}_box_player.csv")
df_team   = pd.read_csv(f"./latest/{iter_year}_box_team.csv")
# Attach the past data
df_box_player = pd.concat([df_player, df_box_player], ignore_index=True)
df_box_team   = pd.concat([df_box_team, df_team], ignore_index=True)
# ---------------------------------------------------
# Save data
# Box scores for each game for each team
df_box_player.to_csv(f"./latest/{iter_year}_box_player.csv", index=False)
df_box_team.to_csv(f"./latest/{iter_year}_box_team.csv", index=False)

# Save 'games" and "game_list"
games.to_csv(f"./latest/{iter_year}_gamelist_raw.csv", index=False)
game_list.to_csv(f"./latest/{iter_year}_gamelist.csv", index=False)