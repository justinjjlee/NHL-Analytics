# 2 for regular season, 3 for playoff
idx_gametype = 2

# Team codes for the list pull
teamcode = pd.read_csv("teamlist.csv")

# For each team
for iter_team in list(teamcode.tricode):
    # Pull seasons played by each  team
    r = requests.get(url=f'https://api-web.nhle.com/v1/roster-season/{iter_team}')
    seasons = r.json()

    # For each season,
    for iter_seasons in seasons:
        try:
            playerstats = []

            # Index of seasons
            idx_season = int(iter_seasons/10000)
            # Pull club stats, only regular seasons for now
            r = requests.get(url=f'https://api-web.nhle.com/v1/club-stats/{iter_team}/{iter_seasons}/{idx_gametype}')
            clubstats = r.json()

            temp_df = pd.json_normalize(clubstats["goalies"])
            temp_df['team_tri'] = iter_team
            temp_df["idx_season"] = idx_season
            temp_df["positionCode"] = "G"
            playerstats.append(temp_df)
            temp_df = pd.json_normalize(clubstats["skaters"])
            temp_df['team_tri'] = iter_team
            temp_df["idx_season"] = idx_season
            playerstats.append(temp_df)

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

            playerstats = playerstats[first_cols+last_cols]

            # Save data
            playerstats.to_csv(f"./data_legacy/player_{iter_seasons}_{iter_team}_{idx_gametype}.csv", index=False)
        except:
            None
        # Pause to play safe with the API
        time.sleep(5)