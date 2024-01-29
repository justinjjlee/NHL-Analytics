
# Load all data files
data_files = glob.glob(cwd_data_college+'/*.xlsx')
# Pull data and stack, with consistent column names
df = pd.read_excel(data_files[0])
colnames = [
    "name_last", "name_first", 
    "college", "nhl", 
    "position", "nationality", "draft",
    "college_yrs", "season"
]
df.columns = colnames
for iter in data_files[1:]:
    tempdf = pd.read_excel(iter)
    tempdf.columns = colnames
    df = pd.concat([df, tempdf], ignore_index=True)

# Full name
df = df.assign(fullName = df.name_first + " " + df.name_last)
# For player went through multiple nhl team
df['team_triCode'] = df.nhl.str.split('/')
# Split and explode
df = df.explode('team_triCode')
# Remove extra space
df['team_triCode'] = df['team_triCode'].str.strip()

# Join with player statistics
#   This API data pull based on my ohter data collection and cleaning process
#   See repository data cleaning .py
cwd_data_player = cwd_data + "nhl/data_gamelvl/cleaned"

# Load all data files
#   Start with regular seasons only
#   NOTE: the season year notation of the NHL data uses start year,
#       unlike the college data, which uses the end year of the season
data_files_agg_02 = glob.glob(cwd_data_player+'/*02_player.csv')
# Consistent column name
df_player = pd.read_csv(data_files_agg_02[0])
# Get the year stat: set year-ending
yr = int(data_files_agg_02[0]\
        .removeprefix(cwd_data_player+"/")\
        .removesuffix("_02_player.csv")
    ) + 1
df_player = df_player.assign(season = yr)
for iter in data_files_agg_02[1:]:
    tempdf = pd.read_csv(iter)
    # Get the year stat: set year-ending
    yr = int(iter\
        .removeprefix(cwd_data_player+"/")\
        .removesuffix("_02_player.csv")) + 1
    tempdf = tempdf.assign(season = yr)
    df_player = pd.concat([df_player, tempdf], ignore_index=True)

# Time-on-ice minute convert to decimal minute
df_player['timeOnIce'] = df_player['timeOnIce'].fillna("00:00")
df_player[['temp_min', 'temp_sec']] = df_player['timeOnIce'].str.split(':', expand=True)
df_player['ToI'] = \
    df_player['temp_min'].astype(int) + \
    df_player['temp_sec'].astype(int)/60
df_player.drop(['temp_min', 'temp_sec'], axis=1, inplace=True)

# ...............................................
# Joining the college records and full NHL player data
#   the successfull match would lead to complete count of college records

# First initally aggregate the player information to be matched
df_agg = df_player\
    .groupby(['season', 'id_player', 'fullName', 'team_triCode'])\
    .agg(
        position = ('player_pos_abv','min'),
        rookie = ('rookie', 'min')
    )

# First pass join: perfect record match
joindf = df[
        ["season", "fullName", "team_triCode",
         "nationality", "draft", "college","college_yrs"
        ]
    ]\
    .set_index(["season","fullName", "team_triCode"])\
    .join(
        df_agg.reset_index().set_index(["season", "fullName", "team_triCode"]),
        lsuffix="_agg"
    )

# Second pass join: using of the last name
joindf_2 = joindf.loc[joindf.id_player.isna()]\
    .drop(['id_player', 'position', 'rookie'], axis=1)\
    .reset_index()
# Drop non-matches from the previous
joindf.dropna(subset="id_player", inplace=True)

# ------------------ Second phase: fuzzy
from fuzzywuzzy import fuzz, process

# Function to grade similarity 
def calc_apx(iter):
    return fuzz.token_sort_ratio(iter['fullName_x'], iter['fullName_y'])

# Cross-join for all possibility
#   At least the season number and team name must match
cross_join = pd.merge(
    joindf_2.assign(key=1), df_agg.reset_index().assign(key=1), 
    on=['key', 'season','team_triCode']
    ).drop('key', axis=1)

# Grade similarity
cross_join['ind_appx'] = cross_join.apply(calc_apx, axis=1)

#17,343,223 records took 100m 54.4s: 
#   with all out matching with name with no ['season', 'teamtriCode]

# Some visual and point validations were necessary on this one. 
#   78 is the benchmark, 
joindf_2_fin = \
    cross_join.loc[cross_join.ind_appx>78].sort_values("ind_appx",ascending=False)

# Keep NHL name for consistency
joindf_2_fin = \
    joindf_2_fin[
        ['season', 'fullName_y', 'team_triCode', 
         'nationality', 'draft', 'college',
       'college_yrs', 'id_player', 'position','rookie']]
# Column renames
joindf_2_fin.columns = joindf.reset_index().columns

# ...............................................
# Combine completed data
dfin = pd.concat([joindf.reset_index(), joindf_2_fin])
# finalize with reset index
dfin.sort_values(['id_player','season'], inplace=True)
dfin.set_index(['id_player', 'fullName', 'season', 'team_triCode'], inplace=True)

# College name fixes
dfin.loc[dfin.college == 'Alabama-Huntsville', 'college'] = 'Alabama Huntsville'
dfin.loc[dfin.college == 'Alaska', 'college'] = 'Alaska Fairbanks'
dfin.loc[dfin.college == 'UAA', 'college'] = 'Alaska Anchorage'
dfin.loc[dfin.college == 'UAF', 'college'] = 'Alaska Fairbanks'
dfin.loc[dfin.college == 'WMU', 'college'] = 'Western Michigan'
dfin.loc[dfin.college == 'BC', 'college'] = 'Boston College'
dfin.loc[dfin.college == 'BU', 'college'] = 'Boston University'
dfin.loc[dfin.college == 'BGSU', 'college'] = 'Bowling Green'
dfin.loc[dfin.college == 'BSU', 'college'] = 'Bemidji State'
dfin.loc[dfin.college == 'Minn. St.', 'college'] = 'Minnesota State'
dfin.loc[dfin.college == 'SCSU', 'college'] = 'St. Cloud State'
dfin.loc[dfin.college == 'N. Dame', 'college'] = 'Notre Dame'
dfin.loc[dfin.college == 'N. Dakota', 'college'] = 'North Dakota'
dfin.loc[dfin.college == 'CC', 'college'] = 'Colorado College'
dfin.loc[dfin.college == 'NU', 'college'] = 'Northeastern'
dfin.loc[dfin.college == 'Minn.', 'college'] = 'Minnesota'
dfin.loc[dfin.college == 'MTU', 'college'] = 'Michigan Tech'
dfin.loc[dfin.college == 'MSU', 'college'] = 'Michigan State'
dfin.loc[dfin.college == 'OSU', 'college'] = 'Ohio State'
dfin.loc[dfin.college == 'RPI', 'college'] = 'Rensselaer'
dfin.loc[dfin.college == 'UNO', 'college'] = 'Omaha'
dfin.loc[dfin.college == 'UNH', 'college'] = 'New Hampshire'
dfin.loc[dfin.college == 'UML', 'college'] = 'UMass Lowell'
dfin.loc[dfin.college == 'UMD', 'college'] = 'Minnesota Duluth'

# ...............................................
# Pull advanced statstics on player performance
# For this analysis, I observe by season-level stats
#   Player can move within season
df_agg = df_player\
    .groupby(['id_player', 'fullName', 'season', 'team_triCode'])\
    .agg(
        games = ('id_player', 'count'),
        # Game-level play statis
        gamemin = ('ToI', "median"),
        # Overall performance
        plusminus = ('plusMinus', 'sum'),
        goals = ('goals', 'sum'),
        goalsPowerPlay = ('powerPlayGoals', 'sum'),
        goalsShortHand = ('shortHandedGoals', 'sum'),
        assists = ('assists', 'sum'),
        assistsPowerPlay = ('powerPlayAssists', 'sum'),
        assistsShortHand = ('shortHandedAssists', 'sum'),
        shots = ('shots', 'sum'),
        blocked = ('blocked', 'sum'),
        hits = ('hits', 'sum'),
        faceoff = ('faceoffTaken', 'sum'),
        faceoffWins = ('faceOffWins', 'sum'),
        takeaways = ('takeaways', 'sum'),
        giveaways = ('giveaways', 'sum'),
    )

# Finalized player-level stats
dfin = dfin.join(df_agg, how='left')