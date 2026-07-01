import pyspark.sql.functions as F
from pyspark.sql.window import Window

def assign_event_flags(df, focal_team_col="focal_team_abbrev"):
    """
    Assign binary flags for required event types given a focal team.
    """
    base_conditions = {
        "hit_for": (F.col("typeDescKey") == "hit") & (F.col("details_eventOwnerTeamId") == F.col(focal_team_col)),
        "hit_against": (F.col("typeDescKey") == "hit") & (F.col("details_eventOwnerTeamId") != F.col(focal_team_col)),
        "shot_for": (F.col("typeDescKey") == "shot-on-goal") & (F.col("details_eventOwnerTeamId") == F.col(focal_team_col)),
        "shot_against": (F.col("typeDescKey") == "shot-on-goal") & (F.col("details_eventOwnerTeamId") != F.col(focal_team_col)),
        "takeaway": (F.col("typeDescKey") == "takeaway") & (F.col("details_eventOwnerTeamId") == F.col(focal_team_col)),
        "giveaway": (F.col("typeDescKey") == "giveaway") & (F.col("details_eventOwnerTeamId") == F.col(focal_team_col)),
        "block_for": (F.col("typeDescKey") == "blocked-shot") & (F.col("details_eventOwnerTeamId") == F.col(focal_team_col)),
        "block_against": (F.col("typeDescKey") == "blocked-shot") & (F.col("details_eventOwnerTeamId") != F.col(focal_team_col)),
        "faceoff_win": (F.col("typeDescKey") == "faceoff") & (F.col("details_eventOwnerTeamId") == F.col(focal_team_col))
    }
    
    for col_name, condition in base_conditions.items():
        df = df.withColumn(col_name, F.when(condition, 1.0).otherwise(0.0))
    return df

def identify_zone_entries(df, focal_team_col="focal_team_abbrev", elapsed_col="elapsed_sec"):
    w = Window.partitionBy("game_id").orderBy(elapsed_col)
    
    df = df.withColumn("prev_zoneCode", F.lag("details_zoneCode").over(w))
    df = df.withColumn("time_gap", F.col(elapsed_col) - F.lag(elapsed_col).over(w))
    
    entry_condition = (
        (F.col("details_zoneCode") == "O") & 
        (F.col("prev_zoneCode").isin("N", "D")) & 
        (F.col("time_gap") <= 10) & 
        (F.col("details_eventOwnerTeamId") == F.col(focal_team_col))
    )
    
    df = df.withColumn("zone_entry", F.when(entry_condition, 1.0).otherwise(0.0))
    return df.drop("prev_zoneCode", "time_gap")

def compute_possession_shift_index(df, focal_team_col="focal_team_abbrev", window_seconds=90):
    """
    Computes rolling 90s possession shift frequency as a proxy for zone flow.
    """
    w_event = Window.partitionBy("game_id").orderBy("elapsed_sec")
    
    df = df.withColumn("prev_event_owner", F.lag("details_eventOwnerTeamId").over(w_event))
    df = df.withColumn(
        "is_possession_shift",
        F.when(
            (F.col("details_eventOwnerTeamId") != F.col("prev_event_owner")) &
            (F.col("details_eventOwnerTeamId") == F.col(focal_team_col)),
            1.0
        ).otherwise(0.0)
    )
    
    w_90s = Window.partitionBy("game_id").orderBy("elapsed_sec").rangeBetween(-window_seconds, 0)
    df = df.withColumn("rolling_possession_shifts", F.sum("is_possession_shift").over(w_90s))
    
    total_events_90s = F.count("*").over(w_90s)
    focal_events_90s = F.sum(F.when(F.col("details_eventOwnerTeamId") == F.col(focal_team_col), 1.0).otherwise(0.0)).over(w_90s)
    
    df = df.withColumn("rolling_zone_share_90s", F.when(total_events_90s > 0, focal_events_90s / total_events_90s).otherwise(0.5))
    
    return df

def pandas_assign_event_flags(df, focal_team_col="focal_team_abbrev"):
    import numpy as np
    
    df['hit_for_O'] = np.where((df['typeDescKey'] == 'hit') & (df['details_eventOwnerTeamId'] == df[focal_team_col]) & (df['details_zoneCode'] == 'O'), 1.0, 0.0)
    df['hit_for_D'] = np.where((df['typeDescKey'] == 'hit') & (df['details_eventOwnerTeamId'] == df[focal_team_col]) & (df['details_zoneCode'] == 'D'), 1.0, 0.0)
    df['hit_for_N'] = np.where((df['typeDescKey'] == 'hit') & (df['details_eventOwnerTeamId'] == df[focal_team_col]) & (df['details_zoneCode'] == 'N'), 1.0, 0.0)
    
    df['hit_against_O'] = np.where((df['typeDescKey'] == 'hit') & (df['details_eventOwnerTeamId'] != df[focal_team_col]) & (df['details_zoneCode'] == 'O'), 1.0, 0.0)
    df['hit_against_D'] = np.where((df['typeDescKey'] == 'hit') & (df['details_eventOwnerTeamId'] != df[focal_team_col]) & (df['details_zoneCode'] == 'D'), 1.0, 0.0)
    df['hit_against_N'] = np.where((df['typeDescKey'] == 'hit') & (df['details_eventOwnerTeamId'] != df[focal_team_col]) & (df['details_zoneCode'] == 'N'), 1.0, 0.0)
    
    df['shot_for'] = np.where((df['typeDescKey'] == 'shot-on-goal') & (df['details_eventOwnerTeamId'] == df[focal_team_col]), 1.0, 0.0)
    df['shot_against'] = np.where((df['typeDescKey'] == 'shot-on-goal') & (df['details_eventOwnerTeamId'] != df[focal_team_col]), 1.0, 0.0)
    
    df['takeaway'] = np.where((df['typeDescKey'] == 'takeaway') & (df['details_eventOwnerTeamId'] == df[focal_team_col]), 1.0, 0.0)
    df['giveaway'] = np.where((df['typeDescKey'] == 'giveaway') & (df['details_eventOwnerTeamId'] == df[focal_team_col]), 1.0, 0.0)
    
    df['block_for'] = np.where((df['typeDescKey'] == 'blocked-shot') & (df['details_eventOwnerTeamId'] == df[focal_team_col]), 1.0, 0.0)
    df['block_against'] = np.where((df['typeDescKey'] == 'blocked-shot') & (df['details_eventOwnerTeamId'] != df[focal_team_col]), 1.0, 0.0)
    
    df['faceoff_win'] = np.where((df['typeDescKey'] == 'faceoff') & (df['details_eventOwnerTeamId'] == df[focal_team_col]), 1.0, 0.0)
    
    return df

def pandas_identify_zone_entries(df, focal_team_col="focal_team_abbrev", elapsed_col="elapsed_sec"):
    import numpy as np
    
    df = df.sort_values(by=['game_id', elapsed_col])
    df['prev_zoneCode'] = df.groupby('game_id')['details_zoneCode'].shift(1)
    df['time_gap'] = df[elapsed_col] - df.groupby('game_id')[elapsed_col].shift(1)
    
    entry_condition = (
        (df['details_zoneCode'] == 'O') &
        (df['prev_zoneCode'].isin(['N', 'D'])) &
        (df['time_gap'] <= 10) &
        (df['details_eventOwnerTeamId'] == df[focal_team_col])
    )
    
    df['zone_entry'] = np.where(entry_condition, 1.0, 0.0)
    return df.drop(columns=['prev_zoneCode', 'time_gap'])

def pandas_compute_possession_shift_index(df, focal_team_col="focal_team_abbrev", window_seconds=90):
    import numpy as np
    import pandas as pd
    
    df = df.sort_values(by=['game_id', 'elapsed_sec'])
    
    df['prev_event_owner'] = df.groupby('game_id')['details_eventOwnerTeamId'].shift(1)
    
    df['is_possession_shift'] = np.where(
        (df['details_eventOwnerTeamId'] != df['prev_event_owner']) &
        (df['details_eventOwnerTeamId'] == df[focal_team_col]),
        1.0, 0.0
    )
    
    def calc_zone_share(group):
        # We need a time index for rolling
        group = group.set_index('elapsed_sec_dt')
        
        # total events 90s
        total_events = group['is_event'].rolling(f'{window_seconds}s').sum()
        focal_events = group['is_focal_event'].rolling(f'{window_seconds}s').sum()
        
        group['rolling_zone_share_90s'] = np.where(total_events > 0, focal_events / total_events, 0.5)
        
        # also rolling possession shifts
        group['rolling_possession_shifts'] = group['is_possession_shift'].rolling(f'{window_seconds}s').sum()
        
        return group.reset_index(drop=True)
    
    # We create a dummy dt column just for rolling
    df['elapsed_sec_dt'] = pd.to_datetime(df['elapsed_sec'], unit='s')
    df['is_event'] = 1.0
    df['is_focal_event'] = np.where(df['details_eventOwnerTeamId'] == df[focal_team_col], 1.0, 0.0)
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        df = df.groupby('game_id', group_keys=False).apply(calc_zone_share).reset_index(drop=True)
    
    return df.drop(columns=['is_event', 'is_focal_event', 'prev_event_owner', 'is_possession_shift'])
