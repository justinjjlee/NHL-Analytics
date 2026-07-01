import pandas as pd
import numpy as np

def compute_local_elo(spark, box_team_csv_path, as_pandas=False):
    """
    Replicates the team success workflow to compute Pythagorean Expectation
    and CF% proxy from local box score data, returning a Spark DataFrame
    compatible with the Unity Catalog elo_betting_efficiency table.
    """
    df = pd.read_csv(box_team_csv_path)
    
    # Sort by game date/id
    df = df.sort_values(by=['gameDate', 'gameid'])
    
    # Calculate rolling goals and shots for Pythagorean and Corsi proxies
    df['rgoals'] = df.groupby('team_tri')['goals'].transform(lambda x: x.shift().cumsum().fillna(0))
    df['rshots'] = df.groupby('team_tri')['shots'].transform(lambda x: x.shift().cumsum().fillna(0))
    
    # Self join to get opponent stats
    df_oppo = df[['gameid', 'team_tri', 'rgoals', 'rshots']].copy()
    df_oppo.columns = ['gameid', 'team_tri_oppo', 'rgoals_oppo', 'rshots_oppo']
    
    # Merge on gameid where team_tri != team_tri_oppo
    merged = pd.merge(df, df_oppo, on='gameid')
    merged = merged[merged['team_tri'] != merged['team_tri_oppo']].copy()
    
    def calc_pe(x, y):
        den = (x**2 + y**2)
        den = np.where(den == 0, 1, den)
        return (x**2) / den
        
    merged['pyexp'] = calc_pe(merged['rgoals'], merged['rgoals_oppo'])
    
    # Calculate Corsi proxy
    merged['corsi_pct'] = merged['rshots'] / np.where((merged['rshots'] + merged['rshots_oppo']) == 0, 1, merged['rshots'] + merged['rshots_oppo'])
    
    # Filter to home team to form the baseline table
    home_df = merged[merged['teamloc'] == 'home'].copy()
    
    # delta = home_val - away_val
    # away pyexp is (1 - home_pyexp)
    home_df['focal_delta_pythagorean'] = home_df['pyexp'] - (1.0 - home_df['pyexp'])
    home_df['focal_delta_CF_pct'] = home_df['corsi_pct'] - (1.0 - home_df['corsi_pct'])
    
    final_df = home_df[['gameid', 'team_tri', 'focal_delta_pythagorean', 'focal_delta_CF_pct']].copy()
    final_df.columns = ['gameid', 'home_team', 'focal_delta_pythagorean', 'focal_delta_CF_pct']
    
    # Fill NAs with 0
    final_df = final_df.fillna(0.0)
    
    if as_pandas:
        return final_df
    return spark.createDataFrame(final_df)
