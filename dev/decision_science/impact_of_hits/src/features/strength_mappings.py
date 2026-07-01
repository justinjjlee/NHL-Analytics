import pyspark.sql.functions as F

def align_priors_to_focal_team(df, focal_team_abbrev: str):
    """
    Aligns Pythagorean and CF/FF process differentials from the
    pregame matrix elo_betting_efficiency to match the focal team perspective.
    """
    # delta_* columns represent Home - Away.
    df = df.withColumn(
        "focal_delta_pythagorean", 
        F.when(F.col("home_team") == focal_team_abbrev, F.col("delta_pythagorean"))
         .otherwise(-F.col("delta_pythagorean"))
    )
    
    df = df.withColumn(
        "focal_delta_CF_pct",
        F.when(F.col("home_team") == focal_team_abbrev, F.col("delta_CF_pct"))
         .otherwise(-F.col("delta_CF_pct"))
    )
    
    df = df.withColumn(
        "focal_delta_FF_pct",
        F.when(F.col("home_team") == focal_team_abbrev, F.col("delta_FF_pct"))
         .otherwise(-F.col("delta_FF_pct"))
    )
    
    # Score state mapping relative to focal team
    df = df.withColumn(
        "focal_score_diff",
        F.when(F.col("home_team") == focal_team_abbrev, F.col("homeTeam_score") - F.col("awayTeam_score"))
         .otherwise(F.col("awayTeam_score") - F.col("homeTeam_score"))
    )
    
    # Restrict to [-2, 2] standard bins
    df = df.withColumn(
        "score_state",
        F.when(F.col("focal_score_diff") >= 2, 2)
         .when(F.col("focal_score_diff") <= -2, -2)
         .otherwise(F.col("focal_score_diff"))
    )
    
    return df

def pandas_align_priors_to_focal_team(df, focal_team_abbrev: str):
    """
    Pandas-based alignment of priors to focal team.
    """
    import numpy as np
    
    # Delta logic
    if 'delta_pythagorean' in df.columns:
        df['focal_delta_pythagorean'] = np.where(
            df['home_team'] == df[focal_team_abbrev],
            df['delta_pythagorean'],
            -df['delta_pythagorean']
        )
    elif 'focal_delta_pythagorean' in df.columns: # local_elo generates this directly
        df['focal_delta_pythagorean'] = np.where(
            df['home_team'] == df[focal_team_abbrev],
            df['focal_delta_pythagorean'],
            -df['focal_delta_pythagorean']
        )
        
    if 'delta_CF_pct' in df.columns:
        df['focal_delta_CF_pct'] = np.where(
            df['home_team'] == df[focal_team_abbrev],
            df['delta_CF_pct'],
            -df['delta_CF_pct']
        )
    elif 'focal_delta_CF_pct' in df.columns:
        df['focal_delta_CF_pct'] = np.where(
            df['home_team'] == df[focal_team_abbrev],
            df['focal_delta_CF_pct'],
            -df['focal_delta_CF_pct']
        )
        
    if 'delta_FF_pct' in df.columns:
        df['focal_delta_FF_pct'] = np.where(
            df['home_team'] == df[focal_team_abbrev],
            df['delta_FF_pct'],
            -df['delta_FF_pct']
        )
        
    df['focal_score_diff'] = np.where(
        df['home_team'] == df[focal_team_abbrev],
        df['homeTeam_score'] - df['awayTeam_score'],
        df['awayTeam_score'] - df['homeTeam_score']
    )
    
    df['score_state'] = np.clip(df['focal_score_diff'], -2, 2)
    
    return df
