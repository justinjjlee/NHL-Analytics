import pyspark.sql.functions as F
from pyspark.sql.window import Window

def load_odds(spark, catalog="`nhl-databricks`.data"):
    """
    Load odds data for 2024 and 2025 seasons, filtering out Canadian markets.
    Converts American odds to decimal, and computes naive probabilities.
    """
    df = spark.sql(f"""
        SELECT 
            CAST(odds.gameid AS INT) as gameid,
            odds.`hometeam.abbrev` AS home_team,
            odds.`awayteam.abbrev` AS away_team,
            odds.home_odds_value,
            odds.away_odds_value,
            odds.odds_description,
            odds.country,
            odds.starttimeutc,
            -- Convert American Odds to Decimal
            CASE WHEN CAST(odds.home_odds_value AS DOUBLE) > 0 THEN (CAST(odds.home_odds_value AS DOUBLE)/100) + 1
                 WHEN CAST(odds.home_odds_value AS DOUBLE) < 0 THEN (100/ABS(CAST(odds.home_odds_value AS DOUBLE))) + 1
                 ELSE NULL END as odds_decimal_home,
            CASE WHEN CAST(odds.away_odds_value AS DOUBLE) > 0 THEN (CAST(odds.away_odds_value AS DOUBLE)/100) + 1
                 WHEN CAST(odds.away_odds_value AS DOUBLE) < 0 THEN (100/ABS(CAST(odds.away_odds_value AS DOUBLE))) + 1
                 ELSE NULL END as odds_decimal_away,
            -- Identify season (Approximation based on start time UTC)
            CASE WHEN odds.starttimeutc >= '2023-08-01' AND odds.starttimeutc < '2024-08-01' THEN 2024
                 WHEN odds.starttimeutc >= '2024-08-01' AND odds.starttimeutc < '2025-08-01' THEN 2025
                 ELSE NULL END as season 
        FROM {catalog}.odds
        WHERE odds.country != 'CA' 
          AND (odds.starttimeutc >= '2023-08-01')
    """)
    return df

def load_game_outcomes(spark, catalog="`nhl-databricks`.data"):
    """
    Load game outcome mapping (Wins/Losses, OT flags).
    """
    return spark.sql(f"""
        SELECT 
            CAST(gameid AS INT) as gameid,
            date,
            tricode_for AS home_team,
            tricode_against AS away_team,
            CAST(metric_score_for AS DOUBLE) AS home_goals,
            CAST(metric_score_against AS DOUBLE) AS away_goals,
            period_ending,
            CASE WHEN CAST(metric_score_for AS DOUBLE) > CAST(metric_score_against AS DOUBLE) THEN 1 ELSE 0 END AS home_win,
            CASE WHEN CAST(metric_score_for AS DOUBLE) < CAST(metric_score_against AS DOUBLE) THEN 1 ELSE 0 END AS away_win,
            -- Extract Season from gameid: gameid '2023020001' is season 2023-2024. 
            CAST(SUBSTRING(CAST(gameid AS STRING), 1, 4) AS INT) + 1 AS season,
            CAST(SUBSTRING(CAST(gameid AS STRING), 5, 2) AS INT) AS game_type -- 02 is regular season
        FROM {catalog}.box
        -- Regular season only
        WHERE CAST(SUBSTRING(CAST(gameid AS STRING), 5, 2) AS INT) = 2
          AND CAST(SUBSTRING(CAST(gameid AS STRING), 1, 4) AS INT) IN (2023, 2024)
    """)

def build_team_panel(spark, catalog="`nhl-databricks`.data"):
    """
    Load box metrics (Corsi, Fenwick, Goals) from box_gamestats table,
    and produce rolling accumulators up to game n-1.
    """
    df_bt = spark.sql(f"""
        SELECT 
            CAST(gameid_for AS INT) as gameid,
            team_tri_for AS team,
            team_tri_against AS opponent,
            CAST(corsi_for AS DOUBLE) as corsi_for,
            CAST(corsi_against AS DOUBLE) as corsi_against,
            CAST(fenwick_for AS DOUBLE) as fenwick_for,
            CAST(fenwick_against AS DOUBLE) as fenwick_against,
            CAST(goals_for AS DOUBLE) as goals_for,
            CAST(goals_against AS DOUBLE) as goals_against,
            -- Same logic for season extraction
            CAST(SUBSTRING(CAST(gameid_for AS STRING), 1, 4) AS INT) + 1 AS season
        FROM {catalog}.box_gamestats
        WHERE CAST(SUBSTRING(CAST(gameid_for AS STRING), 5, 2) AS INT) = 2
          AND CAST(SUBSTRING(CAST(gameid_for AS STRING), 1, 4) AS INT) IN (2023, 2024)
    """)

    # Window partitions: per team, per season, order by gameid (ascending assumes chronological order of gameid)
    w1 = Window.partitionBy("team", "season").orderBy("gameid")
    # Window for prior games (1 to n-1)
    w_prior = w1.rowsBetween(Window.unboundedPreceding, -1)
    
    # Calculate rolling sums excluding current game to prevent lookahead bias
    df_panel = df_bt.withColumn("n_g", F.row_number().over(w1)) \
        .withColumn("cum_CF", F.sum("corsi_for").over(w_prior)) \
        .withColumn("cum_CA", F.sum("corsi_against").over(w_prior)) \
        .withColumn("cum_FF", F.sum("fenwick_for").over(w_prior)) \
        .withColumn("cum_FA", F.sum("fenwick_against").over(w_prior)) \
        .withColumn("cum_GF", F.sum("goals_for").over(w_prior)) \
        .withColumn("cum_GA", F.sum("goals_against").over(w_prior))
        
    # Fill nulls with 0 for the first game of each season
    df_panel = df_panel.fillna(0, subset=["cum_CF", "cum_CA", "cum_FF", "cum_FA", "cum_GF", "cum_GA"])
    
    # Calculate proportions representing our process signals
    df_panel = df_panel.withColumn("rolling_CF_pct", F.when((F.col("cum_CF") + F.col("cum_CA")) > 0, F.col("cum_CF") / (F.col("cum_CF") + F.col("cum_CA"))).otherwise(0.5)) \
        .withColumn("rolling_FF_pct", F.when((F.col("cum_FF") + F.col("cum_FA")) > 0, F.col("cum_FF") / (F.col("cum_FF") + F.col("cum_FA"))).otherwise(0.5)) \
        .withColumn("rolling_pythagorean", F.when((pow(F.col("cum_GF"), 2.0) + pow(F.col("cum_GA"), 2.0)) > 0, 
                                            pow(F.col("cum_GF"), 2.0) / (pow(F.col("cum_GF"), 2.0) + pow(F.col("cum_GA"), 2.0)))
                                            .otherwise(0.5))
                                            
    return df_panel
