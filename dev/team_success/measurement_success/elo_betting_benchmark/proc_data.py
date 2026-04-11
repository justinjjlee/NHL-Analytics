# Databricks notebook source
# MAGIC %md
# MAGIC # Data Processing
# MAGIC Data processing

# COMMAND ----------

import pandas as pd
import numpy as np

import src.elo_betting

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM `nhl-databricks`.data.odds

# COMMAND ----------

# Load odds data from SQL into a Spark DataFrame
odds_df = spark.sql("""
SELECT 
    gameid,
    tricode_for AS home_team,
    tricode_against AS away_team,
    home_odds_value AS odds_home,
    away_odds_value AS odds_away,
    -- Create row number for each team's home and away game 
    row_number() OVER (PARTITION BY tricode_for ORDER BY time_start ASC) AS order_homes,
    row_number() OVER (PARTITION BY tricode_against ORDER BY time_start ASC) AS order_aways
FROM `nhl-databricks`.data.odds
WHERE 
    -- Only include 24-25 season
    time_start >= '2024-10-01'
    -- ONly include regular season
    AND substr(CAST(gameid AS STRING), 7, 1) = 2
""")

# Collect unique teams and games using DataFrame methods
teams = (
    odds_df.select('home_team')
    .union(odds_df.select('away_team'))
    .distinct()
    .orderBy('home_team')
    .toPandas()['home_team']
    .tolist()
)
game_count = odds_df.agg({"order_homes": "max", "order_aways": "max"}).collect()[0]
game_count = max(game_count[0] if game_count[0] is not None else 0, game_count[1] if game_count[1] is not None else 0)

import numpy as np

# Initialize 3D array: teams x home/away x games
odds_3d = np.full((len(teams), 2, game_count), np.nan)
team_idx = {team: i for i, team in enumerate(teams)}

# Collect data to driver for array population
odds_pd = odds_df.toPandas()

for _, row in odds_pd.iterrows():
    # Team label
    h = team_idx[row['home_team']]
    a = team_idx[row['away_team']]
    # Game orders
    oh = row['order_homes']-1
    oa = row['order_aways']-1
    # Save odds data
    odds_3d[h, 0, oh] = row['odds_home']
    odds_3d[a, 1, oa] = row['odds_away']

# COMMAND ----------

# MAGIC %md
# MAGIC # Odds to probability
# MAGIC First, need to convert the American odds (money line) to decimal form.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC   -- Convert (American) odds to decimal
# MAGIC   , CASE
# MAGIC     WHEN home_odds_value > 0 THEN home_odds_value/100 + 1 -- the profit on a $100 wager. 
# MAGIC     WHEN home_odds_value < 0 THEN  100/ABS(home_odds_value) + 1  -- the amount you must wager to profit $100. 
# MAGIC     ELSE NULL -- No betting information
# MAGIC   END AS home_odds_decimal
# MAGIC   , CASE
# MAGIC     WHEN away_odds_value > 0 THEN away_odds_value/100 + 1 -- the profit on a $100 wager. 
# MAGIC     WHEN away_odds_value < 0 THEN  100/ABS(away_odds_value) + 1  -- the amount you must wager to profit $100. 
# MAGIC     ELSE NULL
# MAGIC   END AS away_odds_decimal
# MAGIC FROM `nhl-databricks`.data.odds

# COMMAND ----------

# MAGIC %md
# MAGIC Use already existing 

# COMMAND ----------

!pip install shin

# COMMAND ----------

# Import repository
import shin

# Test to see if it works
shin.calculate_implied_probabilities([1.5, 2.74], full_output=True)

