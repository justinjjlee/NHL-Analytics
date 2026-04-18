# Databricks notebook source
# MAGIC %md
# MAGIC # Combine Volumes to Unity Catalog Tables
# MAGIC This notebook parses all partitioned CSV files within the Unity Catalog Volumes (`box`, `play`, `team`, `betting`) and merges them into unified Delta schema tables inside `nhl-databricks.data`.
# MAGIC
# MAGIC > This runs standard PySpark to merge files seamlessly and create Delta Tables.

# COMMAND ----------

from pyspark.sql.functions import input_file_name, col, when
import re

# Catalog name without backticks for paths
CATALOG_NAME = "nhl-databricks"
# Catalog name with backticks for SQL identifiers
CATALOG = "`nhl-databricks`"
DB = "data"
BASE_VOL = f"/Volumes/{CATALOG_NAME}/{DB}"

# Use spark.sql to ensure the database schema exists
spark.sql(f"CREATE DATABASE IF NOT EXISTS {CATALOG}.{DB}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Box Volume Tables
# MAGIC Matching distinct file endings exactly to prevent glob cross-contamination.

# COMMAND ----------

# Game List Raw
df_gamelist_raw = spark.read.option("header", "true").csv(f"{BASE_VOL}/box/*_gamelist_raw.csv")
df_gamelist_raw.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{DB}.gamelist_raw")

# Box Team
df_box_team = spark.read.option("header", "true").csv(f"{BASE_VOL}/box/*_box_team.csv")
df_box_team.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{DB}.box_team")

# Box Player
df_box_player = spark.read.option("header", "true").csv(f"{BASE_VOL}/box/*_box_player.csv")
df_box_player.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{DB}.box_player")

# Box Game Stats
df_box_gameStats = spark.read.option("header", "true").csv(f"{BASE_VOL}/box/*_box_gameStats.csv")
df_box_gameStats.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{DB}.box_gameStats")

# Base Box
df_box_base = spark.read.option("header", "true").csv(f"{BASE_VOL}/box/*_box.csv")
df_box_base = df_box_base.filter(col("_metadata.file_path").rlike(r"\d{4}_box\.csv$"))
df_box_base.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{DB}.box")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Team Volume Tables

# COMMAND ----------

# Team Season
df_team_season = spark.read.option("header", "true").csv(f"{BASE_VOL}/team/*_team_season.csv")
df_team_season.write.mode("overwrite").saveAsTable(f"{CATALOG}.{DB}.team_season")

# Player Season
df_player = spark.read.option("header", "true").csv(f"{BASE_VOL}/team/*_player.csv")
# Filter out anything else like playbyplay_player if it accidentally ended up here
df_player = df_player.filter(col("_metadata.file_path").rlike(r"\d{4}_player\.csv$"))
df_player.write.mode("overwrite").saveAsTable(f"{CATALOG}.{DB}.player")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Play Volume Tables

# COMMAND ----------

# Play by Play Shifts
df_play_shift = spark.read.option("header", "true").csv(f"{BASE_VOL}/play/*_playbyplay_shift.csv")
df_play_shift.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{DB}.playbyplay_shift")

# Play by Play Player
df_play_player = spark.read.option("header", "true").csv(f"{BASE_VOL}/play/*_playbyplay_player.csv")
df_play_player.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{DB}.playbyplay_player")

# Play by Play Base
df_playbyplay = spark.read.option("header", "true").csv(f"{BASE_VOL}/play/*_playbyplay.csv")
df_playbyplay = df_playbyplay.filter(col("_metadata.file_path").rlike(r"\d{4}_playbyplay\.csv$"))
df_playbyplay.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{DB}.playbyplay")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Betting / Odds Volume Tables
# MAGIC Odds data requires specific transformations to combine both CA and US locales alongside the 2024 legacy format.

# COMMAND ----------

# Read odds files separately to handle schema differences
# 2024 legacy file uses 'gameId', while 2025 files use 'gameid'

# Helper function to normalize column names and add country
def read_and_tag_odds(path_pattern, country_code):
    df = spark.read.option("header", "true").csv(path_pattern)
    # Normalize all column names to lowercase for consistency
    for col_name in df.columns:
        df = df.withColumnRenamed(col_name, col_name.lower())
    df = df.withColumn("country", lit(country_code))
    return df

from pyspark.sql.functions import lit

# Read each file pattern
df_2024 = read_and_tag_odds(f"{BASE_VOL}/betting/2024_odds.csv", "Legacy_Merged")
df_us = read_and_tag_odds(f"{BASE_VOL}/betting/*_odds_US.csv", "US")
df_ca = read_and_tag_odds(f"{BASE_VOL}/betting/*_odds_CA.csv", "CA")

# Union all dataframes (unionByName handles different column orders and missing columns)
df_odds = df_2024.unionByName(df_us, allowMissingColumns=True) \
                 .unionByName(df_ca, allowMissingColumns=True)

df_odds.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{DB}.odds")

print("Successfully merged all Databricks Volumes into Unity Catalog Delta Tables.")
