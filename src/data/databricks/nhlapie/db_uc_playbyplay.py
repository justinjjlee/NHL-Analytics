# Databricks notebook source
# MAGIC %md
# MAGIC # Play-by-play to Databricks
# MAGIC
# MAGIC Most data are pulled and downloaded in daily/weekly cadency. If the data are not available in local repository, then need to separately pull. 
# MAGIC
# MAGIC For the databricks environment, given the infrequent use, I would manually append them using the workflow below.
# MAGIC * When a season is completed, push the file into Databricks volume and run the code below

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract, col

spark = SparkSession.builder.getOrCreate()

path = "dbfs:/Volumes/nhl-databricks/data/play/*_playbyplay.csv"
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    StringType,
    DoubleType,
    DateType,
    TimestampType,
    LongType
)
schema = StructType([
    StructField("gameid", LongType(), True),
    StructField("idx_season", LongType(), True),
    StructField("seasonIdx", LongType(), True),
    StructField("gameDate", DateType(), True),
    StructField("eventId", LongType(), True),
    StructField("timeInPeriod", StringType(), True),
    StructField("timeRemaining", StringType(), True),
    StructField("periodDescriptor.number", LongType(), True),
    StructField("periodDescriptor.periodType", StringType(), True),
    StructField("typeDescKey", StringType(), True),
    StructField("details.reason", StringType(), True),
    StructField("details.eventOwnerTeam", StringType(), True),
    StructField("details.xCoord", DoubleType(), True),
    StructField("details.yCoord", DoubleType(), True),
    StructField("details.zoneCode", StringType(), True),
    StructField("homeTeamDefendingSide", StringType(), True),
    StructField("details.losingPlayerId", DoubleType(), True),
    StructField("details.winningPlayerId", DoubleType(), True),
    StructField("details.shotType", StringType(), True),
    StructField("details.awaySOG", LongType(), True),
    StructField("details.homeSOG", LongType(), True),
    StructField("details.shootingPlayerId", DoubleType(), True),
    StructField("details.blockingPlayerId", DoubleType(), True),
    StructField("details.goalieInNetId", DoubleType(), True),
    StructField("details.scoringPlayerId", DoubleType(), True),
    StructField("details.assist1PlayerId", DoubleType(), True),
    StructField("details.assist2PlayerId", DoubleType(), True),
    StructField("details.hittingPlayerId", DoubleType(), True),
    StructField("details.hitteePlayerId", DoubleType(), True),
    StructField("details.descKey", StringType(), True),
    StructField("details.duration", DoubleType(), True),
    StructField("details.committedByPlayerId", DoubleType(), True),
    StructField("details.drawnByPlayerId", DoubleType(), True)
])

df = spark.read.option("header", "true").schema(schema).csv(path)

for c in df.columns:
    if 'details.' in c:
        df = df.withColumnRenamed(c, c.replace('details.', ''))
    elif 'periodDescriptor.' in c:
        df = df.withColumnRenamed(c, c.replace('periodDescriptor.', 'period_'))
    # timeInPeriod and timeRemaining is in format of (minute):(seconds), change the data format so that it would be minute with decimal of seconds
    elif 'time' in c:
        df = df\
            .withColumn(f"min_part", regexp_extract(col(c), r"^(\d+):(\d+)$", 1).cast("double")) \
            .withColumn("sec_part", regexp_extract(col(c), r"^(\d+):(\d+)$", 2).cast("double")) \
            .withColumn(f"{c}_minutes_decimal", (col("min_part") + col("sec_part") / 60.0)) \
            .drop("min_part", "sec_part")

df.printSchema()

# COMMAND ----------

# Convert necessary column to integer
for c in df.columns:
    if ('PlayerId' in c) or ('goalieInNetId' in c) or ('duration' in c):
        df = df.withColumn(
            c,
            col(c).cast('int')
        )
display(df)

# COMMAND ----------

# Push data into `nhl-databricks`.data.playbyplay
df.write.mode('overwrite').saveAsTable('`nhl-databricks`.data.playbyplay')

# COMMAND ----------

from pyspark.sql.functions import min, max

display(
    df.select(
        min("gameDate").alias("min_gameDate"),
        max("gameDate").alias("max_gameDate")
    )
)
