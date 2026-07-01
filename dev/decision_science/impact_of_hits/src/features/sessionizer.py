import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType

def convert_time_in_period(df: pd.DataFrame, time_in_period_col: str = "timeInPeriod", period_col: str = "period", elapsed_col: str = "elapsed_sec") -> pd.DataFrame:
    """
    Pandas-based utility for elapsed seconds.
    Converts MM:SS format and period number to absolute elapsed seconds in a game.
    Assumes each period is 20 minutes (1200 seconds).
    """
    df[['mm', 'ss']] = df[time_in_period_col].str.split(":", expand=True).astype(int)
    period_offset = (df[period_col] - 1) * 1200
    df[elapsed_col] = period_offset + (df['mm'] * 60) + df['ss']
    return df.drop(columns=['mm', 'ss'])

def spark_convert_time_in_period(df, time_in_period_col: str = "timeInPeriod", period_col: str = "periodDescriptor_number", elapsed_col: str = "elapsed_sec"):
    """
    PySpark wrapper for converting MM:SS and period into total elapsed seconds.
    """
    df = df.withColumn('mm', F.split(F.col(time_in_period_col), ":")[0].cast(IntegerType())) \
           .withColumn('ss', F.split(F.col(time_in_period_col), ":")[1].cast(IntegerType()))
    df = df.withColumn('period_offset', (F.col(period_col) - 1) * 1200)
    df = df.withColumn(elapsed_col, F.col('period_offset') + (F.col('mm') * 60) + F.col('ss'))
    return df.drop('mm', 'ss', 'period_offset')
