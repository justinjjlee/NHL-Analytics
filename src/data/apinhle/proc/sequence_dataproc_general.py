import duckdb
import os

class nhl_sequence:
    '''
        Objects to process play-by-play data in sequence
    '''
    def __init__(self, db_nhldata):
        # Path to the folder containing the CSV files
        self.db_nhldata = db_nhldata 

    def database_generate(self):
        '''
            Generate basic database

            Output: self.conn - connection of table sequence
        '''
        # Initialize a DuckDB connection
        self.conn = duckdb.connect()

        # .......................
        # Pull play-by-play data
        folder_path = self.db_nhldata + "/playbyplay"

        # List all CSV files in the folder
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

        # Iterate over each CSV file and read it into DuckDB using out-of-core processing
        for csv_file in csv_files:
            file_path = os.path.join(folder_path, csv_file)
            table_name = os.path.splitext(csv_file)[0]  # Use the file name (without extension) as the table name
            self.conn.execute(f"""
                CREATE TABLE "{table_name}" AS 
                SELECT
                    *,
                    -- Create columns of specific condition
                    -- //
                    --  Goal-scoring team
                    CASE
                        WHEN typeDescKey = 'goal' THEN "details.eventOwnerTeam"
                        ELSE NULL
                    END AS eventOnwerTeam_goalscore
                FROM read_csv('{file_path}', AUTO_DETECT=TRUE)
            """)

        # Combine all tables into one table named 'combined_table'
        union_query = " UNION ALL ".join([f'SELECT * FROM "{os.path.splitext(csv_file)[0]}"' for csv_file in csv_files])
        self.conn.execute(f"CREATE TABLE pbp AS {union_query}")

        # Verify by listing all tables
        tables = self.conn.execute("SHOW TABLES").fetchall()
        print("Tables loaded into DuckDB:", tables)

        # ...........................................
        # Pull play-by-play corresponding player data

        # Path to the folder containing the CSV files
        folder_path = self.db_nhldata + "/playbyplay_player"

        # List all CSV files in the folder
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

        # Iterate over each CSV file and read it into DuckDB using out-of-core processing
        for csv_file in csv_files:
            file_path = os.path.join(folder_path, csv_file)
            table_name = os.path.splitext(csv_file)[0]  # Use the file name (without extension) as the table name
            self.conn.execute(f"""
                CREATE TABLE "{table_name}" AS 
                SELECT
                    *
                FROM read_csv('{file_path}', AUTO_DETECT=TRUE)
            """)

        # Combine all tables into one table named 'combined_table'
        union_query = " UNION ALL ".join([f'SELECT * FROM "{os.path.splitext(csv_file)[0]}"' for csv_file in csv_files])
        self.conn.execute(f"CREATE TABLE pbp_p AS {union_query}")

        # Verify by listing all tables
        tables = self.conn.execute("SHOW TABLES").fetchall()
        print("Tables loaded into DuckDB:", tables)

        # Return - nothing. saved as object

    def database_DataEngineering(self):
        '''
            Add features to the data

            Outcome: creating DuckDB table named 'dfseq' in database
        '''
        self.conn.execute(
            """
                DROP TABLE IF EXISTS dfseq;
                CREATE TABLE dfseq AS 
                WITH df AS 
                (
                    SELECT
                        pbp.* , 
                        -- Additional information about
                        -- Create the time in period to be continuous to be computing to be easier
                        ("periodDescriptor.number" - 1) * 20 
                            + CAST(LEFT(timeInPeriod, 2) AS INT)
                            + CAST(RIGHT(timeInPeriod, 2) AS INT) * (1/60)
                        AS timeInPeriod_continue,
                        -- Goal standing and sequence
                        -- \\
                        -- Last goal scoring team
                        --  If not current scoring team, figure out who was the last scoring team
                        --      Commented out such that the current method for the record of goal scoring at the time
                        --      is paired with lagged column of previous goal scoring team for future sequence analyses
                        --          (The first goal of the game would be marked NULL)
                        --COALESCE(
                        --    eventOnwerTeam_goalscore, 
                            LAST_VALUE(eventOnwerTeam_goalscore IGNORE NULLS) OVER (
                                PARTITION BY pbp.gameid ORDER BY "periodDescriptor.number" ASC, timeInPeriod ASC
                                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING--CURRENT ROW
                        --    ) 
                        ) AS eventOnwerTeam_goalscore_last,
                        -- Matching information of players
                        -- // Shooter and blocker
                        player_shooter.team_tri AS shooter_team, 
                        player_blocker.team_tri AS blocker_team,
                        player_blocker.playerId AS blocker_player,
                        player_shooter.playerId AS shooter_player, 
                        -- // Penalty
                        player_penalty_commited.team_tri AS penalty_commited_team,
                        player_penalty_drawing.team_tri AS penalty_drawing_team
                    FROM pbp AS pbp
                    LEFT JOIN pbp_p AS player_shooter
                    ON 
                        pbp.gameid = player_shooter.gameid AND 
                        pbp."details.shootingPlayerId" = player_shooter.playerId
                    LEFT JOIN pbp_p AS player_blocker
                    ON 
                        pbp.gameid = player_blocker.gameid AND 
                        pbp."details.blockingPlayerId" = player_blocker.playerId
                    LEFT JOIN pbp_p AS player_penalty_commited
                    ON 
                        pbp.gameid = player_penalty_commited.gameid AND 
                        pbp."details.committedByPlayerId" = player_penalty_commited.playerId
                    LEFT JOIN pbp_p AS player_penalty_drawing
                    ON 
                        pbp.gameid = player_penalty_drawing.gameid AND 
                        pbp."details.drawnByPlayerId" = player_penalty_drawing.playerId
                ), df_proc AS (
                    SELECT
                        *,
                        -- Other data points to create
                        -- \\ Power play durations to capture
                        -- Last minute 
                        timeInPeriod_continue + "details.duration" AS timeInPeriod_penalty_maxtime,
                        -- Period and power play team of the power play sequence
                        LAST_VALUE(timeInPeriod_penalty_maxtime IGNORE NULLS) OVER (
                            PARTITION BY gameid ORDER BY "periodDescriptor.number" ASC, timeInPeriod ASC
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                        ) AS temp_pp_lag,
                        LAST_VALUE(penalty_drawing_team IGNORE NULLS) OVER (
                            PARTITION BY gameid ORDER BY "periodDescriptor.number" ASC, timeInPeriod ASC
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                        ) AS temp_pp_lag_team,
                        -- Period sought and establishment if within the periods
                        CASE
                            WHEN timeInPeriod_continue > temp_pp_lag THEN NULL -- Past penalty period
                            ELSE temp_pp_lag
                        END AS penalty_period,
                        CASE
                            WHEN timeInPeriod_continue > temp_pp_lag THEN NULL -- Past penalty period
                            ELSE temp_pp_lag_team
                        END AS penalty_team,

                        -- \\ Regarding previous events
                        -- owner of previous event
                        LAG("details.eventOwnerTeam") OVER (
                            PARTITION BY gameid, "periodDescriptor.number"
                            ORDER BY [timeInPeriod]
                        ) AS eventOwnerTeam_lag1,
                        -- Zone code for previous event
                        LAG("details.zoneCode") OVER (
                            PARTITION BY gameid, "periodDescriptor.number"
                            ORDER BY [timeInPeriod]
                        ) AS zoneCode_lag1, 
                        -- Previous event stats
                        LAG("typeDescKey") OVER (
                            PARTITION BY gameid, "periodDescriptor.number"
                            ORDER BY [timeInPeriod]
                        ) AS typeDescKey_lag1, 
                        -- Team and player of blocker team
                        LAG(blocker_team) OVER (
                            PARTITION BY gameid, "periodDescriptor.number"
                            ORDER BY [timeInPeriod]
                        ) AS "blocker_team_lag1", 
                        LAG(blocker_player) OVER (
                            PARTITION BY gameid, "periodDescriptor.number"
                            ORDER BY [timeInPeriod]
                        ) AS "blocker_player_lag1", 
                        -- Team and player of shooter team
                        LAG(shooter_team) OVER (
                            PARTITION BY gameid, "periodDescriptor.number"
                            ORDER BY [timeInPeriod]
                        ) AS "shooter_team_lag1", 
                        LAG(shooter_player) OVER (
                            PARTITION BY gameid, "periodDescriptor.number"
                            ORDER BY [timeInPeriod]
                        ) AS "shooter_player_lag1",
                        -- Write out about the action of the team,
                        --  Is the action by the same team or different?
                        CASE 
                            -- For blocked shot, need to flip
                            --  To simplify, I can just match with first argument equal to "blocker_team_lag1"
                            WHEN 
                                ("details.eventOwnerTeam" != "eventOwnerTeam_lag1") AND
                                ("blocker_team_lag1" IS NOT NULL)
                                THEN 'Own action'
                            WHEN 
                                ("details.eventOwnerTeam" == "eventOwnerTeam_lag1") AND
                                ("blocker_team_lag1" IS NOT NULL)
                                THEN 'Opponent action'
                            -- The rest
                            WHEN ("eventOwnerTeam_lag1" IS NULL) THEN 'Neutral'
                            WHEN ("details.eventOwnerTeam" = "eventOwnerTeam_lag1") THEN 'Own action'
                            WHEN ("details.eventOwnerTeam" IS NULL) THEN 'Neutral'
                            ELSE 'Opponent action'
                        END AS eventOwnerTeam_sequence
                    FROM df
                )
                SELECT *
                FROM df_proc
            """
        )

if __name__ == "__main__":
    None