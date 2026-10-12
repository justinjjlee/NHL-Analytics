"""
Hit Analytics Processing Engine (Decade Review & Daily Pipeline).

Extracts and tabulates post-hit dynamics, event sequences (+1 to +4),
rink coordinate normalizations, 30-second post-hit scoring windows,
and team/zone benchmarks across single seasons or a multi-year decade.

Designed for reusable execution across:
  - Local CLI workflows (uv run python ...)
  - GitHub Actions automated daily updates
  - Databricks job workflows & Unity Catalog volumes
"""

import os
import sys
import glob
import time
import shutil
import logging
import duckdb
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HitAnalytics")


CATEGORY_MAP = {
    "shot-on-goal": "Shot Attempt",
    "missed-shot": "Shot Attempt",
    "blocked-shot": "Shot Attempt",
    "goal": "Goal",
    "takeaway": "Possession Change",
    "giveaway": "Possession Change",
    "hit": "Physical Battle",
    "stoppage": "Stoppage",
    "penalty": "Penalty",
    "faceoff": "Stoppage",
    "period-end": "Stoppage",
    "game-end": "Stoppage",
    "delayed-penalty": "Penalty"
}


class HitAnalyticsProcessor:
    """
    Object-oriented processor for National Hockey League hit analytics.
    Extracts EDA and downstream impact telemetry broken out by season year.
    """

    def __init__(self, play_dir: str = None, output_dir: str = None):
        """
        Initialize the Hit Analytics Processor.

        :param play_dir: Directory containing '{year}_playbyplay.csv' source files.
        :param output_dir: Destination directory for processed CSV artifacts (e.g. latest/action/hits).
        """
        # Resolve repository root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(current_dir, "../../../.."))

        try:
            from config import get_play_dir, get_action_hits_dir
            default_play = get_play_dir()
            default_out = get_action_hits_dir()
        except ImportError:
            try:
                from src.data.apinhle.config import get_play_dir, get_action_hits_dir
                default_play = get_play_dir()
                default_out = get_action_hits_dir()
            except ImportError:
                default_play = os.path.join(repo_root, "latest/play")
                default_out = os.path.join(repo_root, "latest/action/hits")

        self.repo_root = repo_root
        self.play_dir = play_dir or default_play
        self.output_dir = output_dir or default_out

        # Legacy and app sync directories for backwards compatibility
        self.impact_data_dir = os.path.join(repo_root, "dev/decision_science/impact_of_hits/data")
        self.app_eda_dir = os.path.join(repo_root, "app/data/hit_eda")
        self.dev_eda_dir = os.path.join(repo_root, "dev/decision_science/hit_eda/data")

        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create output dir {self.output_dir}: {e}")

    def get_available_seasons(self) -> list:
        """Scan play_dir and return all available playbyplay year prefixes sorted."""
        files = glob.glob(os.path.join(self.play_dir, "*_playbyplay.csv"))
        seasons = []
        for f in files:
            base = os.path.basename(f)
            if not base.endswith("_player.csv") and not base.endswith("_shift.csv"):
                prefix = base.split("_")[0]
                if prefix.isdigit():
                    seasons.append(prefix)
        return sorted(list(set(seasons)))

    def get_decade_seasons(self) -> list:
        """
        Return the 10+ seasons covering 2015-2016 through 2025-2026 (plus current season if present).
        Files: 2015 (20152016) through 2025 (20252026), and 2026 (20262027) if available.
        """
        available = self.get_available_seasons()
        # Default decade targets: 2015 through 2026
        target_years = [str(y) for y in range(2015, 2027)]
        decade = [y for y in target_years if y in available]
        return sorted(decade)

    def process_season(self, season_year: str, save_individual: bool = True) -> dict:
        """
        Process a single season's play-by-play data using DuckDB.
        Generates 8 analytical dataframes for EDA and downstream impact.

        :param season_year: 4-digit start year (e.g. '2015' for 2015-2016 season).
        :param save_individual: Whether to write {season_year}_*.csv files to output_dir.
        :return: Dict of {artifact_name: pd.DataFrame}
        """
        play_file = os.path.join(self.play_dir, f"{season_year}_playbyplay.csv")
        if not os.path.exists(play_file):
            raise FileNotFoundError(f"Source play-by-play file not found: {play_file}")

        logger.info(f"Processing season {season_year} ({os.path.basename(play_file)})...")
        t_start = time.time()

        con = duckdb.connect()

        # Ingest regular-season play-by-play events
        con.execute(f"""
            CREATE TABLE raw_pbp AS
            SELECT 
                gameid,
                idx_season,
                "periodDescriptor.number" AS period,
                "periodDescriptor.periodType" AS period_type,
                eventId,
                timeInPeriod,
                hour(CAST(timeInPeriod AS TIME)) * 60 + minute(CAST(timeInPeriod AS TIME)) AS sec_in_period,
                typeDescKey,
                "details.eventOwnerTeam" AS event_team,
                "details.zoneCode" AS zone_code,
                "details.xCoord" AS x_coord,
                "details.yCoord" AS y_coord,
                "details.hittingPlayerId" AS hitting_player_id,
                "details.hitteePlayerId" AS hittee_player_id,
                "details.shootingPlayerId" AS shooting_player_id,
                "details.scoringPlayerId" AS scoring_player_id
            FROM read_csv('{play_file}', AUTO_DETECT=TRUE)
            WHERE "periodDescriptor.periodType" = 'REG'
              AND timeInPeriod IS NOT NULL
        """)

        # Extract idx_season code (e.g. 20152016)
        season_code_res = con.execute("SELECT idx_season FROM raw_pbp WHERE idx_season IS NOT NULL LIMIT 1").fetchone()
        season_code = str(season_code_res[0]) if season_code_res else f"{season_year}{int(season_year)+1}"

        # 1. Coordinate normalization (attacking direction factor: +1 or -1)
        con.execute("""
            CREATE TABLE team_period_dir AS
            WITH zone_samples AS (
                SELECT 
                    gameid,
                    period,
                    event_team,
                    AVG(CASE 
                        WHEN zone_code = 'O' AND x_coord > 0 THEN 1.0
                        WHEN zone_code = 'O' AND x_coord < 0 THEN -1.0
                        WHEN zone_code = 'D' AND x_coord < 0 THEN 1.0
                        WHEN zone_code = 'D' AND x_coord > 0 THEN -1.0
                        ELSE NULL 
                    END) AS avg_dir
                FROM raw_pbp
                WHERE event_team IS NOT NULL AND x_coord IS NOT NULL
                GROUP BY gameid, period, event_team
            )
            SELECT 
                gameid,
                period,
                event_team,
                CASE WHEN avg_dir < 0 THEN -1.0 ELSE 1.0 END AS dir_factor
            FROM zone_samples
            WHERE avg_dir IS NOT NULL
        """)

        # 2. Extract Hit Sequences (+1 to +4)
        con.execute("""
            CREATE TABLE hit_sequences AS
            WITH pbp_dir AS (
                SELECT 
                    p.*,
                    COALESCE(d.dir_factor, 1.0) AS dir_factor,
                    p.x_coord * COALESCE(d.dir_factor, 1.0) AS norm_x,
                    p.y_coord * COALESCE(d.dir_factor, 1.0) AS norm_y,
                    CASE 
                        WHEN (p.x_coord * COALESCE(d.dir_factor, 1.0)) >= 25.0 THEN 'O'
                        WHEN (p.x_coord * COALESCE(d.dir_factor, 1.0)) <= -25.0 THEN 'D'
                        ELSE 'N'
                    END AS spatial_zone,
                    ROUND((p.x_coord * COALESCE(d.dir_factor, 1.0)) / 10.0) * 10 AS x_bin,
                    ROUND((p.y_coord * COALESCE(d.dir_factor, 1.0)) / 10.0) * 10 AS y_bin
                FROM raw_pbp p
                LEFT JOIN team_period_dir d 
                  ON p.gameid = d.gameid AND p.period = d.period AND p.event_team = d.event_team
            ),
            pbp_lead AS (
                SELECT 
                    *,
                    -- Lead 1
                    LEAD(typeDescKey, 1) OVER w AS next_1_type,
                    LEAD(event_team, 1) OVER w AS next_1_team,
                    LEAD(spatial_zone, 1) OVER w AS next_1_zone,
                    LEAD(sec_in_period, 1) OVER w - sec_in_period AS next_1_dsec,
                    LEAD(norm_x, 1) OVER w AS next_1_norm_x,
                    LEAD(norm_y, 1) OVER w AS next_1_norm_y,
                    
                    -- Lead 2
                    LEAD(typeDescKey, 2) OVER w AS next_2_type,
                    LEAD(event_team, 2) OVER w AS next_2_team,
                    LEAD(spatial_zone, 2) OVER w AS next_2_zone,
                    LEAD(sec_in_period, 2) OVER w - sec_in_period AS next_2_dsec,
                    
                    -- Lead 3
                    LEAD(typeDescKey, 3) OVER w AS next_3_type,
                    LEAD(event_team, 3) OVER w AS next_3_team,
                    LEAD(spatial_zone, 3) OVER w AS next_3_zone,
                    LEAD(sec_in_period, 3) OVER w - sec_in_period AS next_3_dsec,
                    
                    -- Lead 4
                    LEAD(typeDescKey, 4) OVER w AS next_4_type,
                    LEAD(event_team, 4) OVER w AS next_4_team,
                    LEAD(spatial_zone, 4) OVER w AS next_4_zone,
                    LEAD(sec_in_period, 4) OVER w - sec_in_period AS next_4_dsec
                FROM pbp_dir
                WINDOW w AS (PARTITION BY gameid, period ORDER BY sec_in_period, eventId)
            )
            SELECT *
            FROM pbp_lead
            WHERE typeDescKey = 'hit' AND norm_x IS NOT NULL
        """)

        # 3. Extract 30-Second Post-Hit Scoring Window
        df_goal_30s = con.execute("""
            WITH hits AS (
                SELECT 
                    p.gameid,
                    p.idx_season,
                    p.period,
                    p.eventId AS hit_id,
                    p.sec_in_period AS hit_sec,
                    p.event_team AS hit_team,
                    p.spatial_zone AS hit_zone,
                    p.norm_x AS hit_norm_x,
                    p.norm_y AS hit_norm_y,
                    p.x_bin,
                    p.y_bin,
                    p.hitting_player_id,
                    p.hittee_player_id
                FROM hit_sequences p
            ),
            scoring_events AS (
                SELECT 
                    p.gameid,
                    p.period,
                    p.eventId AS event_id,
                    p.sec_in_period AS event_sec,
                    p.event_team,
                    p.typeDescKey AS event_type,
                    p.x_coord,
                    p.y_coord,
                    p.shooting_player_id,
                    p.scoring_player_id
                FROM raw_pbp p
                WHERE p.typeDescKey IN ('shot-on-goal', 'goal', 'missed-shot', 'blocked-shot')
            )
            SELECT 
                h.gameid,
                h.idx_season,
                h.period,
                h.hit_id,
                h.hit_sec,
                h.hit_team,
                h.hit_zone,
                ROUND(h.hit_norm_x, 1) AS hit_norm_x,
                ROUND(h.hit_norm_y, 1) AS hit_norm_y,
                h.x_bin,
                h.y_bin,
                h.hitting_player_id,
                s.event_id,
                s.event_sec,
                s.event_type,
                ROUND(s.x_coord * COALESCE(d.dir_factor, 1.0), 1) AS event_norm_x,
                ROUND(s.y_coord * COALESCE(d.dir_factor, 1.0), 1) AS event_norm_y,
                s.event_sec - h.hit_sec AS delta_sec,
                CASE 
                    WHEN s.event_team = h.hit_team THEN 'Hitting Team (For)'
                    ELSE 'Opponent (Against)'
                END AS outcome_side,
                CASE WHEN s.event_type = 'goal' THEN 1 ELSE 0 END AS is_goal
            FROM hits h
            LEFT JOIN team_period_dir d 
              ON h.gameid = d.gameid AND h.period = d.period AND h.hit_team = d.event_team
            JOIN scoring_events s 
              ON h.gameid = s.gameid 
             AND h.period = s.period 
             AND s.event_sec >= h.hit_sec 
             AND s.event_sec <= h.hit_sec + 30
             AND s.event_id > h.hit_id
            ORDER BY h.gameid, h.hit_id, s.event_sec
        """).df()

        # Register goal_30s table in DuckDB for downstream queries
        con.register("tbl_goal_30s", df_goal_30s)

        # ----------------------------------------------------------------------
        # ARTIFACT 1: EDA Hit Zone Summary (hit_zone_summary.csv)
        # ----------------------------------------------------------------------
        df_zone_summary = con.execute(f"""
            WITH base AS (
                SELECT 
                    spatial_zone AS zoneCode,
                    COUNT(*) AS total_hits,
                    ROUND(SUM(CASE WHEN next_1_team = event_team THEN 1.0 ELSE 0.0 END) * 100.0 / COUNT(*), 2) AS recovery_pct,
                    ROUND(SUM(CASE WHEN next_1_team IS NOT NULL AND next_1_team != event_team THEN 1.0 ELSE 0.0 END) * 100.0 / COUNT(*), 2) AS opponent_possession_pct,
                    ROUND(AVG(CASE WHEN next_1_dsec >= 0 THEN next_1_dsec ELSE NULL END), 2) AS avg_delta_sec,
                    ROUND(MEDIAN(CASE WHEN next_1_dsec >= 0 THEN next_1_dsec ELSE NULL END), 2) AS median_delta_sec,
                    SUM(CASE WHEN next_1_type = 'shot-on-goal' AND next_1_team = event_team THEN 1 ELSE 0 END) AS sog_for,
                    SUM(CASE WHEN next_1_type = 'shot-on-goal' AND next_1_team != event_team THEN 1 ELSE 0 END) AS sog_against,
                    SUM(CASE WHEN next_1_type = 'goal' AND next_1_team = event_team THEN 1 ELSE 0 END) AS goals_for,
                    SUM(CASE WHEN next_1_type = 'goal' AND next_1_team != event_team THEN 1 ELSE 0 END) AS goals_against,
                    SUM(CASE WHEN spatial_zone = 'D' AND next_1_zone IN ('N', 'O') THEN 1 ELSE 0 END) AS zone_exits
                FROM hit_sequences
                WHERE next_1_type IS NOT NULL
                GROUP BY spatial_zone
            )
            SELECT 
                '{season_code}' AS season,
                zoneCode, total_hits, recovery_pct, opponent_possession_pct,
                avg_delta_sec, median_delta_sec, sog_for, sog_against, goals_for, goals_against, zone_exits,
                ROUND(total_hits * 100.0 / SUM(total_hits) OVER (), 2) AS share_of_all_hits_pct,
                ROUND(sog_for * 100.0 / total_hits, 2) AS sog_for_per_100,
                ROUND(sog_against * 100.0 / total_hits, 2) AS sog_against_per_100,
                ROUND((sog_for - sog_against) * 100.0 / total_hits, 2) AS net_sog_per_100,
                ROUND(goals_for * 100.0 / total_hits, 3) AS goals_for_per_100,
                ROUND(goals_against * 100.0 / total_hits, 3) AS goals_against_per_100,
                ROUND((goals_for - goals_against) * 100.0 / total_hits, 3) AS net_goals_per_100,
                CASE WHEN zoneCode = 'D' THEN ROUND(zone_exits * 100.0 / total_hits, 2) ELSE NULL END AS zone_exit_pct
            FROM base
            ORDER BY zoneCode
        """).df()

        # ----------------------------------------------------------------------
        # ARTIFACT 2: EDA Next Events Matrix (hit_next_events.csv)
        # ----------------------------------------------------------------------
        df_next_events = con.execute(f"""
            WITH raw_next AS (
                SELECT 
                    spatial_zone AS zoneCode,
                    CASE 
                        WHEN next_1_type IN ('shot-on-goal', 'missed-shot', 'blocked-shot') THEN 'Shot Attempt'
                        WHEN next_1_type = 'goal' THEN 'Goal'
                        WHEN next_1_type IN ('takeaway', 'giveaway') THEN 'Possession Change'
                        WHEN next_1_type = 'hit' THEN 'Physical Battle'
                        WHEN next_1_type IN ('stoppage', 'faceoff', 'period-end', 'game-end') THEN 'Stoppage'
                        WHEN next_1_type IN ('penalty', 'delayed-penalty') THEN 'Penalty'
                        ELSE 'Other'
                    END AS next_category,
                    CASE 
                        WHEN next_1_team = event_team THEN 'Hitting Team'
                        WHEN next_1_team IS NOT NULL THEN 'Opponent'
                        ELSE 'Stoppage / Neutral'
                    END AS next_owner_rel,
                    next_1_dsec
                FROM hit_sequences
                WHERE next_1_type IS NOT NULL
            ),
            aggregated AS (
                SELECT 
                    zoneCode, next_category, next_owner_rel,
                    COUNT(*) AS event_count,
                    ROUND(AVG(CASE WHEN next_1_dsec >= 0 THEN next_1_dsec ELSE NULL END), 2) AS avg_delta_sec,
                    ROUND(MEDIAN(CASE WHEN next_1_dsec >= 0 THEN next_1_dsec ELSE NULL END), 2) AS median_delta_sec
                FROM raw_next
                GROUP BY zoneCode, next_category, next_owner_rel
            )
            SELECT 
                '{season_code}' AS season,
                a.*,
                ROUND(a.event_count * 100.0 / SUM(a.event_count) OVER (PARTITION BY a.zoneCode), 2) AS pct_of_zone
            FROM aggregated a
            ORDER BY zoneCode, event_count DESC
        """).df()

        # ----------------------------------------------------------------------
        # ARTIFACT 3: Impact Zone KPI (hit_zone_summary_kpi.csv)
        # ----------------------------------------------------------------------
        df_zone_kpi = con.execute(f"""
            WITH total_hits_by_zone AS (
                SELECT 
                    spatial_zone AS hit_zone,
                    COUNT(*) AS total_hits
                FROM hit_sequences
                GROUP BY 1
            ),
            step1_retention AS (
                SELECT 
                    spatial_zone AS hit_zone,
                    SUM(CASE WHEN next_1_team = event_team THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS step1_possession_pct,
                    SUM(CASE WHEN next_1_type = 'takeaway' AND next_1_team = event_team THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS step1_takeaway_pct
                FROM hit_sequences
                GROUP BY 1
            ),
            shots_30s AS (
                SELECT 
                    hit_zone,
                    SUM(CASE WHEN outcome_side = 'Hitting Team (For)' THEN 1 ELSE 0 END) AS shots_for,
                    SUM(CASE WHEN outcome_side = 'Opponent (Against)' THEN 1 ELSE 0 END) AS shots_against,
                    SUM(CASE WHEN outcome_side = 'Hitting Team (For)' AND is_goal = 1 THEN 1 ELSE 0 END) AS goals_for,
                    SUM(CASE WHEN outcome_side = 'Opponent (Against)' AND is_goal = 1 THEN 1 ELSE 0 END) AS goals_against
                FROM tbl_goal_30s
                GROUP BY 1
            )
            SELECT 
                '{season_code}' AS season,
                t.hit_zone,
                CASE 
                    WHEN t.hit_zone = 'D' THEN 'Defensive Zone (Pin & In-Zone Battle)'
                    WHEN t.hit_zone = 'O' THEN 'Offensive Zone (Forecheck Turnover)'
                    WHEN t.hit_zone = 'N' THEN 'Neutral Zone (Rush Disruption)'
                    ELSE t.hit_zone 
                END AS zone_name,
                t.total_hits,
                ROUND(r.step1_possession_pct, 2) AS immediate_possession_pct,
                ROUND(r.step1_takeaway_pct, 2) AS immediate_takeaway_pct,
                COALESCE(s.shots_for, 0) AS shots_for,
                COALESCE(s.shots_against, 0) AS shots_against,
                ROUND(COALESCE(s.shots_for, 0) * 100.0 / t.total_hits, 2) AS shots_for_per_100_hits,
                ROUND(COALESCE(s.shots_against, 0) * 100.0 / t.total_hits, 2) AS shots_against_per_100_hits,
                ROUND((COALESCE(s.shots_for, 0) - COALESCE(s.shots_against, 0)) * 100.0 / t.total_hits, 2) AS net_shot_diff_per_100,
                COALESCE(s.goals_for, 0) AS goals_for,
                COALESCE(s.goals_against, 0) AS goals_against,
                ROUND(COALESCE(s.goals_for, 0) * 100.0 / t.total_hits, 3) AS goals_for_per_100_hits,
                ROUND(COALESCE(s.goals_against, 0) * 100.0 / t.total_hits, 3) AS goals_against_per_100_hits,
                ROUND((COALESCE(s.goals_for, 0) - COALESCE(s.goals_against, 0)) * 100.0 / t.total_hits, 3) AS net_goal_diff_per_100
            FROM total_hits_by_zone t
            JOIN step1_retention r ON t.hit_zone = r.hit_zone
            LEFT JOIN shots_30s s ON t.hit_zone = s.hit_zone
            ORDER BY t.hit_zone
        """).df()

        # ----------------------------------------------------------------------
        # ARTIFACT 4: Tabulated Sequences +1 to +4 (tabulated_hit_sequences.csv)
        # ----------------------------------------------------------------------
        unioned_steps = []
        for step in [1, 2, 3, 4]:
            unioned_steps.append(f"""
                SELECT 
                    spatial_zone AS hit_zone,
                    {step} AS event_step,
                    CASE 
                        WHEN next_{step}_team = event_team THEN 'Hitting Team'
                        WHEN next_{step}_team IS NOT NULL THEN 'Opponent'
                        ELSE 'Neutral / Stoppage'
                    END AS event_owner,
                    CASE 
                        WHEN next_{step}_type IN ('shot-on-goal', 'missed-shot', 'blocked-shot') THEN 'Shot Attempt'
                        WHEN next_{step}_type = 'goal' THEN 'Goal'
                        WHEN next_{step}_type IN ('takeaway', 'giveaway') THEN 'Possession Change'
                        WHEN next_{step}_type = 'hit' THEN 'Physical Battle'
                        WHEN next_{step}_type IN ('stoppage', 'faceoff', 'penalty', 'period-end') THEN 'Stoppage'
                        ELSE 'Other'
                    END AS event_category,
                    next_{step}_type AS event_type,
                    COUNT(*) AS count,
                    ROUND(AVG(CASE WHEN next_{step}_dsec >= 0 THEN next_{step}_dsec ELSE NULL END), 2) AS avg_delta_sec
                FROM hit_sequences
                WHERE next_{step}_type IS NOT NULL
                GROUP BY 1, 2, 3, 4, 5
            """)

        union_sql = " UNION ALL ".join(unioned_steps)
        df_tabulated_sequences = con.execute(f"""
            WITH step_counts AS ({union_sql}),
            zone_step_totals AS (
                SELECT hit_zone, event_step, SUM(count) AS total_step_count
                FROM step_counts
                GROUP BY hit_zone, event_step
            )
            SELECT 
                '{season_code}' AS season,
                s.*,
                ROUND(s.count * 100.0 / t.total_step_count, 3) AS pct_of_step
            FROM step_counts s
            JOIN zone_step_totals t ON s.hit_zone = t.hit_zone AND s.event_step = t.event_step
            ORDER BY s.hit_zone, s.event_step, s.count DESC
        """).df()

        # ----------------------------------------------------------------------
        # ARTIFACT 9: Sequence Transitions (+1 to +4) (hit_sequence_transitions.csv)
        # ----------------------------------------------------------------------
        df_sequence_transitions = con.execute(f"""
            WITH chain_base AS (
                SELECT
                    spatial_zone AS hit_zone,
                    CASE 
                        WHEN next_1_type IN ('shot-on-goal', 'missed-shot', 'blocked-shot') THEN 'Shot Attempt'
                        WHEN next_1_type = 'goal' THEN 'Goal'
                        WHEN next_1_type IN ('takeaway', 'giveaway') THEN 'Possession Change'
                        WHEN next_1_type = 'hit' THEN 'Physical Battle'
                        WHEN next_1_type IN ('stoppage', 'faceoff', 'penalty', 'period-end') THEN 'Stoppage'
                        ELSE 'Other'
                    END AS step_1_cat,
                    CASE 
                        WHEN next_1_team = event_team THEN 'Hitting Team'
                        WHEN next_1_team IS NOT NULL THEN 'Opponent'
                        ELSE 'Neutral / Stoppage'
                    END AS step_1_owner,
                    CASE 
                        WHEN next_2_type IN ('shot-on-goal', 'missed-shot', 'blocked-shot') THEN 'Shot Attempt'
                        WHEN next_2_type = 'goal' THEN 'Goal'
                        WHEN next_2_type IN ('takeaway', 'giveaway') THEN 'Possession Change'
                        WHEN next_2_type = 'hit' THEN 'Physical Battle'
                        WHEN next_2_type IN ('stoppage', 'faceoff', 'penalty', 'period-end') THEN 'Stoppage'
                        ELSE 'Other'
                    END AS step_2_cat,
                    CASE 
                        WHEN next_2_team = event_team THEN 'Hitting Team'
                        WHEN next_2_team IS NOT NULL THEN 'Opponent'
                        ELSE 'Neutral / Stoppage'
                    END AS step_2_owner,
                    CASE 
                        WHEN next_3_type IN ('shot-on-goal', 'missed-shot', 'blocked-shot') THEN 'Shot Attempt'
                        WHEN next_3_type = 'goal' THEN 'Goal'
                        WHEN next_3_type IN ('takeaway', 'giveaway') THEN 'Possession Change'
                        WHEN next_3_type = 'hit' THEN 'Physical Battle'
                        WHEN next_3_type IN ('stoppage', 'faceoff', 'penalty', 'period-end') THEN 'Stoppage'
                        ELSE 'Other'
                    END AS step_3_cat,
                    CASE 
                        WHEN next_4_type IN ('shot-on-goal', 'missed-shot', 'blocked-shot') THEN 'Shot Attempt'
                        WHEN next_4_type = 'goal' THEN 'Goal'
                        WHEN next_4_type IN ('takeaway', 'giveaway') THEN 'Possession Change'
                        WHEN next_4_type = 'hit' THEN 'Physical Battle'
                        WHEN next_4_type IN ('stoppage', 'faceoff', 'penalty', 'period-end') THEN 'Stoppage'
                        ELSE 'Other'
                    END AS step_4_cat,
                    next_1_type, next_2_type, next_3_type, next_4_type
                FROM hit_sequences
            )
            SELECT '{season_code}' AS season, 0 AS from_step, 1 AS to_step, hit_zone, hit_zone AS source_node, step_1_cat AS target_node, COUNT(*) AS count
            FROM chain_base WHERE next_1_type IS NOT NULL GROUP BY 1, 2, 3, 4, 5, 6
            UNION ALL
            SELECT '{season_code}' AS season, 1 AS from_step, 2 AS to_step, hit_zone, step_1_cat AS source_node, step_2_cat AS target_node, COUNT(*) AS count
            FROM chain_base WHERE next_1_type IS NOT NULL AND next_2_type IS NOT NULL GROUP BY 1, 2, 3, 4, 5, 6
            UNION ALL
            SELECT '{season_code}' AS season, 2 AS from_step, 3 AS to_step, hit_zone, step_2_cat AS source_node, step_3_cat AS target_node, COUNT(*) AS count
            FROM chain_base WHERE next_2_type IS NOT NULL AND next_3_type IS NOT NULL GROUP BY 1, 2, 3, 4, 5, 6
            UNION ALL
            SELECT '{season_code}' AS season, 3 AS from_step, 4 AS to_step, hit_zone, step_3_cat AS source_node, step_4_cat AS target_node, COUNT(*) AS count
            FROM chain_base WHERE next_3_type IS NOT NULL AND next_4_type IS NOT NULL GROUP BY 1, 2, 3, 4, 5, 6
            UNION ALL
            SELECT '{season_code}' AS season, 1 AS from_step, 11 AS to_step, hit_zone, step_1_cat AS source_node, step_1_owner AS target_node, COUNT(*) AS count
            FROM chain_base WHERE next_1_type IS NOT NULL GROUP BY 1, 2, 3, 4, 5, 6
            UNION ALL
            SELECT '{season_code}' AS season, 2 AS from_step, 22 AS to_step, hit_zone, step_2_cat AS source_node, step_2_owner AS target_node, COUNT(*) AS count
            FROM chain_base WHERE next_2_type IS NOT NULL GROUP BY 1, 2, 3, 4, 5, 6
            ORDER BY from_step, to_step, hit_zone, count DESC
        """).df()

        # ----------------------------------------------------------------------
        # ARTIFACT 5: Rink Grid Hit Sequences (rink_grid_hit_sequences.csv)
        # ----------------------------------------------------------------------
        df_rink_grid = con.execute(f"""
            WITH temp_goal30s_bins AS (
                SELECT 
                    x_bin, y_bin,
                    SUM(CASE WHEN outcome_side = 'Hitting Team (For)' THEN 1 ELSE 0 END) AS shots_for_30s,
                    SUM(CASE WHEN outcome_side = 'Opponent (Against)' THEN 1 ELSE 0 END) AS shots_against_30s,
                    SUM(CASE WHEN outcome_side = 'Hitting Team (For)' AND is_goal = 1 THEN 1 ELSE 0 END) AS goals_for_30s,
                    SUM(CASE WHEN outcome_side = 'Opponent (Against)' AND is_goal = 1 THEN 1 ELSE 0 END) AS goals_against_30s
                FROM tbl_goal_30s
                GROUP BY 1, 2
            ),
            grid_hits AS (
                SELECT 
                    x_bin,
                    y_bin,
                    spatial_zone,
                    COUNT(*) AS total_hits,
                    ROUND(AVG(CASE WHEN next_1_dsec >= 0 THEN next_1_dsec ELSE NULL END), 1) AS avg_delta_sec,
                    ROUND(SUM(CASE WHEN next_1_team = event_team THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS puck_win_pct,
                    ROUND(SUM(CASE WHEN next_1_type IN ('shot-on-goal','missed-shot','blocked-shot') AND next_1_team = event_team THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS shot_for_pct,
                    ROUND(SUM(CASE WHEN next_1_type IN ('shot-on-goal','missed-shot','blocked-shot') AND next_1_team != event_team THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS shot_against_pct,
                    ROUND(SUM(CASE WHEN next_1_type = 'takeaway' AND next_1_team = event_team THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS takeaway_for_pct,
                    ROUND(SUM(CASE WHEN next_1_type = 'giveaway' AND next_1_team != event_team THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS giveaway_against_pct,
                    ROUND(SUM(CASE WHEN next_1_type = 'hit' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS physical_battle_pct,
                    ROUND(SUM(CASE WHEN next_1_type IN ('stoppage','faceoff','penalty','period-end') THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS stoppage_pct,
                    ROUND(SUM(CASE WHEN next_1_type = 'goal' AND next_1_team = event_team THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS goal_for_pct,
                    ROUND(SUM(CASE WHEN next_1_type = 'goal' AND next_1_team != event_team THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS goal_against_pct,
                    ROUND(SUM(CASE WHEN next_2_team = event_team THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS step2_puck_win_pct,
                    ROUND(SUM(CASE WHEN next_2_type IN ('shot-on-goal','missed-shot','blocked-shot') AND next_2_team = event_team THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS step2_shot_for_pct,
                    ROUND(SUM(CASE WHEN next_2_type IN ('shot-on-goal','missed-shot','blocked-shot') AND next_2_team != event_team THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS step2_shot_against_pct,
                    ROUND(SUM(CASE WHEN next_2_type = 'takeaway' AND next_2_team = event_team THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS step2_takeaway_for_pct,
                    ROUND(SUM(CASE WHEN next_2_type = 'giveaway' AND next_2_team != event_team THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS step2_giveaway_against_pct,
                    ROUND(SUM(CASE WHEN next_2_type = 'hit' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS step2_physical_battle_pct,
                    ROUND(SUM(CASE WHEN next_2_type = 'goal' AND next_2_team = event_team THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS step2_goal_for_pct,
                    ROUND(SUM(CASE WHEN next_2_type = 'goal' AND next_2_team != event_team THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS step2_goal_against_pct
                FROM hit_sequences
                WHERE x_bin IS NOT NULL AND y_bin IS NOT NULL
                  AND ABS(x_bin) <= 100 AND ABS(y_bin) <= 40
                GROUP BY 1, 2, 3
            )
            SELECT 
                '{season_code}' AS season,
                g.*,
                ROUND(g.total_hits * 100.0 / SUM(g.total_hits) OVER (), 2) AS density_pct,
                COALESCE(b.shots_for_30s, 0) AS shots_for_30s,
                COALESCE(b.shots_against_30s, 0) AS shots_against_30s,
                COALESCE(b.goals_for_30s, 0) AS goals_for_30s,
                COALESCE(b.goals_against_30s, 0) AS goals_against_30s,
                ROUND(COALESCE(b.shots_for_30s, 0) * 100.0 / g.total_hits, 2) AS sog_for_per_100,
                ROUND(COALESCE(b.shots_against_30s, 0) * 100.0 / g.total_hits, 2) AS sog_against_per_100,
                ROUND((COALESCE(b.shots_for_30s, 0) - COALESCE(b.shots_against_30s, 0)) * 100.0 / g.total_hits, 2) AS net_sog_per_100,
                ROUND(COALESCE(b.goals_for_30s, 0) * 100.0 / g.total_hits, 3) AS goals_for_per_100,
                ROUND(COALESCE(b.goals_against_30s, 0) * 100.0 / g.total_hits, 3) AS goals_against_per_100,
                ROUND((COALESCE(b.goals_for_30s, 0) - COALESCE(b.goals_against_30s, 0)) * 100.0 / g.total_hits, 3) AS net_goals_per_100
            FROM grid_hits g
            LEFT JOIN temp_goal30s_bins b ON g.x_bin = b.x_bin AND g.y_bin = b.y_bin
            ORDER BY g.spatial_zone, g.total_hits DESC
        """).df()

        # ----------------------------------------------------------------------
        # ARTIFACT 6: Team Season Hit KPI (team_season_hit_kpi.csv)
        # ----------------------------------------------------------------------
        df_team_kpi = con.execute(f"""
            WITH team_games AS (
                SELECT 
                    event_team AS team,
                    COUNT(DISTINCT gameid) AS games_played
                FROM raw_pbp
                WHERE event_team IS NOT NULL
                GROUP BY 1
            ),
            team_hits AS (
                SELECT 
                    h.event_team AS team,
                    COUNT(*) AS total_hits,
                    SUM(CASE WHEN h.spatial_zone = 'O' THEN 1 ELSE 0 END) AS ozone_hits,
                    SUM(CASE WHEN h.spatial_zone = 'D' THEN 1 ELSE 0 END) AS dzone_hits,
                    SUM(CASE WHEN h.spatial_zone = 'N' THEN 1 ELSE 0 END) AS nzone_hits,
                    SUM(CASE WHEN h.next_1_team = h.event_team THEN 1 ELSE 0 END) AS puck_wins,
                    SUM(CASE WHEN h.next_1_type IN ('shot-on-goal','missed-shot','blocked-shot') AND h.next_1_team = h.event_team THEN 1 ELSE 0 END) AS step1_shots_for,
                    SUM(CASE WHEN h.next_1_type IN ('shot-on-goal','missed-shot','blocked-shot') AND h.next_1_team != h.event_team THEN 1 ELSE 0 END) AS step1_shots_against
                FROM hit_sequences h
                WHERE h.event_team IS NOT NULL
                GROUP BY 1
            ),
            team_shots_30s AS (
                SELECT 
                    hit_team AS team,
                    SUM(CASE WHEN outcome_side = 'Hitting Team (For)' THEN 1 ELSE 0 END) AS shots_for_30s,
                    SUM(CASE WHEN outcome_side = 'Opponent (Against)' THEN 1 ELSE 0 END) AS shots_against_30s,
                    SUM(CASE WHEN outcome_side = 'Hitting Team (For)' AND is_goal = 1 THEN 1 ELSE 0 END) AS goals_for_30s,
                    SUM(CASE WHEN outcome_side = 'Opponent (Against)' AND is_goal = 1 THEN 1 ELSE 0 END) AS goals_against_30s
                FROM tbl_goal_30s
                GROUP BY 1
            ),
            base_teams AS (
                SELECT 
                    '{season_code}' AS season,
                    th.team,
                    tg.games_played,
                    th.total_hits,
                    ROUND(th.total_hits * 1.0 / tg.games_played, 1) AS hits_per_game,
                    ROUND(th.ozone_hits * 100.0 / th.total_hits, 1) AS ozone_pct,
                    ROUND(th.dzone_hits * 100.0 / th.total_hits, 1) AS dzone_pct,
                    ROUND(th.nzone_hits * 100.0 / th.total_hits, 1) AS nzone_pct,
                    ROUND(th.puck_wins * 100.0 / th.total_hits, 1) AS puck_win_pct,
                    ROUND(th.step1_shots_for * 100.0 / th.total_hits, 1) AS step1_shot_for_pct,
                    ROUND(th.step1_shots_against * 100.0 / th.total_hits, 1) AS step1_shot_against_pct,
                    COALESCE(ts.shots_for_30s, 0) AS shots_for_30s,
                    COALESCE(ts.shots_against_30s, 0) AS shots_against_30s,
                    ROUND(COALESCE(ts.shots_for_30s, 0) * 100.0 / th.total_hits, 2) AS shots_for_per_100,
                    ROUND(COALESCE(ts.shots_against_30s, 0) * 100.0 / th.total_hits, 2) AS shots_against_per_100,
                    ROUND((COALESCE(ts.shots_for_30s, 0) - COALESCE(ts.shots_against_30s, 0)) * 100.0 / th.total_hits, 2) AS net_shots_per_100,
                    COALESCE(ts.goals_for_30s, 0) AS goals_for_30s,
                    COALESCE(ts.goals_against_30s, 0) AS goals_against_30s,
                    ROUND((COALESCE(ts.goals_for_30s, 0) - COALESCE(ts.goals_against_30s, 0)) * 100.0 / th.total_hits, 3) AS net_goals_per_100
                FROM team_hits th
                JOIN team_games tg ON th.team = tg.team
                LEFT JOIN team_shots_30s ts ON th.team = ts.team
            ),
            league_avg AS (
                SELECT 
                    '{season_code}' AS season,
                    'LEAGUE_AVG' AS team,
                    SUM(games_played) AS games_played,
                    SUM(total_hits) AS total_hits,
                    ROUND(AVG(hits_per_game), 1) AS hits_per_game,
                    ROUND(AVG(ozone_pct), 1) AS ozone_pct,
                    ROUND(AVG(dzone_pct), 1) AS dzone_pct,
                    ROUND(AVG(nzone_pct), 1) AS nzone_pct,
                    ROUND(AVG(puck_win_pct), 1) AS puck_win_pct,
                    ROUND(AVG(step1_shot_for_pct), 1) AS step1_shot_for_pct,
                    ROUND(AVG(step1_shot_against_pct), 1) AS step1_shot_against_pct,
                    SUM(shots_for_30s) AS shots_for_30s,
                    SUM(shots_against_30s) AS shots_against_30s,
                    ROUND(AVG(shots_for_per_100), 2) AS shots_for_per_100,
                    ROUND(AVG(shots_against_per_100), 2) AS shots_against_per_100,
                    ROUND(AVG(net_shots_per_100), 2) AS net_shots_per_100,
                    SUM(goals_for_30s) AS goals_for_30s,
                    SUM(goals_against_30s) AS goals_against_30s,
                    ROUND(AVG(net_goals_per_100), 3) AS net_goals_per_100
                FROM base_teams
            )
            SELECT * FROM base_teams
            UNION ALL
            SELECT * FROM league_avg
            ORDER BY team
        """).df()

        # ----------------------------------------------------------------------
        # ARTIFACT 7: Team Season Spatial Hits (team_season_spatial_hits.csv)
        # ----------------------------------------------------------------------
        df_team_spatial = con.execute(f"""
            WITH team_bins AS (
                SELECT 
                    event_team AS team,
                    x_bin,
                    y_bin,
                    COUNT(*) AS hit_count
                FROM hit_sequences
                WHERE event_team IS NOT NULL AND x_bin IS NOT NULL AND y_bin IS NOT NULL
                  AND ABS(x_bin) <= 100 AND ABS(y_bin) <= 40
                GROUP BY 1, 2, 3
            ),
            team_totals AS (
                SELECT team, SUM(hit_count) AS total_hits
                FROM team_bins
                GROUP BY 1
            ),
            league_bins AS (
                SELECT 
                    'LEAGUE_AVG' AS team,
                    x_bin,
                    y_bin,
                    ROUND(AVG(hit_count), 1) AS hit_count
                FROM team_bins
                GROUP BY 1, 2, 3
            ),
            league_totals AS (
                SELECT team, SUM(hit_count) AS total_hits
                FROM league_bins
                GROUP BY 1
            ),
            combined AS (
                SELECT 
                    '{season_code}' AS season,
                    b.team, b.x_bin, b.y_bin, b.hit_count, 
                    ROUND(b.hit_count * 100.0 / t.total_hits, 2) AS density_pct
                FROM team_bins b JOIN team_totals t ON b.team = t.team
                UNION ALL
                SELECT 
                    '{season_code}' AS season,
                    b.team, b.x_bin, b.y_bin, b.hit_count, 
                    ROUND(b.hit_count * 100.0 / t.total_hits, 2) AS density_pct
                FROM league_bins b JOIN league_totals t ON b.team = t.team
            )
            SELECT * FROM combined
            ORDER BY team, hit_count DESC
        """).df()

        results = {
            "hit_zone_summary": df_zone_summary,
            "hit_next_events": df_next_events,
            "hit_zone_summary_kpi": df_zone_kpi,
            "tabulated_hit_sequences": df_tabulated_sequences,
            "rink_grid_hit_sequences": df_rink_grid,
            "team_season_hit_kpi": df_team_kpi,
            "team_season_spatial_hits": df_team_spatial,
            "hit_goal_window_30s": df_goal_30s,
            "hit_sequence_transitions": df_sequence_transitions
        }

        # ----------------------------------------------------------------------
        # Save Season-Specific CSV Artifacts in latest/action/hits/
        # ----------------------------------------------------------------------
        if save_individual:
            for name, df in results.items():
                out_path = os.path.join(self.output_dir, f"{season_year}_{name}.csv")
                df.to_csv(out_path, index=False)

            logger.info(f"Saved {len(results)} season files for {season_year} under {self.output_dir} in {time.time() - t_start:.2f}s")

        return results

    def combine_and_save_master(self, season_results_list: list):
        """
        Merge per-season datasets into unified master tables containing all years + 'ALL' totals,
        and write them directly to latest/action/hits/*.csv.
        """
        if not season_results_list:
            logger.warning("No season results provided to combine.")
            return

        logger.info(f"Compiling master datasets across {len(season_results_list)} seasons...")

        # 1. Zone Summary Master (EDA)
        dfs_summary = [res["hit_zone_summary"] for res in season_results_list if "hit_zone_summary" in res]
        if dfs_summary:
            master_summary = pd.concat(dfs_summary, ignore_index=True)
            # Add 'ALL' multi-year rollup row
            all_summary = master_summary.groupby("zoneCode").agg({
                "total_hits": "sum",
                "sog_for": "sum",
                "sog_against": "sum",
                "goals_for": "sum",
                "goals_against": "sum",
                "zone_exits": "sum"
            }).reset_index()
            # Calculate weighted recovery and latencies
            weighted_rec = master_summary.groupby("zoneCode").apply(
                lambda g: (g["recovery_pct"] * g["total_hits"]).sum() / g["total_hits"].sum(),
                include_groups=False
            ).reset_index(name="recovery_pct")
            weighted_opp = master_summary.groupby("zoneCode").apply(
                lambda g: (g["opponent_possession_pct"] * g["total_hits"]).sum() / g["total_hits"].sum(),
                include_groups=False
            ).reset_index(name="opponent_possession_pct")
            weighted_avg_dt = master_summary.groupby("zoneCode").apply(
                lambda g: (g["avg_delta_sec"] * g["total_hits"]).sum() / g["total_hits"].sum(),
                include_groups=False
            ).reset_index(name="avg_delta_sec")
            weighted_med_dt = master_summary.groupby("zoneCode").apply(
                lambda g: (g["median_delta_sec"] * g["total_hits"]).sum() / g["total_hits"].sum(),
                include_groups=False
            ).reset_index(name="median_delta_sec")

            all_summary = all_summary.merge(weighted_rec, on="zoneCode") \
                                     .merge(weighted_opp, on="zoneCode") \
                                     .merge(weighted_avg_dt, on="zoneCode") \
                                     .merge(weighted_med_dt, on="zoneCode")

            tot_all = all_summary["total_hits"].sum()
            all_summary["season"] = "ALL"
            all_summary["share_of_all_hits_pct"] = (all_summary["total_hits"] / tot_all) * 100
            all_summary["sog_for_per_100"] = (all_summary["sog_for"] / all_summary["total_hits"]) * 100
            all_summary["sog_against_per_100"] = (all_summary["sog_against"] / all_summary["total_hits"]) * 100
            all_summary["net_sog_per_100"] = all_summary["sog_for_per_100"] - all_summary["sog_against_per_100"]
            all_summary["goals_for_per_100"] = (all_summary["goals_for"] / all_summary["total_hits"]) * 100
            all_summary["goals_against_per_100"] = (all_summary["goals_against"] / all_summary["total_hits"]) * 100
            all_summary["net_goals_per_100"] = all_summary["goals_for_per_100"] - all_summary["goals_against_per_100"]
            all_summary["zone_exit_pct"] = np.where(
                all_summary["zoneCode"] == "D",
                (all_summary["zone_exits"] / all_summary["total_hits"]) * 100,
                np.nan
            )
            # Reorder columns to match
            col_order = master_summary.columns.tolist()
            all_summary = all_summary[col_order]
            final_summary = pd.concat([master_summary, all_summary], ignore_index=True)
            final_summary.to_csv(os.path.join(self.output_dir, "hit_zone_summary.csv"), index=False)

        # 2. Next Events Master (EDA)
        dfs_events = [res["hit_next_events"] for res in season_results_list if "hit_next_events" in res]
        if dfs_events:
            master_events = pd.concat(dfs_events, ignore_index=True)
            all_events = master_events.groupby(["zoneCode", "next_category", "next_owner_rel"]).agg(
                event_count=("event_count", "sum"),
                avg_delta_sec=("avg_delta_sec", "mean"),
                median_delta_sec=("median_delta_sec", "mean")
            ).reset_index()
            all_events["season"] = "ALL"
            all_events["pct_of_zone"] = all_events.groupby("zoneCode")["event_count"].transform(
                lambda x: (x / x.sum()) * 100
            )
            col_order = master_events.columns.tolist()
            all_events = all_events[col_order]
            final_events = pd.concat([master_events, all_events], ignore_index=True)
            final_events.to_csv(os.path.join(self.output_dir, "hit_next_events.csv"), index=False)

        # 3. Zone KPI Master (Impact of Hits)
        dfs_kpi = [res["hit_zone_summary_kpi"] for res in season_results_list if "hit_zone_summary_kpi" in res]
        if dfs_kpi:
            master_kpi = pd.concat(dfs_kpi, ignore_index=True)
            all_kpi = master_kpi.groupby(["hit_zone", "zone_name"]).agg({
                "total_hits": "sum",
                "shots_for": "sum",
                "shots_against": "sum",
                "goals_for": "sum",
                "goals_against": "sum"
            }).reset_index()
            # Weighted possession & takeaways
            w_poss = master_kpi.groupby("hit_zone").apply(
                lambda g: (g["immediate_possession_pct"] * g["total_hits"]).sum() / g["total_hits"].sum(),
                include_groups=False
            ).reset_index(name="immediate_possession_pct")
            w_take = master_kpi.groupby("hit_zone").apply(
                lambda g: (g["immediate_takeaway_pct"] * g["total_hits"]).sum() / g["total_hits"].sum(),
                include_groups=False
            ).reset_index(name="immediate_takeaway_pct")

            all_kpi = all_kpi.merge(w_poss, on="hit_zone").merge(w_take, on="hit_zone")
            all_kpi["season"] = "ALL"
            all_kpi["shots_for_per_100_hits"] = (all_kpi["shots_for"] / all_kpi["total_hits"]) * 100
            all_kpi["shots_against_per_100_hits"] = (all_kpi["shots_against"] / all_kpi["total_hits"]) * 100
            all_kpi["net_shot_diff_per_100"] = all_kpi["shots_for_per_100_hits"] - all_kpi["shots_against_per_100_hits"]
            all_kpi["goals_for_per_100_hits"] = (all_kpi["goals_for"] / all_kpi["total_hits"]) * 100
            all_kpi["goals_against_per_100_hits"] = (all_kpi["goals_against"] / all_kpi["total_hits"]) * 100
            all_kpi["net_goal_diff_per_100"] = all_kpi["goals_for_per_100_hits"] - all_kpi["goals_against_per_100_hits"]
            col_order = master_kpi.columns.tolist()
            all_kpi = all_kpi[col_order]
            final_kpi = pd.concat([master_kpi, all_kpi], ignore_index=True)
            final_kpi.to_csv(os.path.join(self.output_dir, "hit_zone_summary_kpi.csv"), index=False)

        # 4. Tabulated Sequences Master
        dfs_tab = [res["tabulated_hit_sequences"] for res in season_results_list if "tabulated_hit_sequences" in res]
        if dfs_tab:
            master_tab = pd.concat(dfs_tab, ignore_index=True)
            all_tab = master_tab.groupby(["hit_zone", "event_step", "event_owner", "event_category", "event_type"]).agg(
                count=("count", "sum"),
                avg_delta_sec=("avg_delta_sec", "mean")
            ).reset_index()
            all_tab["season"] = "ALL"
            all_tab["pct_of_step"] = all_tab.groupby(["hit_zone", "event_step"])["count"].transform(
                lambda x: (x / x.sum()) * 100
            )
            col_order = master_tab.columns.tolist()
            all_tab = all_tab[col_order]
            final_tab = pd.concat([master_tab, all_tab], ignore_index=True)
            final_tab.to_csv(os.path.join(self.output_dir, "tabulated_hit_sequences.csv"), index=False)

        # 5. Rink Grid Sequences Master
        dfs_grid = [res["rink_grid_hit_sequences"] for res in season_results_list if "rink_grid_hit_sequences" in res]
        if dfs_grid:
            master_grid = pd.concat(dfs_grid, ignore_index=True)
            all_grid = master_grid.groupby(["x_bin", "y_bin", "spatial_zone"]).agg({
                "total_hits": "sum",
                "shots_for_30s": "sum",
                "shots_against_30s": "sum",
                "goals_for_30s": "sum",
                "goals_against_30s": "sum"
            }).reset_index()
            # Mean rates across seasons for spatial cell
            rate_cols = [c for c in master_grid.columns if (c.endswith("_pct") and c != "density_pct") or c == "avg_delta_sec"]
            mean_rates = master_grid.groupby(["x_bin", "y_bin", "spatial_zone"])[rate_cols].mean().reset_index()
            all_grid = all_grid.merge(mean_rates, on=["x_bin", "y_bin", "spatial_zone"])
            all_grid["season"] = "ALL"
            all_grid["density_pct"] = (all_grid["total_hits"] * 100.0 / all_grid["total_hits"].sum()).round(2)
            all_grid["sog_for_per_100"] = (all_grid["shots_for_30s"] / all_grid["total_hits"]) * 100
            all_grid["sog_against_per_100"] = (all_grid["shots_against_30s"] / all_grid["total_hits"]) * 100
            all_grid["net_sog_per_100"] = all_grid["sog_for_per_100"] - all_grid["sog_against_per_100"]
            all_grid["goals_for_per_100"] = (all_grid["goals_for_30s"] / all_grid["total_hits"]) * 100
            all_grid["goals_against_per_100"] = (all_grid["goals_against_30s"] / all_grid["total_hits"]) * 100
            all_grid["net_goals_per_100"] = all_grid["goals_for_per_100"] - all_grid["goals_against_per_100"]
            col_order = master_grid.columns.tolist()
            all_grid = all_grid[col_order]
            final_grid = pd.concat([master_grid, all_grid], ignore_index=True)
            final_grid.to_csv(os.path.join(self.output_dir, "rink_grid_hit_sequences.csv"), index=False)

        # 6. Team Season KPI Master
        dfs_team = [res["team_season_hit_kpi"] for res in season_results_list if "team_season_hit_kpi" in res]
        if dfs_team:
            master_team = pd.concat(dfs_team, ignore_index=True)
            # Add 'ALL' season rows per team
            regular_teams = master_team[master_team["team"] != "LEAGUE_AVG"]
            all_teams = regular_teams.groupby("team").agg({
                "games_played": "sum",
                "total_hits": "sum",
                "shots_for_30s": "sum",
                "shots_against_30s": "sum",
                "goals_for_30s": "sum",
                "goals_against_30s": "sum"
            }).reset_index()
            all_teams["season"] = "ALL"
            all_teams["hits_per_game"] = (all_teams["total_hits"] / all_teams["games_played"]).round(1)
            # Weighted rates
            pct_cols = ["ozone_pct", "dzone_pct", "nzone_pct", "puck_win_pct", "step1_shot_for_pct", "step1_shot_against_pct"]
            mean_pcts = regular_teams.groupby("team")[pct_cols].mean().round(1).reset_index()
            all_teams = all_teams.merge(mean_pcts, on="team")
            all_teams["shots_for_per_100"] = ((all_teams["shots_for_30s"] * 100.0) / all_teams["total_hits"]).round(2)
            all_teams["shots_against_per_100"] = ((all_teams["shots_against_30s"] * 100.0) / all_teams["total_hits"]).round(2)
            all_teams["net_shots_per_100"] = (all_teams["shots_for_per_100"] - all_teams["shots_against_per_100"]).round(2)
            all_teams["goals_for_30s"] = all_teams["goals_for_30s"]
            all_teams["goals_against_30s"] = all_teams["goals_against_30s"]
            all_teams["net_goals_per_100"] = (((all_teams["goals_for_30s"] - all_teams["goals_against_30s"]) * 100.0) / all_teams["total_hits"]).round(3)

            # League average for ALL
            league_all = pd.DataFrame([{
                "season": "ALL",
                "team": "LEAGUE_AVG",
                "games_played": all_teams["games_played"].sum(),
                "total_hits": all_teams["total_hits"].sum(),
                "hits_per_game": round(all_teams["hits_per_game"].mean(), 1),
                "ozone_pct": round(all_teams["ozone_pct"].mean(), 1),
                "dzone_pct": round(all_teams["dzone_pct"].mean(), 1),
                "nzone_pct": round(all_teams["nzone_pct"].mean(), 1),
                "puck_win_pct": round(all_teams["puck_win_pct"].mean(), 1),
                "step1_shot_for_pct": round(all_teams["step1_shot_for_pct"].mean(), 1),
                "step1_shot_against_pct": round(all_teams["step1_shot_against_pct"].mean(), 1),
                "shots_for_30s": all_teams["shots_for_30s"].sum(),
                "shots_against_30s": all_teams["shots_against_30s"].sum(),
                "shots_for_per_100": round(all_teams["shots_for_per_100"].mean(), 2),
                "shots_against_per_100": round(all_teams["shots_against_per_100"].mean(), 2),
                "net_shots_per_100": round(all_teams["net_shots_per_100"].mean(), 2),
                "goals_for_30s": all_teams["goals_for_30s"].sum(),
                "goals_against_30s": all_teams["goals_against_30s"].sum(),
                "net_goals_per_100": round(all_teams["net_goals_per_100"].mean(), 3)
            }])

            col_order = master_team.columns.tolist()
            all_teams = all_teams[col_order]
            league_all = league_all[col_order]
            final_team = pd.concat([master_team, all_teams, league_all], ignore_index=True)
            final_team.to_csv(os.path.join(self.output_dir, "team_season_hit_kpi.csv"), index=False)

        # 7. Team Spatial Hits Master
        dfs_spatial = [res["team_season_spatial_hits"] for res in season_results_list if "team_season_spatial_hits" in res]
        if dfs_spatial:
            master_spatial = pd.concat(dfs_spatial, ignore_index=True)
            # ALL rollups per team
            reg_spat = master_spatial[master_spatial["team"] != "LEAGUE_AVG"]
            all_spat_bins = reg_spat.groupby(["team", "x_bin", "y_bin"])["hit_count"].sum().reset_index()
            all_spat_totals = all_spat_bins.groupby("team")["hit_count"].sum().reset_index(name="tot_hits")
            all_spat_comb = all_spat_bins.merge(all_spat_totals, on="team")
            all_spat_comb["season"] = "ALL"
            all_spat_comb["density_pct"] = (all_spat_comb["hit_count"] * 100.0 / all_spat_comb["tot_hits"]).round(2)
            all_spat_comb = all_spat_comb.drop(columns=["tot_hits"])

            # League average ALL
            all_league_bins = all_spat_comb.groupby(["x_bin", "y_bin"])["hit_count"].mean().round(1).reset_index()
            all_league_tot = all_league_bins["hit_count"].sum()
            all_league_bins["season"] = "ALL"
            all_league_bins["team"] = "LEAGUE_AVG"
            all_league_bins["density_pct"] = (all_league_bins["hit_count"] * 100.0 / all_league_tot).round(2)

            col_order = master_spatial.columns.tolist()
            all_spat_comb = all_spat_comb[col_order]
            all_league_bins = all_league_bins[col_order]
            final_spatial = pd.concat([master_spatial, all_spat_comb, all_league_bins], ignore_index=True)
            final_spatial.to_csv(os.path.join(self.output_dir, "team_season_spatial_hits.csv"), index=False)

        # 8. 30s Window Master
        dfs_30s = [res["hit_goal_window_30s"] for res in season_results_list if "hit_goal_window_30s" in res]
        if dfs_30s:
            master_30s = pd.concat(dfs_30s, ignore_index=True)
            master_30s.to_csv(os.path.join(self.output_dir, "hit_goal_window_30s.csv"), index=False)

        # 9. Sequence Transitions Master
        dfs_trans = [res["hit_sequence_transitions"] for res in season_results_list if "hit_sequence_transitions" in res]
        if dfs_trans:
            master_trans = pd.concat(dfs_trans, ignore_index=True)
            all_trans = master_trans.groupby(["from_step", "to_step", "hit_zone", "source_node", "target_node"]).agg(
                count=("count", "sum")
            ).reset_index()
            all_trans["season"] = "ALL"
            col_order = master_trans.columns.tolist()
            all_trans = all_trans[col_order]
            final_trans = pd.concat([master_trans, all_trans], ignore_index=True)
            final_trans.to_csv(os.path.join(self.output_dir, "hit_sequence_transitions.csv"), index=False)

        logger.info(f"Successfully generated all master datasets under {self.output_dir}")

    def sync_to_legacy_locations(self):
        """
        Mirror the newly compiled master datasets to legacy directories
        (dev/decision_science/impact_of_hits/data/ and app/data/hit_eda/)
        to ensure zero regressions across old test scripts or standalone consumers.
        """
        try:
            from config import IS_DATABRICKS
        except ImportError:
            try:
                from src.data.apinhle.config import IS_DATABRICKS
            except ImportError:
                IS_DATABRICKS = False

        if IS_DATABRICKS:
            logger.info("Running in Databricks environment; skipping local repository clone mirroring.")
            return

        artifacts_to_copy = [
            "hit_zone_summary.csv",
            "hit_next_events.csv",
            "hit_zone_summary_kpi.csv",
            "tabulated_hit_sequences.csv",
            "rink_grid_hit_sequences.csv",
            "team_season_hit_kpi.csv",
            "team_season_spatial_hits.csv",
            "hit_goal_window_30s.csv",
            "hit_sequence_transitions.csv"
        ]

        for target_dir in [self.impact_data_dir, self.app_eda_dir, self.dev_eda_dir]:
            os.makedirs(target_dir, exist_ok=True)
            for art in artifacts_to_copy:
                src = os.path.join(self.output_dir, art)
                dst = os.path.join(target_dir, art)
                if os.path.exists(src):
                    try:
                        shutil.copy2(src, dst)
                    except Exception as e:
                        logger.warning(f"Could not copy {src} to {dst}: {e}")

        logger.info("Synchronized master datasets to legacy/app data directories.")

    def run(self, seasons: list = None, incremental: bool = False, sync_legacy: bool = True):
        """
        End-to-end execution method.

        :param seasons: List of years to process (e.g. ['2015', ..., '2026'] or None for decade).
        :param incremental: If True, skip already processed historical seasons.
        :param sync_legacy: If True, mirror master artifacts to legacy directories.
        """
        target_seasons = seasons or self.get_decade_seasons()
        logger.info(f"Initiating Hit Analytics pipeline for {len(target_seasons)} seasons: {target_seasons}")

        try:
            from config import get_current_season_year
            current_season = str(get_current_season_year())
        except ImportError:
            try:
                from src.data.apinhle.config import get_current_season_year
                current_season = str(get_current_season_year())
            except ImportError:
                current_season = str(time.strftime("%Y"))

        max_target_season = str(max([int(s) for s in target_seasons])) if target_seasons else current_season
        active_seasons = {current_season, max_target_season}

        season_results = []

        for s in target_seasons:
            summary_file = os.path.join(self.output_dir, f"{s}_hit_zone_summary.csv")
            # If incremental and not an active season, attempt to load cached files
            if incremental and os.path.exists(summary_file) and (s not in active_seasons):
                logger.info(f"Season {s} already processed and cached. Reading cached records...")
                try:
                    res = {
                        "hit_zone_summary": pd.read_csv(os.path.join(self.output_dir, f"{s}_hit_zone_summary.csv")),
                        "hit_next_events": pd.read_csv(os.path.join(self.output_dir, f"{s}_hit_next_events.csv")),
                        "hit_zone_summary_kpi": pd.read_csv(os.path.join(self.output_dir, f"{s}_hit_zone_summary_kpi.csv")),
                        "tabulated_hit_sequences": pd.read_csv(os.path.join(self.output_dir, f"{s}_tabulated_hit_sequences.csv")),
                        "rink_grid_hit_sequences": pd.read_csv(os.path.join(self.output_dir, f"{s}_rink_grid_hit_sequences.csv")),
                        "team_season_hit_kpi": pd.read_csv(os.path.join(self.output_dir, f"{s}_team_season_hit_kpi.csv")),
                        "team_season_spatial_hits": pd.read_csv(os.path.join(self.output_dir, f"{s}_team_season_spatial_hits.csv")),
                        "hit_goal_window_30s": pd.read_csv(os.path.join(self.output_dir, f"{s}_hit_goal_window_30s.csv")),
                        "hit_sequence_transitions": pd.read_csv(os.path.join(self.output_dir, f"{s}_hit_sequence_transitions.csv"))
                    }
                    season_results.append(res)
                    continue
                except Exception as e:
                    logger.warning(f"Cached files incomplete for {s}: {e}. Recomputing...")

            # Process season
            res = self.process_season(s, save_individual=True)
            season_results.append(res)

        # Merge into master multi-season tables
        self.combine_and_save_master(season_results)

        if sync_legacy:
            self.sync_to_legacy_locations()

        logger.info(f"Hit Analytics processing completed successfully for {len(target_seasons)} seasons!")
