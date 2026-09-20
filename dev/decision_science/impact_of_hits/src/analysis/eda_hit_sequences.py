'''
Exploratory Data Analysis & Sequence Extraction: Impact of Hits
Tracks event sequences (Events +1 through +4), standardizes rink coordinates,
evaluates 30-second shot & goal windows, and produces tabulated data artifacts
for spatial heatmaps, click-level distribution analysis, and team-season benchmarking.
'''

import os
import duckdb
import pandas as pd
import numpy as np

def run_pipeline():
    # Resolve repository root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, "../../../../.."))
    output_dir = os.path.join(repo_root, "dev/decision_science/impact_of_hits/data")
    os.makedirs(output_dir, exist_ok=True)

    play_2024 = os.path.join(repo_root, "latest/play/2024_playbyplay.csv")
    play_2025 = os.path.join(repo_root, "latest/play/2025_playbyplay.csv")

    print("Connecting to DuckDB and ingesting 2024 & 2025 play-by-play files...")
    con = duckdb.connect()

    # Load 2024 & 2025 regular season play-by-play data
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
        FROM read_csv(['{play_2024}', '{play_2025}'], AUTO_DETECT=TRUE)
        WHERE "periodDescriptor.periodType" = 'REG'
          AND timeInPeriod IS NOT NULL
    """)

    total_events = con.execute("SELECT COUNT(*) FROM raw_pbp").fetchone()[0]
    total_hits = con.execute("SELECT COUNT(*) FROM raw_pbp WHERE typeDescKey = 'hit'").fetchone()[0]
    print(f"Loaded {total_events:,} regular season events, including {total_hits:,} recorded hits.")

    # 1. Determine attacking direction per game, period, and team to normalize coordinates:
    # If a team has hits/shots in O-zone with x < 0, they attack towards -X (factor = -1), else +X (factor = 1).
    print("Computing attacking direction mapping for coordinate normalization...")
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

    # 2. Extract Event Sequences (+1 through +4) following each hit with physical ice zones & bins
    print("Extracting multi-event sequences (+1 to +4) following each hit...")
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

    # 3. Tabulate Sequence Statistics across Steps 1 to 4
    print("Tabulating sequence transitions and probabilities...")
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
                ROUND(AVG(next_{step}_dsec), 2) AS avg_delta_sec
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
            s.*,
            ROUND(s.count * 100.0 / t.total_step_count, 3) AS pct_of_step
        FROM step_counts s
        JOIN zone_step_totals t ON s.hit_zone = t.hit_zone AND s.event_step = t.event_step
        ORDER BY s.hit_zone, s.event_step, s.count DESC
    """).df()

    tabulated_seq_path = os.path.join(output_dir, "tabulated_hit_sequences.csv")
    df_tabulated_sequences.to_csv(tabulated_seq_path, index=False)
    print(f"Saved tabulated sequences to {tabulated_seq_path} ({len(df_tabulated_sequences):,} rows)")

    # 4. Extract 30-Second Window Shot & Goal Events
    print("Extracting 30-second post-hit shot attempts and goals...")
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

    goal_30s_path = os.path.join(output_dir, "hit_goal_window_30s.csv")
    df_goal_30s.to_csv(goal_30s_path, index=False)
    print(f"Saved 30s window events to {goal_30s_path} ({len(df_goal_30s):,} records)")

    # 5. Compute Executive Zone KPI Summary
    print("Computing executive KPI summary table...")
    df_kpi = con.execute("""
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
                COUNT(DISTINCT hit_id) AS hits_with_shot_attempts,
                SUM(CASE WHEN outcome_side = 'Hitting Team (For)' THEN 1 ELSE 0 END) AS shots_for,
                SUM(CASE WHEN outcome_side = 'Opponent (Against)' THEN 1 ELSE 0 END) AS shots_against,
                SUM(CASE WHEN outcome_side = 'Hitting Team (For)' AND is_goal = 1 THEN 1 ELSE 0 END) AS goals_for,
                SUM(CASE WHEN outcome_side = 'Opponent (Against)' AND is_goal = 1 THEN 1 ELSE 0 END) AS goals_against
            FROM read_csv_auto('{0}')
            GROUP BY 1
        )
        SELECT 
            t.hit_zone,
            CASE 
                WHEN t.hit_zone = 'O' THEN 'Offensive Zone (Forecheck)'
                WHEN t.hit_zone = 'D' THEN 'Defensive Zone (In-Zone)'
                WHEN t.hit_zone = 'N' THEN 'Neutral Zone (Rush Disruption)'
                ELSE t.hit_zone 
            END AS zone_name,
            t.total_hits,
            ROUND(r.step1_possession_pct, 2) AS immediate_possession_pct,
            ROUND(r.step1_takeaway_pct, 2) AS immediate_takeaway_pct,
            s.shots_for,
            s.shots_against,
            ROUND(s.shots_for * 100.0 / t.total_hits, 2) AS shots_for_per_100_hits,
            ROUND(s.shots_against * 100.0 / t.total_hits, 2) AS shots_against_per_100_hits,
            ROUND((s.shots_for - s.shots_against) * 100.0 / t.total_hits, 2) AS net_shot_diff_per_100,
            s.goals_for,
            s.goals_against,
            ROUND(s.goals_for * 100.0 / t.total_hits, 3) AS goals_for_per_100_hits,
            ROUND(s.goals_against * 100.0 / t.total_hits, 3) AS goals_against_per_100_hits,
            ROUND((s.goals_for - s.goals_against) * 100.0 / t.total_hits, 3) AS net_goal_diff_per_100
        FROM total_hits_by_zone t
        JOIN step1_retention r ON t.hit_zone = r.hit_zone
        JOIN shots_30s s ON t.hit_zone = s.hit_zone
        ORDER BY t.hit_zone
    """.format(goal_30s_path)).df()

    kpi_path = os.path.join(output_dir, "hit_zone_summary_kpi.csv")
    df_kpi.to_csv(kpi_path, index=False)
    print(f"Saved executive KPI summary to {kpi_path}")

    # 6. Compute Spatial Rink Grid (Heatmap + Click-Inspection Sequences)
    print("Computing spatial ice grid metrics for interactive heatmap and click inspection...")
    con.execute(f"""
        CREATE TABLE temp_goal30s_bins AS
        SELECT 
            x_bin, y_bin,
            SUM(CASE WHEN outcome_side = 'Hitting Team (For)' THEN 1 ELSE 0 END) AS shots_for_30s,
            SUM(CASE WHEN outcome_side = 'Opponent (Against)' THEN 1 ELSE 0 END) AS shots_against_30s,
            SUM(CASE WHEN outcome_side = 'Hitting Team (For)' AND is_goal = 1 THEN 1 ELSE 0 END) AS goals_for_30s,
            SUM(CASE WHEN outcome_side = 'Opponent (Against)' AND is_goal = 1 THEN 1 ELSE 0 END) AS goals_against_30s
        FROM read_csv_auto('{goal_30s_path}')
        GROUP BY 1, 2
    """)

    df_rink_grid = con.execute("""
        WITH grid_hits AS (
            SELECT 
                x_bin,
                y_bin,
                spatial_zone,
                COUNT(*) AS total_hits,
                ROUND(AVG(next_1_dsec), 1) AS avg_delta_sec,
                ROUND(SUM(CASE WHEN next_1_team = event_team THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS puck_win_pct,
                -- Event +1 detail categories
                ROUND(SUM(CASE WHEN next_1_type IN ('shot-on-goal','missed-shot','blocked-shot') AND next_1_team = event_team THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS shot_for_pct,
                ROUND(SUM(CASE WHEN next_1_type IN ('shot-on-goal','missed-shot','blocked-shot') AND next_1_team != event_team THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS shot_against_pct,
                ROUND(SUM(CASE WHEN next_1_type = 'takeaway' AND next_1_team = event_team THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS takeaway_for_pct,
                ROUND(SUM(CASE WHEN next_1_type = 'giveaway' AND next_1_team != event_team THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS giveaway_against_pct,
                ROUND(SUM(CASE WHEN next_1_type = 'hit' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS physical_battle_pct,
                ROUND(SUM(CASE WHEN next_1_type IN ('stoppage','faceoff','penalty','period-end') THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS stoppage_pct,
                ROUND(SUM(CASE WHEN next_1_type = 'goal' AND next_1_team = event_team THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS goal_for_pct,
                ROUND(SUM(CASE WHEN next_1_type = 'goal' AND next_1_team != event_team THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS goal_against_pct,
                -- Event +2 quick summary
                ROUND(SUM(CASE WHEN next_2_team = event_team THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS step2_puck_win_pct,
                ROUND(SUM(CASE WHEN next_2_type IN ('shot-on-goal','missed-shot','blocked-shot') AND next_2_team = event_team THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS step2_shot_for_pct,
                ROUND(SUM(CASE WHEN next_2_type IN ('shot-on-goal','missed-shot','blocked-shot') AND next_2_team != event_team THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS step2_shot_against_pct
            FROM hit_sequences
            WHERE x_bin IS NOT NULL AND y_bin IS NOT NULL
              AND ABS(x_bin) <= 100 AND ABS(y_bin) <= 40
            GROUP BY 1, 2, 3
        ),
        total_all AS (
            SELECT SUM(total_hits) AS all_hits FROM grid_hits
        )
        SELECT 
            g.*,
            ROUND(g.total_hits * 100.0 / a.all_hits, 2) AS density_pct,
            COALESCE(s.shots_for_30s, 0) AS shots_for_30s,
            COALESCE(s.shots_against_30s, 0) AS shots_against_30s,
            COALESCE(s.shots_for_30s, 0) - COALESCE(s.shots_against_30s, 0) AS net_shots_30s,
            COALESCE(s.goals_for_30s, 0) AS goals_for_30s,
            COALESCE(s.goals_against_30s, 0) AS goals_against_30s,
            COALESCE(s.goals_for_30s, 0) - COALESCE(s.goals_against_30s, 0) AS net_goals_30s
        FROM grid_hits g
        CROSS JOIN total_all a
        LEFT JOIN temp_goal30s_bins s ON g.x_bin = s.x_bin AND g.y_bin = s.y_bin
        ORDER BY g.total_hits DESC
    """).df()

    rink_grid_path = os.path.join(output_dir, "rink_grid_hit_sequences.csv")
    df_rink_grid.to_csv(rink_grid_path, index=False)
    print(f"Saved spatial rink grid to {rink_grid_path} ({len(df_rink_grid):,} grid cells)")

    # 7. Compute Team & Season KPI Benchmarks
    print("Computing team and season hitting profiles & benchmarks...")
    con.execute("""
        CREATE TABLE temp_team_seasons AS
        SELECT DISTINCT idx_season, event_team AS team FROM hit_sequences WHERE event_team IS NOT NULL
    """)

    # Season games played per team
    con.execute("""
        CREATE TABLE team_games AS
        SELECT 
            idx_season,
            event_team AS team,
            COUNT(DISTINCT gameid) AS games_played
        FROM raw_pbp
        WHERE event_team IS NOT NULL
        GROUP BY 1, 2
    """)

    # Base team hitting stats by season
    con.execute(f"""
        CREATE TABLE team_season_base AS
        WITH team_hits AS (
            SELECT 
                h.idx_season,
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
            GROUP BY 1, 2
        ),
        team_shots_30s AS (
            SELECT 
                idx_season,
                hit_team AS team,
                SUM(CASE WHEN outcome_side = 'Hitting Team (For)' THEN 1 ELSE 0 END) AS shots_for_30s,
                SUM(CASE WHEN outcome_side = 'Opponent (Against)' THEN 1 ELSE 0 END) AS shots_against_30s,
                SUM(CASE WHEN outcome_side = 'Hitting Team (For)' AND is_goal = 1 THEN 1 ELSE 0 END) AS goals_for_30s,
                SUM(CASE WHEN outcome_side = 'Opponent (Against)' AND is_goal = 1 THEN 1 ELSE 0 END) AS goals_against_30s
            FROM read_csv_auto('{goal_30s_path}')
            GROUP BY 1, 2
        )
        SELECT 
            th.idx_season,
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
        JOIN team_games tg ON th.idx_season = tg.idx_season AND th.team = tg.team
        LEFT JOIN team_shots_30s ts ON th.idx_season = ts.idx_season AND th.team = ts.team
    """)

    # Combine individual seasons + ALL combined + LEAGUE_AVG
    df_team_kpi = con.execute("""
        WITH by_season AS (
            SELECT CAST(idx_season AS VARCHAR) AS season, * EXCLUDE (idx_season) FROM team_season_base
        ),
        league_by_season AS (
            SELECT 
                CAST(idx_season AS VARCHAR) AS season,
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
            FROM team_season_base
            GROUP BY idx_season
        ),
        all_seasons_team AS (
            SELECT 
                'ALL' AS season,
                team,
                SUM(games_played) AS games_played,
                SUM(total_hits) AS total_hits,
                ROUND(SUM(total_hits) * 1.0 / SUM(games_played), 1) AS hits_per_game,
                ROUND(AVG(ozone_pct), 1) AS ozone_pct,
                ROUND(AVG(dzone_pct), 1) AS dzone_pct,
                ROUND(AVG(nzone_pct), 1) AS nzone_pct,
                ROUND(AVG(puck_win_pct), 1) AS puck_win_pct,
                ROUND(AVG(step1_shot_for_pct), 1) AS step1_shot_for_pct,
                ROUND(AVG(step1_shot_against_pct), 1) AS step1_shot_against_pct,
                SUM(shots_for_30s) AS shots_for_30s,
                SUM(shots_against_30s) AS shots_against_30s,
                ROUND(SUM(shots_for_30s) * 100.0 / SUM(total_hits), 2) AS shots_for_per_100,
                ROUND(SUM(shots_against_30s) * 100.0 / SUM(total_hits), 2) AS shots_against_per_100,
                ROUND((SUM(shots_for_30s) - SUM(shots_against_30s)) * 100.0 / SUM(total_hits), 2) AS net_shots_per_100,
                SUM(goals_for_30s) AS goals_for_30s,
                SUM(goals_against_30s) AS goals_against_30s,
                ROUND((SUM(goals_for_30s) - SUM(goals_against_30s)) * 100.0 / SUM(total_hits), 3) AS net_goals_per_100
            FROM team_season_base
            GROUP BY team
        ),
        league_all AS (
            SELECT 
                'ALL' AS season,
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
            FROM all_seasons_team
        )
        SELECT * FROM by_season
        UNION ALL
        SELECT * FROM league_by_season
        UNION ALL
        SELECT * FROM all_seasons_team
        UNION ALL
        SELECT * FROM league_all
        ORDER BY season, team
    """).df()

    team_kpi_path = os.path.join(output_dir, "team_season_hit_kpi.csv")
    df_team_kpi.to_csv(team_kpi_path, index=False)
    print(f"Saved team & season KPI table to {team_kpi_path} ({len(df_team_kpi):,} records)")

    # 8. Compute Team Spatial Hitting Density (for Team Heatmaps)
    print("Computing team spatial hitting density...")
    df_team_spatial = con.execute("""
        WITH team_season_bins AS (
            SELECT 
                CAST(idx_season AS VARCHAR) AS season,
                event_team AS team,
                x_bin,
                y_bin,
                COUNT(*) AS hit_count
            FROM hit_sequences
            WHERE event_team IS NOT NULL AND x_bin IS NOT NULL AND y_bin IS NOT NULL
              AND ABS(x_bin) <= 100 AND ABS(y_bin) <= 40
            GROUP BY 1, 2, 3, 4
        ),
        team_season_totals AS (
            SELECT season, team, SUM(hit_count) AS total_hits
            FROM team_season_bins
            GROUP BY 1, 2
        ),
        all_season_bins AS (
            SELECT 
                'ALL' AS season,
                team,
                x_bin,
                y_bin,
                SUM(hit_count) AS hit_count
            FROM team_season_bins
            GROUP BY 1, 2, 3, 4
        ),
        all_season_totals AS (
            SELECT season, team, SUM(hit_count) AS total_hits
            FROM all_season_bins
            GROUP BY 1, 2
        ),
        league_bins AS (
            SELECT 
                season,
                'LEAGUE_AVG' AS team,
                x_bin,
                y_bin,
                ROUND(AVG(hit_count), 1) AS hit_count
            FROM team_season_bins
            GROUP BY 1, 2, 3, 4
            UNION ALL
            SELECT 
                season,
                'LEAGUE_AVG' AS team,
                x_bin,
                y_bin,
                ROUND(AVG(hit_count), 1) AS hit_count
            FROM all_season_bins
            GROUP BY 1, 2, 3, 4
        ),
        league_totals AS (
            SELECT season, team, SUM(hit_count) AS total_hits
            FROM league_bins
            GROUP BY 1, 2
        ),
        combined AS (
            SELECT b.season, b.team, b.x_bin, b.y_bin, b.hit_count, ROUND(b.hit_count * 100.0 / t.total_hits, 2) AS density_pct
            FROM team_season_bins b JOIN team_season_totals t ON b.season = t.season AND b.team = t.team
            UNION ALL
            SELECT b.season, b.team, b.x_bin, b.y_bin, b.hit_count, ROUND(b.hit_count * 100.0 / t.total_hits, 2) AS density_pct
            FROM all_season_bins b JOIN all_season_totals t ON b.season = t.season AND b.team = t.team
            UNION ALL
            SELECT b.season, b.team, b.x_bin, b.y_bin, b.hit_count, ROUND(b.hit_count * 100.0 / t.total_hits, 2) AS density_pct
            FROM league_bins b JOIN league_totals t ON b.season = t.season AND b.team = t.team
        )
        SELECT * FROM combined
        ORDER BY season, team, hit_count DESC
    """).df()

    team_spatial_path = os.path.join(output_dir, "team_season_spatial_hits.csv")
    df_team_spatial.to_csv(team_spatial_path, index=False)
    print(f"Saved team spatial hits to {team_spatial_path} ({len(df_team_spatial):,} records)")

    print("\nData processing complete! All analytical artifacts generated.")

if __name__ == "__main__":
    run_pipeline()
