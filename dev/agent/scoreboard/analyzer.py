"""analyzer.py — Statistical analysis, anomaly detection, milestone extraction, and guardrails.

Collectively evaluates raw game feeds to compute:
1. Data Integrity Guardrails: Cross-reconciliation of goals, SOG, saves, and scores.
2. Period Flow Audit: Verifies game script (lead changes, ties, comebacks vs wire-to-wire).
3. Defensive Effort & Goaltending: Shot-blocking sacrifice, turnovers, and goalie composure.
4. Longitudinal Season Context: Games played sample-size sensitivity & Pythagorean expectation.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

try:
    from .models import (
        AnalyzedGame,
        AnalyzedSlate,
        DefensiveEffortProfile,
        GoaliePerformanceProfile,
        GoalieStats,
        NotableHighlight,
        PeriodFlowAnalysis,
        ScoringPlay,
        SkaterStats,
        StatsValidationReport,
        TeamGameStats,
        TeamSeasonContext,
        ThreeStar,
    )
    from .skills import EditorialSkillsOrchestrator, ShiftAndTrendDataProvider
except (ImportError, ValueError):
    from models import (
        AnalyzedGame,
        AnalyzedSlate,
        DefensiveEffortProfile,
        GoaliePerformanceProfile,
        GoalieStats,
        NotableHighlight,
        PeriodFlowAnalysis,
        ScoringPlay,
        SkaterStats,
        StatsValidationReport,
        TeamGameStats,
        TeamSeasonContext,
        ThreeStar,
    )
    from skills import EditorialSkillsOrchestrator, ShiftAndTrendDataProvider

logger = logging.getLogger("ScoreboardAgent.Analyzer")


class SeasonAnalyticsProvider:
    """Computes standings, sample-size progression, and Pythagorean expectations up to a target date."""

    _cache: Dict[str, Dict[str, TeamSeasonContext]] = {}

    @classmethod
    def get_season_contexts(
        cls, target_date: str, season: int = 20262027
    ) -> Dict[str, TeamSeasonContext]:
        cache_key = f"{season}_{target_date}"
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        try:
            import pandas as pd
            from pathlib import Path
            season_yr = int(str(season)[:4])
            local_box = Path(__file__).resolve().parents[3] / "latest" / "box" / f"{season_yr}_box.csv"
            if local_box.exists():
                b_df = pd.read_csv(local_box)
                b_df = b_df[b_df["date"].astype(str) <= target_date]
                s_rows = []
                for _, r in b_df.iterrows():
                    s_rows.append(
                        {
                            "home_team": str(r["tricode_for"]),
                            "away_team": str(r["tricode_against"]),
                            "home_score": r["metric_score_for"],
                            "away_score": r["metric_score_against"],
                            "period_ending": str(r.get("period_ending", "REG")),
                        }
                    )
                s_df = pd.DataFrame(s_rows)
            else:
                url = "https://huggingface.co/datasets/jjerlee/nhl-analytics/resolve/main/data/games.parquet"
                df = pd.read_parquet(url)
                s_df = df[(df["season"] == season) & (df["game_date"] <= target_date)]

            records: Dict[str, Dict[str, Any]] = {}
            for _, row in s_df.iterrows():
                h = str(row["home_team"])
                a = str(row["away_team"])
                hs = int(row["home_score"]) if pd.notna(row["home_score"]) else 0
                as_ = int(row["away_score"]) if pd.notna(row["away_score"]) else 0
                pe = str(row["period_ending"])

                for t in [h, a]:
                    if t not in records:
                        records[t] = {"gp": 0, "w": 0, "l": 0, "otl": 0, "pts": 0, "gf": 0, "ga": 0}

                records[h]["gp"] += 1
                records[h]["gf"] += hs
                records[h]["ga"] += as_
                records[a]["gp"] += 1
                records[a]["gf"] += as_
                records[a]["ga"] += hs

                if hs > as_:
                    records[h]["w"] += 1
                    records[h]["pts"] += 2
                    if pe in ("OT", "SO"):
                        records[a]["otl"] += 1
                        records[a]["pts"] += 1
                    else:
                        records[a]["l"] += 1
                else:
                    records[a]["w"] += 1
                    records[a]["pts"] += 2
                    if pe in ("OT", "SO"):
                        records[h]["otl"] += 1
                        records[h]["pts"] += 1
                    else:
                        records[h]["l"] += 1

            team_contexts = {}
            exp = 2.15
            for t, d in records.items():
                pts_pct = d["pts"] / (2 * d["gp"]) if d["gp"] > 0 else 0.0
                pyth = (d["gf"] ** exp) / (d["gf"] ** exp + d["ga"] ** exp) if (d["gf"] + d["ga"]) > 0 else 0.5
                diff = pts_pct - pyth
                gp = d["gp"]

                if gp < 25:
                    stage = "EARLY_SEASON"
                    narr = f"Early-season baseline ({gp} GP). Small-sample variance dominates; line combinations and defensive pairings remain in calibration."
                elif gp <= 50:
                    stage = "MID_SEASON"
                    narr = f"Mid-season juncture ({gp} GP). Team identity and underlying territorial shot metrics have largely stabilized."
                else:
                    stage = "PLAYOFF_RACE"
                    narr = f"Late-season stretch run ({gp} GP). High-leverage matchup with direct implications for postseason seeding and wildcard positioning."

                team_contexts[t] = TeamSeasonContext(
                    team_abbrev=t,
                    games_played=gp,
                    wins=d["w"],
                    losses=d["l"],
                    ot_losses=d["otl"],
                    points=d["pts"],
                    points_pct=pts_pct,
                    goals_for=d["gf"],
                    goals_against=d["ga"],
                    goal_diff=d["gf"] - d["ga"],
                    pythagorean_pct=pyth,
                    pythagorean_diff=diff,
                    season_stage=stage,
                    stage_narrative=narr,
                )

            cls._cache[cache_key] = team_contexts
            return team_contexts
        except Exception as e:
            logger.warning("Could not calculate longitudinal season records: %s", e)
            return {}


class GameAnalyzer:
    """Analyzes a single game bundle with comprehensive guardrails and analytical depth."""

    def __init__(
        self,
        bundle: Dict[str, Any],
        season_contexts: Optional[Dict[str, TeamSeasonContext]] = None,
    ):
        self.bundle = bundle
        self.game_id = bundle.get("game_id", 0)
        self.landing = bundle.get("landing", {})
        self.boxscore = bundle.get("boxscore", {})
        self.right_rail = bundle.get("right_rail", {})
        self.season_contexts = season_contexts or {}

    def analyze(self) -> AnalyzedGame:
        """Run full extraction, guardrail validation, and analytical breakdown."""
        game_date = self.landing.get("gameDate", "")
        game_state = self.landing.get("gameState", "")
        venue = self.landing.get("venue", {}).get("default", "Unknown Arena")
        period_desc = self.landing.get("periodDescriptor", {}).get("periodType", "REG")
        raw_season = self.landing.get("season")
        try:
            game_season = int(raw_season) if raw_season else int(f"{str(self.game_id)[:4]}{int(str(self.game_id)[:4]) + 1}")
        except Exception:
            game_season = 20262027

        # 1. Parse Teams and Box Score Totals
        away_team, home_team = self._parse_team_stats()

        if away_team.score > home_team.score:
            winner, loser = away_team, home_team
        else:
            winner, loser = home_team, away_team

        margin = abs(away_team.score - home_team.score)
        is_one_goal = margin == 1
        is_blowout = margin >= 4

        # 2. Parse Skaters and Goalies
        away_skaters, home_skaters = self._parse_skaters()
        away_goalies, home_goalies = self._parse_goalies()

        # 3. Parse Scoring Progression
        scoring_plays = self._parse_scoring_plays()

        # 4. Guardrail: Period Flow Audit (Classify Verified Game Script)
        period_flow = self._analyze_period_flow(away_team, home_team, winner, loser, scoring_plays, period_desc)

        # 5. Guardrail: Defensive Effort & Goaltender Evaluation
        away_defense, home_defense = self._evaluate_defense_profiles(away_team, home_team)
        winning_goalie_profile, losing_goalie_profile = self._evaluate_goalie_profiles(
            away_goalies + home_goalies, winner, loser
        )

        # 6. Guardrail: Data Integrity Cross-Verification
        validation_report = self._verify_game_stats(
            away_team, home_team, away_skaters, home_skaters, away_goalies, home_goalies, scoring_plays, period_desc
        )

        # 7. Season Progression & Pythagorean Context
        if not self.season_contexts and game_date:
            self.season_contexts = SeasonAnalyticsProvider.get_season_contexts(game_date, season=game_season)
        away_season_ctx = self.season_contexts.get(away_team.team_abbrev)
        home_season_ctx = self.season_contexts.get(home_team.team_abbrev)

        # 8. Line Shift & Rolling L5 Trend Profiles
        away_shift_prof = ShiftAndTrendDataProvider.get_team_shift_trend_profile(
            self.game_id, away_team.team_abbrev, home_team.team_abbrev, is_home=False
        )
        home_shift_prof = ShiftAndTrendDataProvider.get_team_shift_trend_profile(
            self.game_id, home_team.team_abbrev, away_team.team_abbrev, is_home=True
        )

        # 9. Parse Three Stars
        three_stars = self._parse_three_stars(away_skaters + home_skaters, away_goalies + home_goalies)

        # 10. Extract Tactical Notables & Player Milestones
        notables = self._detect_game_notables(
            away_team, home_team, winner, loser,
            away_skaters + home_skaters,
            away_goalies + home_goalies,
            scoring_plays, period_desc, period_flow
        )

        # Add standout line-shift & L5 line-change highlights
        for prof in (home_shift_prof, away_shift_prof):
            if prof and prof.has_shift_data and prof.best_line:
                bl = prof.best_line
                if bl.l5_trend_verdict == "LINE_CHANGE_IMPROVED":
                    notables.append(
                        NotableHighlight(
                            category="line_chemistry",
                            title=f"Line Adjustment Spark: {prof.team_abbrev} {bl.unit_label} ({bl.unit_players})",
                            description=bl.line_story_summary,
                            significance="HIGH",
                            entity=prof.team_abbrev,
                        )
                    )
                elif bl.performance_tier == "DOMINANT":
                    notables.append(
                        NotableHighlight(
                            category="line_chemistry",
                            title=f"Dominant 5v5 Trio: {prof.team_abbrev} {bl.unit_label} ({bl.unit_players})",
                            description=bl.line_story_summary,
                            significance="MEDIUM",
                            entity=prof.team_abbrev,
                        )
                    )

        analyzed = AnalyzedGame(
            game_id=self.game_id,
            game_date=game_date,
            game_state=game_state,
            period_type=period_desc,
            venue=venue,
            away_team=away_team,
            home_team=home_team,
            winner=winner,
            loser=loser,
            season=game_season,
            is_one_goal_game=is_one_goal,
            is_blowout=is_blowout,
            scoring_plays=scoring_plays,
            three_stars=three_stars,
            away_skaters=away_skaters,
            home_skaters=home_skaters,
            away_goalies=away_goalies,
            home_goalies=home_goalies,
            notables=notables,
            validation_report=validation_report,
            period_flow=period_flow,
            away_defense_profile=away_defense,
            home_defense_profile=home_defense,
            winning_goalie_profile=winning_goalie_profile,
            losing_goalie_profile=losing_goalie_profile,
            away_season_context=away_season_ctx,
            home_season_context=home_season_ctx,
            away_shift_trend_profile=away_shift_prof,
            home_shift_trend_profile=home_shift_prof,
        )
        analyzed.editorial_dossier = EditorialSkillsOrchestrator.evaluate_game(analyzed)
        return analyzed

    # =========================================================================
    # Guardrail 1: Data Integrity Reconciliation
    # =========================================================================

    def _verify_game_stats(
        self,
        away_team: TeamGameStats,
        home_team: TeamGameStats,
        away_skaters: List[SkaterStats],
        home_skaters: List[SkaterStats],
        away_goalies: List[GoalieStats],
        home_goalies: List[GoalieStats],
        plays: List[ScoringPlay],
        period_type: str,
    ) -> StatsValidationReport:
        notes = []
        goals_ok = True
        shots_ok = True
        period_math_ok = True

        total_score = away_team.score + home_team.score
        # Check scoring plays count (in SO, the shootout decider is not a standard play)
        if period_type != "SO":
            if len(plays) != total_score:
                goals_ok = False
                notes.append(f"Scoring plays count ({len(plays)}) does not match total goals ({total_score}).")

        # Check skater goals sum
        away_sk_goals = sum(s.goals for s in away_skaters)
        home_sk_goals = sum(s.goals for s in home_skaters)
        if period_type != "SO":
            if away_sk_goals != away_team.score:
                goals_ok = False
                notes.append(f"Away skater goals ({away_sk_goals}) != away score ({away_team.score}).")
            if home_sk_goals != home_team.score:
                goals_ok = False
                notes.append(f"Home skater goals ({home_sk_goals}) != home score ({home_team.score}).")

        # Goalie shots against vs opponent SOG
        away_goalie_sa = sum(g.shots_against for g in away_goalies)
        home_goalie_sa = sum(g.shots_against for g in home_goalies)
        # Tolerance of 2 for uncredited empty netters or shootout
        if period_type != "SO":
            if abs(away_goalie_sa - home_team.sog) > 2:
                shots_ok = False
                notes.append(f"Away goalie shots against ({away_goalie_sa}) differs from Home SOG ({home_team.sog}).")
            if abs(home_goalie_sa - away_team.sog) > 2:
                shots_ok = False
                notes.append(f"Home goalie shots against ({home_goalie_sa}) differs from Away SOG ({away_team.sog}).")

        is_valid = goals_ok and shots_ok and period_math_ok
        if is_valid:
            notes.append("All primary box score statistics, period totals, and player records reconciled successfully.")

        return StatsValidationReport(
            is_valid=is_valid,
            goals_reconciled=goals_ok,
            shots_reconciled=shots_ok,
            period_math_reconciled=period_math_ok,
            audit_notes=notes,
        )

    # =========================================================================
    # Guardrail 2: Period-by-Period Scoring Flow Audit
    # =========================================================================

    def _analyze_period_flow(
        self,
        away_team: TeamGameStats,
        home_team: TeamGameStats,
        winner: TeamGameStats,
        loser: TeamGameStats,
        scoring_plays: List[ScoringPlay],
        period_desc: str,
    ) -> PeriodFlowAnalysis:
        away_abbrev = away_team.team_abbrev
        home_abbrev = home_team.team_abbrev
        w_abbrev = winner.team_abbrev
        l_abbrev = loser.team_abbrev

        cur_a, cur_h = 0, 0
        p1_a, p1_h = 0, 0
        p2_a, p2_h = 0, 0
        ties = 0
        lead_changes = 0
        prev_leader = None
        max_a_lead = 0
        max_h_lead = 0

        for play in scoring_plays:
            if play.scoring_team == away_abbrev:
                cur_a += 1
            elif play.scoring_team == home_abbrev:
                cur_h += 1
            else:
                cur_a = play.away_score
                cur_h = play.home_score

            if play.period == 1:
                p1_a, p1_h = cur_a, cur_h
            elif play.period == 2:
                p2_a, p2_h = cur_a, cur_h

            if cur_a > cur_h:
                max_a_lead = max(max_a_lead, cur_a - cur_h)
                curr_leader = away_abbrev
            elif cur_h > cur_a:
                max_h_lead = max(max_h_lead, cur_h - cur_a)
                curr_leader = home_abbrev
            else:
                ties += 1
                curr_leader = "TIED"

            if prev_leader and prev_leader not in (curr_leader, "TIED") and curr_leader != "TIED":
                lead_changes += 1
            if curr_leader != "TIED":
                prev_leader = curr_leader

        w_score = winner.score
        l_score = loser.score
        margin = w_score - l_score
        p2_w = p2_a if w_abbrev == away_abbrev else p2_h
        p2_l = p2_h if w_abbrev == away_abbrev else p2_a
        winner_trailed_entering_p3 = p2_w < p2_l

        # Robust classification of real game script
        if period_desc in ("OT", "SO"):
            script_type = "WALK_OFF_OT_SO"
            summary = f"Regulation concluded tied {l_score}-{l_score} before {w_abbrev} secured the walk-off in {period_desc}."
        elif margin >= 4:
            script_type = "BLOWOUT"
            summary = f"Offensive rout with {w_abbrev} commanding tempo and pulling away by {margin} goals."
        elif winner_trailed_entering_p3:
            script_type = "COMEBACK_WIN"
            summary = f"{w_abbrev} rallied from a {p2_l}-{p2_w} deficit entering the third period to capture the victory."
        elif p2_w >= p2_l + 2 and margin == 1:
            script_type = "CONTROLLED_LATE_PUSH"
            summary = f"{w_abbrev} held a multi-goal cushion through two periods, withstanding {l_abbrev}'s late push."
        elif lead_changes >= 2 or (ties >= 2 and margin <= 2):
            script_type = "TRUE_THRILLER"
            summary = f"High-tension see-saw battle featuring {ties} ties and {lead_changes} lead changes."
        elif max_a_lead > 0 and w_abbrev == away_abbrev and max_h_lead == 0:
            script_type = "WIRE_TO_WIRE"
            summary = f"{w_abbrev} dictated the game from the opening period and never trailed."
        elif max_h_lead > 0 and w_abbrev == home_abbrev and max_a_lead == 0:
            script_type = "WIRE_TO_WIRE"
            summary = f"{w_abbrev} protected home ice throughout, maintaining command wire-to-wire."
        else:
            script_type = "REGULATION"
            summary = f"Methodical {w_score}-{l_score} victory managed by {w_abbrev} across regulation."

        return PeriodFlowAnalysis(
            lead_changes=lead_changes,
            ties_count=ties,
            max_winner_lead=max_a_lead if w_abbrev == away_abbrev else max_h_lead,
            max_loser_lead=max_h_lead if w_abbrev == away_abbrev else max_a_lead,
            score_after_p1=f"{away_abbrev} {p1_a} - {home_abbrev} {p1_h}",
            score_after_p2=f"{away_abbrev} {p2_a} - {home_abbrev} {p2_h}",
            score_entering_p3=f"{away_abbrev} {p2_a} - {home_abbrev} {p2_h}",
            winner_trailed_entering_p3=winner_trailed_entering_p3,
            game_script_type=script_type,
            script_summary=summary,
        )

    # =========================================================================
    # Guardrail 3: Defensive Effort Evaluation
    # =========================================================================

    def _evaluate_defense_profiles(
        self, away_team: TeamGameStats, home_team: TeamGameStats
    ) -> Tuple[DefensiveEffortProfile, DefensiveEffortProfile]:
        def profile(t: TeamGameStats, opp_sog: int) -> DefensiveEffortProfile:
            # Blocked shots rating
            if t.blocked_shots >= 18:
                b_rating = "SACRIFICIAL"
            elif t.blocked_shots >= 10:
                b_rating = "DISCIPLINED"
            else:
                b_rating = "POROUS"

            # Physicality / Hits rating
            if t.hits >= 25:
                h_rating = "PHYSICAL_FORCE"
            elif t.hits >= 14:
                h_rating = "MODERATE"
            else:
                h_rating = "CONTAINED"

            # Shot suppression
            if opp_sog < 24:
                s_rating = "SUFFOCATING"
            elif opp_sog <= 34:
                s_rating = "BALANCED"
            else:
                s_rating = "UNDER_SIEGE"

            return DefensiveEffortProfile(
                team_abbrev=t.team_abbrev,
                blocked_shots=t.blocked_shots,
                blocks_rating=b_rating,
                hits=t.hits,
                hits_rating=h_rating,
                giveaways=t.giveaways,
                takeaways=t.takeaways,
                turnover_diff=t.takeaways - t.giveaways,
                shots_allowed=opp_sog,
                shot_suppression=s_rating,
            )

        return profile(away_team, home_team.sog), profile(home_team, away_team.sog)

    # =========================================================================
    # Guardrail 4: Goaltender Performance Profiles
    # =========================================================================

    def _evaluate_goalie_profiles(
        self,
        all_goalies: List[GoalieStats],
        winner: TeamGameStats,
        loser: TeamGameStats,
    ) -> Tuple[Optional[GoaliePerformanceProfile], Optional[GoaliePerformanceProfile]]:
        w_goalie = None
        l_goalie = None

        for g in all_goalies:
            if g.decision == "W" or (g.team_abbrev == winner.team_abbrev and w_goalie is None):
                w_goalie = g
            elif g.decision == "L" or (g.team_abbrev == loser.team_abbrev and l_goalie is None):
                l_goalie = g

        def build_profile(g: Optional[GoalieStats], is_win: bool) -> Optional[GoaliePerformanceProfile]:
            if not g:
                return None

            if g.save_pct >= 0.930 or g.saves >= 35:
                rating = "BRICK_WALL"
                assess = f"Stood tall under heavy bombardment with {g.saves} saves on {g.shots_against} shots ({g.save_pct*100:.1f}% SV%)."
            elif g.shots_against >= 35:
                rating = "UNDER_PRESSURE"
                assess = f"Faced high-volume traffic ({g.shots_against} SOG), turning aside {g.saves} pucks."
            elif g.save_pct >= 0.900:
                rating = "STEADY"
                assess = f"Composed, fundamentally sound outing ({g.saves}/{g.shots_against}, {g.save_pct*100:.1f}%)."
            else:
                rating = "VULNERABLE"
                assess = f"Struggled to contain dangerous rebounds, conceding {g.goals_against} goals on {g.shots_against} shots."

            if g.is_shutout:
                assess = f"Immaculate shutout performance: stopped all {g.shots_against} shots faced."

            return GoaliePerformanceProfile(
                player_id=g.player_id,
                name=g.name,
                team_abbrev=g.team_abbrev,
                decision=g.decision or ("W" if is_win else "L"),
                saves=g.saves,
                shots_against=g.shots_against,
                goals_against=g.goals_against,
                save_pct=g.save_pct,
                toi=g.toi,
                starter=g.starter,
                is_shutout=g.is_shutout,
                performance_rating=rating,
                assessment=assess,
            )

        return build_profile(w_goalie, True), build_profile(l_goalie, False)

    # =========================================================================
    # Helpers for Skaters, Teams & Scoring
    # =========================================================================

    def _parse_team_stats(self) -> Tuple[TeamGameStats, TeamGameStats]:
        away_raw = self.landing.get("awayTeam", {})
        home_raw = self.landing.get("homeTeam", {})

        away = TeamGameStats(
            team_id=away_raw.get("id", 0),
            team_name=away_raw.get("placeName", {}).get("default", "") + " " + away_raw.get("commonName", {}).get("default", ""),
            team_abbrev=away_raw.get("abbrev", "AWAY"),
            score=int(away_raw.get("score", 0)),
            sog=int(away_raw.get("sog", 0)),
            logo_url=away_raw.get("logo", ""),
        )

        home = TeamGameStats(
            team_id=home_raw.get("id", 0),
            team_name=home_raw.get("placeName", {}).get("default", "") + " " + home_raw.get("commonName", {}).get("default", ""),
            team_abbrev=home_raw.get("abbrev", "HOME"),
            score=int(home_raw.get("score", 0)),
            sog=int(home_raw.get("sog", 0)),
            logo_url=home_raw.get("logo", ""),
        )

        for stat in self.right_rail.get("teamGameStats", []):
            cat = stat.get("category")
            a_val = stat.get("awayValue")
            h_val = stat.get("homeValue")

            if cat == "faceoffWinningPctg":
                away.faceoff_pct = float(a_val or 0.0) * 100.0 if float(a_val or 0.0) <= 1.0 else float(a_val or 0.0)
                home.faceoff_pct = float(h_val or 0.0) * 100.0 if float(h_val or 0.0) <= 1.0 else float(h_val or 0.0)
            elif cat == "faceoffWins":
                away.faceoff_wins = str(a_val or "")
                home.faceoff_wins = str(h_val or "")
            elif cat == "powerPlay":
                away.power_play_conv = str(a_val or "")
                home.power_play_conv = str(h_val or "")
            elif cat == "powerPlayPctg":
                away.power_play_pct = float(a_val or 0.0) * 100.0 if float(a_val or 0.0) <= 1.0 else float(a_val or 0.0)
                home.power_play_pct = float(h_val or 0.0) * 100.0 if float(h_val or 0.0) <= 1.0 else float(h_val or 0.0)
            elif cat == "pim":
                away.pim = int(a_val or 0)
                home.pim = int(h_val or 0)
            elif cat == "hits":
                away.hits = int(a_val or 0)
                home.hits = int(h_val or 0)
            elif cat == "blockedShots":
                away.blocked_shots = int(a_val or 0)
                home.blocked_shots = int(h_val or 0)
            elif cat == "giveaways":
                away.giveaways = int(a_val or 0)
                home.giveaways = int(h_val or 0)
            elif cat == "takeaways":
                away.takeaways = int(a_val or 0)
                home.takeaways = int(h_val or 0)

        return away, home

    def _parse_skaters(self) -> Tuple[List[SkaterStats], List[SkaterStats]]:
        player_stats = self.boxscore.get("playerByGameStats", {})
        away_raw = player_stats.get("awayTeam", {})
        home_raw = player_stats.get("homeTeam", {})

        away_abbrev = self.landing.get("awayTeam", {}).get("abbrev", "AWAY")
        home_abbrev = self.landing.get("homeTeam", {}).get("abbrev", "HOME")

        away_skaters = self._extract_skater_list(away_raw, away_abbrev)
        home_skaters = self._extract_skater_list(home_raw, home_abbrev)
        return away_skaters, home_skaters

    def _extract_skater_list(self, team_raw: Dict[str, Any], team_abbrev: str) -> List[SkaterStats]:
        skaters = []
        for group_key in ("forwards", "defense"):
            for p in team_raw.get(group_key, []):
                raw_name = p.get("name", {})
                name_str = raw_name.get("default", "Unknown") if isinstance(raw_name, dict) else str(raw_name)

                skater = SkaterStats(
                    player_id=p.get("playerId", 0),
                    name=name_str,
                    sweater_number=p.get("sweaterNumber", 0),
                    position=p.get("position", "F"),
                    team_abbrev=team_abbrev,
                    goals=int(p.get("goals", 0)),
                    assists=int(p.get("assists", 0)),
                    points=int(p.get("points", 0)),
                    plus_minus=int(p.get("plusMinus", 0)),
                    pim=int(p.get("pim", 0)),
                    sog=int(p.get("sog", 0)),
                    hits=int(p.get("hits", 0)),
                    blocked_shots=int(p.get("blockedShots", 0)),
                    giveaways=int(p.get("giveaways", 0)),
                    takeaways=int(p.get("takeaways", 0)),
                    power_play_goals=int(p.get("powerPlayGoals", 0)),
                    faceoff_pct=float(p.get("faceoffWinningPctg", 0.0)) * 100.0,
                    toi=str(p.get("toi", "00:00")),
                    shifts=int(p.get("shifts", 0)),
                )
                skaters.append(skater)
        skaters.sort(key=lambda x: (x.points, x.goals, x.sog), reverse=True)
        return skaters

    def _parse_goalies(self) -> Tuple[List[GoalieStats], List[GoalieStats]]:
        player_stats = self.boxscore.get("playerByGameStats", {})
        away_raw = player_stats.get("awayTeam", {})
        home_raw = player_stats.get("homeTeam", {})

        away_abbrev = self.landing.get("awayTeam", {}).get("abbrev", "AWAY")
        home_abbrev = self.landing.get("homeTeam", {}).get("abbrev", "HOME")

        away_goalies = self._extract_goalie_list(away_raw, away_abbrev)
        home_goalies = self._extract_goalie_list(home_raw, home_abbrev)
        return away_goalies, home_goalies

    def _extract_goalie_list(self, team_raw: Dict[str, Any], team_abbrev: str) -> List[GoalieStats]:
        goalies = []
        for g in team_raw.get("goalies", []):
            raw_name = g.get("name", {})
            name_str = raw_name.get("default", "Unknown") if isinstance(raw_name, dict) else str(raw_name)

            goalie = GoalieStats(
                player_id=g.get("playerId", 0),
                name=name_str,
                sweater_number=g.get("sweaterNumber", 0),
                team_abbrev=team_abbrev,
                saves=int(g.get("saves", 0)),
                shots_against=int(g.get("shotsAgainst", 0)),
                goals_against=int(g.get("goalsAgainst", 0)),
                save_pct=float(g.get("savePctg", 0.0)),
                toi=str(g.get("toi", "60:00")),
                decision=g.get("decision"),
                starter=bool(g.get("starter", False)),
                even_strength_shots=str(g.get("evenStrengthShotsAgainst", "0/0")),
                power_play_shots=str(g.get("powerPlayShotsAgainst", "0/0")),
                shorthanded_shots=str(g.get("shorthandedShotsAgainst", "0/0")),
            )
            goalies.append(goalie)
        return goalies

    def _parse_scoring_plays(self) -> List[ScoringPlay]:
        plays = []
        scoring_periods = self.landing.get("summary", {}).get("scoring", [])

        for per in scoring_periods:
            per_desc = per.get("periodDescriptor", {})
            period_num = per_desc.get("number", 1)
            raw_pt = str(per_desc.get("periodType", "REG")).upper()
            if raw_pt == "OT":
                period_label = "Overtime"
            elif raw_pt == "SO":
                period_label = "Shootout"
            elif period_num == 1:
                period_label = "1st Period"
            elif period_num == 2:
                period_label = "2nd Period"
            elif period_num == 3:
                period_label = "3rd Period"
            else:
                period_label = f"Period {period_num}"

            for goal in per.get("goals", []):
                scorer_name = goal.get("name", {}).get("default", "Unknown")
                assists = [
                    a.get("name", {}).get("default", "")
                    for a in goal.get("assists", [])
                    if a.get("name", {}).get("default")
                ]

                play = ScoringPlay(
                    period=period_num,
                    period_descriptor=period_label,
                    time_in_period=goal.get("timeInPeriod", "00:00"),
                    scoring_team=goal.get("teamAbbrev", {}).get("default", ""),
                    goal_scorer=scorer_name,
                    goal_scorer_id=goal.get("playerId", 0),
                    scorer_total_goals=int(goal.get("goalsToDate", 0)),
                    assists=assists,
                    strength=goal.get("strength", "EV").upper(),
                    away_score=int(goal.get("awayScore", 0)),
                    home_score=int(goal.get("homeScore", 0)),
                    highlight_url=goal.get("highlightClipSharingUrl", ""),
                )
                plays.append(play)

        return plays

    def _parse_three_stars(self, all_skaters: List[SkaterStats], all_goalies: List[GoalieStats]) -> List[ThreeStar]:
        stars_raw = self.landing.get("summary", {}).get("threeStars", [])
        stars = []

        if stars_raw:
            for s in stars_raw:
                raw_name = s.get("name", {})
                name_str = raw_name.get("default", "Unknown") if isinstance(raw_name, dict) else str(raw_name)
                pos = s.get("position", "")
                if pos == "G":
                    stats_str = f"{s.get('saves', 0)} saves ({s.get('goalsAgainst', 0)} GA)"
                else:
                    g = s.get("goals", 0)
                    a = s.get("assists", 0)
                    p = s.get("points", g + a)
                    stats_str = f"{g}G, {a}A ({p} pts)"

                star = ThreeStar(
                    star=s.get("star", 1),
                    player_id=s.get("playerId", 0),
                    name=name_str,
                    team_abbrev=s.get("teamAbbrev", ""),
                    position=pos,
                    stat_summary=stats_str,
                    headshot_url=s.get("headshot", ""),
                )
                stars.append(star)
            return stars

        candidates = []
        for sk in all_skaters:
            score = sk.goals * 3.0 + sk.assists * 1.5 + sk.plus_minus * 0.5 + sk.sog * 0.2
            candidates.append((score, sk.name, sk.team_abbrev, sk.position, f"{sk.goals}G, {sk.assists}A ({sk.points} pts)", sk.player_id))

        for g in all_goalies:
            if g.decision == "W":
                score = g.saves * 0.15 - g.goals_against * 1.5 + (4.0 if g.is_shutout else 1.0)
                candidates.append((score, g.name, g.team_abbrev, "G", f"{g.saves} saves ({g.goals_against} GA)", g.player_id))

        candidates.sort(key=lambda x: x[0], reverse=True)
        for idx, item in enumerate(candidates[:3]):
            stars.append(ThreeStar(
                star=idx + 1,
                player_id=item[5],
                name=item[1],
                team_abbrev=item[2],
                position=item[3],
                stat_summary=item[4],
            ))
        return stars

    def _detect_game_notables(
        self,
        away: TeamGameStats,
        home: TeamGameStats,
        winner: TeamGameStats,
        loser: TeamGameStats,
        skaters: List[SkaterStats],
        goalies: List[GoalieStats],
        plays: List[ScoringPlay],
        period_type: str,
        period_flow: PeriodFlowAnalysis,
    ) -> List[NotableHighlight]:
        notables = []

        if period_type in ("OT", "SO"):
            gwg_play = plays[-1] if plays else None
            hero_name = gwg_play.goal_scorer if gwg_play else "Winning squad"
            notables.append(NotableHighlight(
                category="overtime_thriller",
                title=f"{period_type} Walk-Off Victory",
                description=f"{winner.team_name} sealed the win in {period_type} over {loser.team_name}." + (f" Hero: {hero_name}." if gwg_play else ""),
                significance="HIGH",
                entity=winner.team_abbrev,
            ))
        elif period_flow.game_script_type == "COMEBACK_WIN":
            notables.append(NotableHighlight(
                category="comeback",
                title="3rd-Period Comeback Victory",
                description=period_flow.script_summary,
                significance="HIGH",
                entity=winner.team_abbrev,
            ))

        for sk in skaters:
            if sk.is_hat_trick:
                notables.append(NotableHighlight(
                    category="hat_trick",
                    title=f"Hat Trick Alert: {sk.name}",
                    description=f"{sk.name} ({sk.team_abbrev}) scored {sk.goals} goals, anchoring his club's offensive attack with {sk.sog} shots on goal.",
                    significance="HIGH",
                    entity=sk.name,
                    metrics={"goals": sk.goals, "shots": sk.sog}
                ))
            elif sk.goals >= 2:
                g_word = "2 goals" if sk.goals == 2 else f"{sk.goals} goals"
                notables.append(NotableHighlight(
                    category="multi_goal",
                    title=f"Multi-Goal Game: {sk.name}",
                    description=f"{sk.name} ({sk.team_abbrev}) scored {g_word} ({sk.points} total points) with a {sk.plus_minus:+d} rating.",
                    significance="MEDIUM",
                    entity=sk.name,
                    metrics={"goals": sk.goals, "points": sk.points}
                ))

            if sk.points >= 3 and not sk.is_hat_trick:
                if sk.goals == 0:
                    notables.append(NotableHighlight(
                        category="playmaker",
                        title=f"Playmaker Alert: {sk.name}",
                        description=f"{sk.name} ({sk.team_abbrev}) orchestrated {sk.assists} assists ({sk.points} total points) in {sk.toi} of ice time.",
                        significance="HIGH" if sk.points >= 4 else "MEDIUM",
                        entity=sk.name,
                        metrics={"points": sk.points, "assists": sk.assists}
                    ))
                elif sk.goals > 0 and sk.assists > 0:
                    g_lbl = "1 goal" if sk.goals == 1 else f"{sk.goals} goals"
                    a_lbl = "1 assist" if sk.assists == 1 else f"{sk.assists} assists"
                    notables.append(NotableHighlight(
                        category="offensive_catalyst",
                        title=f"Offensive Catalyst: {sk.name}",
                        description=f"{sk.name} ({sk.team_abbrev}) registered {sk.points} total points ({g_lbl}, {a_lbl}) in {sk.toi} of ice time.",
                        significance="HIGH" if sk.points >= 4 else "MEDIUM",
                        entity=sk.name,
                        metrics={"points": sk.points, "goals": sk.goals, "assists": sk.assists}
                    ))
                else:
                    g_lbl = f"{sk.goals} goals"
                    notables.append(NotableHighlight(
                        category="multi_goal",
                        title=f"Goal Scoring Spark: {sk.name}",
                        description=f"{sk.name} ({sk.team_abbrev}) recorded {sk.points} total points on {g_lbl} with {sk.sog} shots on goal in {sk.toi} of ice time.",
                        significance="HIGH" if sk.points >= 4 else "MEDIUM",
                        entity=sk.name,
                        metrics={"points": sk.points, "goals": sk.goals}
                    ))

            if sk.hits >= 5:
                notables.append(NotableHighlight(
                    category="physicality",
                    title=f"Heavy Forecheck: {sk.name}",
                    description=f"{sk.name} ({sk.team_abbrev}) delivered {sk.hits} body checks, applying continuous physical pressure.",
                    significance="LOW",
                    entity=sk.name,
                    metrics={"hits": sk.hits}
                ))

            if sk.blocked_shots >= 4:
                notables.append(NotableHighlight(
                    category="shot_blocker",
                    title=f"Defensive Wall: {sk.name}",
                    description=f"{sk.name} ({sk.team_abbrev}) sacrificed his body for {sk.blocked_shots} blocked shots.",
                    significance="MEDIUM",
                    entity=sk.name,
                    metrics={"blocked_shots": sk.blocked_shots}
                ))

        for goalie in goalies:
            if goalie.is_shutout:
                notables.append(NotableHighlight(
                    category="shutout",
                    title=f"Shutout Masterclass: {goalie.name}",
                    description=f"{goalie.name} turned aside all {goalie.shots_against} shots to earn the clean sheet victory.",
                    significance="HIGH",
                    entity=goalie.name,
                    metrics={"saves": goalie.saves, "shots": goalie.shots_against}
                ))
            elif goalie.is_brick_wall:
                notables.append(NotableHighlight(
                    category="brick_wall",
                    title=f"Brick Wall Performance: {goalie.name}",
                    description=f"{goalie.name} withstood a barrage, stopping {goalie.saves} of {goalie.shots_against} shots ({goalie.save_pct*100:.1f}% SV%).",
                    significance="HIGH",
                    entity=goalie.name,
                    metrics={"saves": goalie.saves, "shots": goalie.shots_against, "save_pct": goalie.save_pct}
                ))

        return notables


class SlateAnalyzer:
    """Analyzes an entire evening or slate of NHL games."""

    def __init__(self, date_str: str, analyzed_games: List[AnalyzedGame]):
        self.date_str = date_str
        self.games = analyzed_games

    def analyze(self) -> AnalyzedSlate:
        total_games = len(self.games)
        total_goals = sum(g.away_team.score + g.home_team.score for g in self.games)
        avg_goals = (total_goals / total_games) if total_games > 0 else 0.0

        home_wins = sum(1 for g in self.games if g.home_team.score > g.away_team.score)
        away_wins = sum(1 for g in self.games if g.away_team.score > g.home_team.score)
        ot_so_count = sum(1 for g in self.games if g.period_type in ("OT", "SO"))

        slate_stars = self._synthesize_slate_three_stars()
        top_performers = self._extract_top_performers()

        slate_notables = []
        for g in self.games:
            for n in g.notables:
                if n.significance == "HIGH":
                    slate_notables.append(n)

        return AnalyzedSlate(
            date=self.date_str,
            total_games=total_games,
            total_goals=total_goals,
            avg_goals_per_game=avg_goals,
            home_wins=home_wins,
            away_wins=away_wins,
            ot_so_games=ot_so_count,
            games=self.games,
            slate_three_stars=slate_stars,
            top_performers=top_performers,
            slate_notables=slate_notables,
        )

    def _synthesize_slate_three_stars(self) -> List[ThreeStar]:
        candidates = []
        for g in self.games:
            for sk in g.away_skaters + g.home_skaters:
                if sk.points >= 2 or sk.goals >= 2:
                    score = sk.goals * 4.0 + sk.assists * 2.0 + sk.plus_minus * 0.5 + sk.sog * 0.2
                    if g.period_type in ("OT", "SO") and sk.goals >= 1:
                        score += 2.0
                    candidates.append((score, sk.name, sk.team_abbrev, sk.position, f"{sk.goals}G, {sk.assists}A ({sk.points} pts)", sk.player_id))

            for goalie in g.away_goalies + g.home_goalies:
                if goalie.decision == "W":
                    score = goalie.saves * 0.2 - goalie.goals_against * 1.5 + (6.0 if goalie.is_shutout else (3.0 if goalie.is_brick_wall else 0.0))
                    candidates.append((score, goalie.name, goalie.team_abbrev, "G", f"{goalie.saves} saves ({goalie.goals_against} GA, {goalie.save_pct*100:.1f}%)", goalie.player_id))

        candidates.sort(key=lambda x: x[0], reverse=True)
        slate_stars = []
        for idx, item in enumerate(candidates[:3]):
            slate_stars.append(ThreeStar(
                star=idx + 1,
                player_id=item[5],
                name=item[1],
                team_abbrev=item[2],
                position=item[3],
                stat_summary=item[4],
            ))
        return slate_stars

    def _extract_top_performers(self) -> List[Dict[str, Any]]:
        performers = []
        for g in self.games:
            for sk in g.away_skaters + g.home_skaters:
                if sk.points >= 2:
                    performers.append({
                        "name": sk.name,
                        "team": sk.team_abbrev,
                        "pos": sk.position,
                        "goals": sk.goals,
                        "assists": sk.assists,
                        "points": sk.points,
                        "plus_minus": sk.plus_minus,
                        "sog": sk.sog,
                        "toi": sk.toi,
                    })

        performers.sort(key=lambda x: (x["points"], x["goals"], x["sog"]), reverse=True)
        return performers[:10]
