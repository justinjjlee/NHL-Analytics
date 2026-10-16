"""
skills.py — Multi-Skillset Editorial Agent & Shift/L5 Trend Data Provider.

Implements five specialized analytical skills that evaluate each NHL game across
distinct dimensions and synthesize a tailored EditorialDossier:
  1. GameProgressionSkill — Period-by-period flow, lead swings & turning points.
  2. LineChemistryShiftSkill — On-ice 3F/2D shift combinations, sequential stint
     momentum, P3 bench management, line changes vs. L5 continuity, and whether
     line changes improved or regressed performance relative to the last 5 games.
  3. FormAndSeasonTrajectorySkill — Contrasts full-season KPIs (Corsi, Fenwick,
     Pythagorean, RPI) against rolling Last 5 Games (L5) team form and season stage.
  4. AdversityAndDefenseSkill — Home/Away splits, road-trip/back-to-back schedule
     fatigue, defensive sacrifice, and goaltending composure.
  5. ChiefEditorSynthesizer — Selects the dominant game archetype and compiles
     the 5-paragraph editorial dossier governed by `editorial_playbook.md`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from .models import (
        AnalyzedGame,
        EditorialDossier,
        EditorialSkillBrief,
        LineShiftUnitProfile,
        TeamShiftAndTrendProfile,
    )
except (ImportError, ValueError):
    from models import (
        AnalyzedGame,
        EditorialDossier,
        EditorialSkillBrief,
        LineShiftUnitProfile,
        TeamShiftAndTrendProfile,
    )

logger = logging.getLogger("ScoreboardAgent.Skills")


def load_editorial_playbook() -> str:
    """Load the markdown editorial playbook from disk."""
    playbook_path = Path(__file__).resolve().parent / "editorial_playbook.md"
    if playbook_path.exists():
        try:
            return playbook_path.read_text(encoding="utf-8").strip()
        except Exception as e:
            logger.warning("Failed to read editorial_playbook.md: %s", e)
    return ""


class ShiftAndTrendDataProvider:
    """Provides precomputed and on-demand Line Shift + L5 Rolling Trend profiles."""

    _line_trends_df: Optional[pd.DataFrame] = None
    _team_trends_df: Optional[pd.DataFrame] = None
    _loaded: bool = False

    @classmethod
    def _ensure_loaded(cls) -> None:
        if cls._loaded:
            return
        cls._loaded = True

        repo_root = Path(__file__).resolve().parents[3]
        local_line_csv = repo_root / "latest" / "action" / "shifts" / "game_line_trends_l5.csv"
        local_team_csv = repo_root / "latest" / "action" / "shifts" / "team_game_trends_l5.csv"

        if local_line_csv.exists():
            try:
                cls._line_trends_df = pd.read_csv(local_line_csv, low_memory=False)
            except Exception as e:
                logger.warning("Could not read local %s: %s", local_line_csv, e)

        if local_team_csv.exists():
            try:
                cls._team_trends_df = pd.read_csv(local_team_csv, low_memory=False)
            except Exception as e:
                logger.warning("Could not read local %s: %s", local_team_csv, e)

        # Remote fallback to Hugging Face dataset if local CSVs not present
        hf_base = "https://huggingface.co/datasets/jjerlee/nhl-analytics/resolve/main/measurement/action/shifts"
        if cls._line_trends_df is None:
            try:
                cls._line_trends_df = pd.read_parquet(f"{hf_base}/game_line_trends_l5.parquet")
            except Exception:
                cls._line_trends_df = pd.DataFrame()

        if cls._team_trends_df is None:
            try:
                cls._team_trends_df = pd.read_parquet(f"{hf_base}/team_game_trends_l5.parquet")
            except Exception:
                cls._team_trends_df = pd.DataFrame()

    @classmethod
    def get_team_shift_trend_profile(
        cls, game_id: int, team_abbrev: str, opponent_abbrev: str = "", is_home: bool = True
    ) -> TeamShiftAndTrendProfile:
        """Retrieve or construct TeamShiftAndTrendProfile for (game_id, team_abbrev)."""
        cls._ensure_loaded()
        gid = int(game_id)
        team = str(team_abbrev)

        profile = TeamShiftAndTrendProfile(
            team_abbrev=team,
            opponent_abbrev=opponent_abbrev,
            home_away="HOME" if is_home else "AWAY",
        )

        # 1. Populate team-level L5 & schedule metrics
        if cls._team_trends_df is not None and not cls._team_trends_df.empty:
            t_rows = cls._team_trends_df[
                (cls._team_trends_df["gameid"].astype(int) == gid)
                & (cls._team_trends_df["team"].astype(str) == team)
            ]
            if t_rows.empty:
                # Fallback to most recent game for this team if current game not yet in table
                t_hist = cls._team_trends_df[cls._team_trends_df["team"].astype(str) == team]
                if not t_hist.empty:
                    t_rows = t_hist.tail(1)

            if not t_rows.empty:
                r = t_rows.iloc[-1]
                profile.home_away = str(r.get("home_away", profile.home_away))
                profile.rest_days = int(r.get("rest_days", 3)) if pd.notna(r.get("rest_days")) else 3
                profile.is_back_to_back = bool(r.get("is_back_to_back", False))
                profile.opponent_rest_days = int(r.get("opponent_rest_days", 3)) if pd.notna(r.get("opponent_rest_days")) else 3
                profile.rest_advantage = int(r.get("rest_advantage", 0)) if pd.notna(r.get("rest_advantage")) else 0
                profile.consec_road_games = int(r.get("consec_road_games", 0)) if pd.notna(r.get("consec_road_games")) else 0
                profile.consec_home_games = int(r.get("consec_home_games", 0)) if pd.notna(r.get("consec_home_games")) else 0
                profile.season_gp = int(r.get("season_gp", 0)) if pd.notna(r.get("season_gp")) else 0
                profile.season_wp = float(r.get("season_wp", 50.0)) if pd.notna(r.get("season_wp")) else 50.0
                profile.season_cf_pct = float(r.get("season_cf_pct", 50.0)) if pd.notna(r.get("season_cf_pct")) else 50.0
                profile.season_ff_pct = float(r.get("season_ff_pct", 50.0)) if pd.notna(r.get("season_ff_pct")) else 50.0
                profile.season_pythagorean_pe = float(r.get("season_pythagorean_pe", 50.0)) if pd.notna(r.get("season_pythagorean_pe")) else 50.0
                profile.season_rpi = float(r.get("season_rpi", 50.0)) if pd.notna(r.get("season_rpi")) else 50.0
                profile.l5_games = int(r.get("l5_games", 0)) if pd.notna(r.get("l5_games")) else 0
                profile.l5_record = str(r.get("l5_record", "0-0"))
                profile.l5_gf_per_game = float(r.get("l5_gf_per_game", 0.0)) if pd.notna(r.get("l5_gf_per_game")) else 0.0
                profile.l5_ga_per_game = float(r.get("l5_ga_per_game", 0.0)) if pd.notna(r.get("l5_ga_per_game")) else 0.0
                profile.l5_goal_diff = int(r.get("l5_goal_diff", 0)) if pd.notna(r.get("l5_goal_diff")) else 0
                profile.l5_cf_pct = float(r.get("l5_cf_pct", 50.0)) if pd.notna(r.get("l5_cf_pct")) else 50.0
                profile.l5_ff_pct = float(r.get("l5_ff_pct", 50.0)) if pd.notna(r.get("l5_ff_pct")) else 50.0
                profile.l5_sf_pct = float(r.get("l5_sf_pct", 50.0)) if pd.notna(r.get("l5_sf_pct")) else 50.0
                profile.game_cf_pct = float(r.get("game_cf_pct", 50.0)) if pd.notna(r.get("game_cf_pct")) else 50.0
                profile.game_vs_l5_cf_delta = float(r.get("game_vs_l5_cf_delta", 0.0)) if pd.notna(r.get("game_vs_l5_cf_delta")) else 0.0
                profile.l5_form_status = str(r.get("l5_form_status", "BALANCED"))
                profile.has_shift_data = bool(r.get("has_shift_data", False))
                profile.line_continuity_pct = float(r.get("line_continuity_pct", 100.0)) if pd.notna(r.get("line_continuity_pct")) else 100.0
                profile.new_lines_deployed_count = int(r.get("new_lines_deployed_count", 0)) if pd.notna(r.get("new_lines_deployed_count")) else 0
                profile.in_game_line_blender = bool(r.get("in_game_line_blender", False))

        # 2. Populate unit-level 3F and 2D shift profiles
        if cls._line_trends_df is not None and not cls._line_trends_df.empty:
            u_rows = cls._line_trends_df[
                (cls._line_trends_df["gameid"].astype(int) == gid)
                & (cls._line_trends_df["team"].astype(str) == team)
            ]
            if not u_rows.empty:
                profile.has_shift_data = True
                for _, ur in u_rows.sort_values(["unit_type", "unit_rank"]).iterrows():
                    base_cf = (
                        float(ur["l5_exact_unit_cf_pct"])
                        if pd.notna(ur.get("l5_exact_unit_cf_pct")) and int(ur.get("l5_exact_unit_toi_sec", 0)) >= 180
                        else float(ur.get("l5_members_baseline_cf_pct", 50.0))
                    )
                    unit_prof = LineShiftUnitProfile(
                        team_abbrev=team,
                        unit_type=str(ur.get("unit_type", "3F")),
                        unit_rank=int(ur.get("unit_rank", 1)),
                        unit_label=str(ur.get("unit_label", "Line 1")),
                        unit_key=str(ur.get("unit_key", "")),
                        unit_players=str(ur.get("unit_players", "")),
                        toi_sec=int(ur.get("toi_sec", 0)),
                        toi_formatted=str(ur.get("toi_formatted", "0:00")),
                        toi_share_pct=float(ur.get("toi_share_pct", 0.0)),
                        p1_toi_sec=int(ur.get("p1_toi_sec", 0)),
                        p2_toi_sec=int(ur.get("p2_toi_sec", 0)),
                        p3_toi_sec=int(ur.get("p3_toi_sec", 0)),
                        p3_deployment_shift_pct=float(ur.get("p3_deployment_shift_pct", 0.0)),
                        stints_count=int(ur.get("stints_count", 0)),
                        avg_stint_sec=float(ur.get("avg_stint_sec", 0.0)),
                        cf=int(ur.get("cf", 0)),
                        ca=int(ur.get("ca", 0)),
                        cf_pct=float(ur.get("cf_pct", 50.0)),
                        ff=int(ur.get("ff", 0)),
                        fa=int(ur.get("fa", 0)),
                        ff_pct=float(ur.get("ff_pct", 50.0)),
                        sf=int(ur.get("sf", 0)),
                        sa=int(ur.get("sa", 0)),
                        sf_pct=float(ur.get("sf_pct", 50.0)),
                        gf=int(ur.get("gf", 0)),
                        ga=int(ur.get("ga", 0)),
                        goal_diff=int(ur.get("goal_diff", 0)),
                        ozs_pct=float(ur.get("ozs_pct", 50.0)),
                        dominant_stints=int(ur.get("dominant_stints", 0)),
                        pinned_stints=int(ur.get("pinned_stints", 0)),
                        max_consec_positive_stints=int(ur.get("max_consec_positive_stints", 0)),
                        p1_cf_pct=float(ur.get("p1_cf_pct", 50.0)),
                        p2_cf_pct=float(ur.get("p2_cf_pct", 50.0)),
                        p3_cf_pct=float(ur.get("p3_cf_pct", 50.0)),
                        line_impact_score=float(ur.get("line_impact_score", 50.0)),
                        performance_tier=str(ur.get("performance_tier", "STEADY")),
                        line_change_status=str(ur.get("line_change_status", "ESTABLISHED_UNIT")),
                        l5_exact_unit_games=int(ur.get("l5_exact_unit_games", 0)),
                        l5_baseline_cf_pct=round(base_cf, 1),
                        delta_cf_pct_vs_l5=float(ur.get("delta_cf_pct_vs_l5", 0.0)),
                        delta_impact_vs_l5=float(ur.get("delta_impact_vs_l5", 0.0)),
                        l5_trend_verdict=str(ur.get("l5_trend_verdict", "ESTABLISHED_STEADY")),
                        line_story_summary=str(ur.get("line_story_summary", "")),
                    )
                    if unit_prof.unit_type == "3F":
                        profile.forward_lines.append(unit_prof)
                    else:
                        profile.defense_pairs.append(unit_prof)

                if profile.forward_lines:
                    profile.top_line = profile.forward_lines[0]
                    reg_lines = [l for l in profile.forward_lines if l.toi_sec >= 180] or profile.forward_lines
                    profile.best_line = max(reg_lines, key=lambda x: (x.line_impact_score, x.goal_diff, x.cf_pct))
                    profile.worst_line = min(reg_lines, key=lambda x: (x.line_impact_score, x.goal_diff, x.cf_pct))

                if profile.defense_pairs:
                    profile.top_pair = profile.defense_pairs[0]

        return profile


class EditorialSkillsOrchestrator:
    """Coordinates the 5 specialist analytical skills to evaluate a game and build an EditorialDossier."""

    @classmethod
    def evaluate_game(cls, game: AnalyzedGame) -> EditorialDossier:
        b_prog = cls._run_game_progression_skill(game)
        b_shifts = cls._run_line_chemistry_shift_skill(game)
        b_form = cls._run_form_and_season_skill(game)
        b_adv = cls._run_adversity_and_defense_skill(game)

        briefs = {
            "progression": b_prog,
            "line_shifts": b_shifts,
            "form_and_season": b_form,
            "adversity_and_defense": b_adv,
        }

        archetype_code, archetype_label, rationale = cls._select_primary_archetype(game, briefs)
        synth_text = cls._compile_dossier_brief_text(game, archetype_code, archetype_label, rationale, briefs)

        return EditorialDossier(
            primary_archetype=archetype_code,
            archetype_label=archetype_label,
            archetype_rationale=rationale,
            skill_briefs=briefs,
            synthesized_brief_text=synth_text,
        )

    @classmethod
    def _run_game_progression_skill(cls, game: AnalyzedGame) -> EditorialSkillBrief:
        flow = game.period_flow
        w = game.winner
        l = game.loser
        plays = game.scoring_plays

        points: List[str] = []
        sig = 55.0

        if flow:
            points.append(
                f"Verified Game Script: {flow.game_script_type.replace('_', ' ')} — {flow.script_summary} "
                f"(P1: {flow.score_after_p1}, P2: {flow.score_after_p2}, Lead Changes: {flow.lead_changes}, Ties: {flow.ties_count})."
            )
            if flow.game_script_type in ("COMEBACK_WIN", "TRUE_THRILLER", "WALK_OFF_OT_SO"):
                sig = 82.0
            elif flow.game_script_type == "BLOWOUT":
                sig = 75.0

        if plays:
            first_g = plays[0]
            last_g = plays[-1]
            points.append(
                f"Opening Strike: {first_g.goal_scorer} ({first_g.scoring_team}) opened the scoring at "
                f"{first_g.time_in_period} of {first_g.period_descriptor} ({first_g.strength})."
            )
            if len(plays) > 1:
                points.append(
                    f"Closing Strike: {last_g.goal_scorer} ({last_g.scoring_team}) sealed the {w.score}-{l.score} "
                    f"result at {last_g.time_in_period} of {last_g.period_descriptor} ({last_g.strength})."
                )

        headline = (
            f"{w.team_abbrev} executed a {flow.game_script_type.replace('_', ' ').lower()} script ({w.score}-{l.score})"
            if flow
            else f"{w.team_abbrev} defeated {l.team_abbrev} {w.score}-{l.score}"
        )

        return EditorialSkillBrief(
            skill_name="GameProgressionSkill",
            headline_insight=headline,
            significance_score=sig,
            narrative_points=points,
        )

    @classmethod
    def _run_line_chemistry_shift_skill(cls, game: AnalyzedGame) -> EditorialSkillBrief:
        w = game.winner
        l = game.loser
        w_prof = (
            game.home_shift_trend_profile
            if w.team_abbrev == game.home_team.team_abbrev
            else game.away_shift_trend_profile
        )
        l_prof = (
            game.away_shift_trend_profile
            if w.team_abbrev == game.home_team.team_abbrev
            else game.home_shift_trend_profile
        )

        points: List[str] = []
        sig = 50.0
        headline = "Balanced five-on-five line deployment across both benches"

        if w_prof and w_prof.has_shift_data:
            sig = 68.0
            points.append(
                f"{w.team_abbrev} Line Continuity & Adjustments: {w_prof.line_continuity_pct:.1f}% of forward TOI played by "
                f"established L5 trios ({w_prof.new_lines_deployed_count} new/shuffled combinations deployed; "
                f"mid-game bench blender: {'Yes' if w_prof.in_game_line_blender else 'No'})."
            )
            if w_prof.best_line:
                bl = w_prof.best_line
                points.append(f"{w.team_abbrev} Standout Forward Unit: {bl.line_story_summary}")
                if bl.l5_trend_verdict == "LINE_CHANGE_IMPROVED":
                    sig = 88.0
                    headline = f"{w.team_abbrev}'s newly assembled {bl.unit_label} ({bl.unit_players}) sparked the victory ({bl.cf_pct:.1f}% CF%, {bl.gf}-{bl.ga} GF-GA, {bl.delta_cf_pct_vs_l5:+.1f}% vs L5)"
                elif bl.performance_tier == "DOMINANT":
                    sig = max(sig, 80.0)
                    headline = f"{w.team_abbrev}'s {bl.unit_label} ({bl.unit_players}) dictated 5v5 play ({bl.toi_formatted} TOI, {bl.cf_pct:.1f}% CF%, {bl.gf}-{bl.ga} goals)"

            if w_prof.worst_line and (
                not w_prof.best_line or w_prof.worst_line.unit_key != w_prof.best_line.unit_key
            ):
                wl = w_prof.worst_line
                points.append(f"{w.team_abbrev} Secondary/Depth Line Note: {wl.line_story_summary}")

            if w_prof.top_pair:
                tp = w_prof.top_pair
                points.append(f"{w.team_abbrev} Top Defense Pairing: {tp.line_story_summary}")

        if l_prof and l_prof.has_shift_data:
            if l_prof.best_line:
                lbl = l_prof.best_line
                points.append(f"{l.team_abbrev} Bright Spot Line: {lbl.line_story_summary}")
            if l_prof.worst_line:
                lwl = l_prof.worst_line
                points.append(f"{l.team_abbrev} Vulnerable Line: {lwl.line_story_summary}")
                if lwl.l5_trend_verdict == "LINE_CHANGE_REGRESSED" or lwl.goal_diff <= -2:
                    sig = max(sig, 78.0)

        # Fallback if NHL API did not record shift charts for an early game
        if not points:
            w_skaters = game.home_skaters if w.team_abbrev == game.home_team.team_abbrev else game.away_skaters
            l_skaters = game.away_skaters if w.team_abbrev == game.home_team.team_abbrev else game.home_skaters
            top_w = sorted(w_skaters, key=lambda s: (s.points, s.sog, s.shifts), reverse=True)[:3]
            top_l = sorted(l_skaters, key=lambda s: (s.points, s.sog, s.shifts), reverse=True)[:2]
            if top_w:
                w_desc = ", ".join(
                    f"{s.name} ({s.goals}G-{s.assists}A, {s.sog} SOG, {s.toi} TOI across {s.shifts} shifts)"
                    for s in top_w
                )
                points.append(f"{w.team_abbrev} Skater Deployment Leaders: {w_desc}.")
            if top_l:
                l_desc = ", ".join(
                    f"{s.name} ({s.goals}G-{s.assists}A, {s.sog} SOG, {s.toi} TOI across {s.shifts} shifts)"
                    for s in top_l
                )
                points.append(f"{l.team_abbrev} Skater Deployment Leaders: {l_desc}.")

        return EditorialSkillBrief(
            skill_name="LineChemistryShiftSkill",
            headline_insight=headline,
            significance_score=sig,
            narrative_points=points,
        )

    @classmethod
    def _run_form_and_season_skill(cls, game: AnalyzedGame) -> EditorialSkillBrief:
        w = game.winner
        l = game.loser
        w_ctx = game.home_season_context if w.team_abbrev == game.home_team.team_abbrev else game.away_season_context
        l_ctx = game.away_season_context if w.team_abbrev == game.home_team.team_abbrev else game.home_season_context
        w_tr = (
            game.home_shift_trend_profile
            if w.team_abbrev == game.home_team.team_abbrev
            else game.away_shift_trend_profile
        )
        l_tr = (
            game.away_shift_trend_profile
            if w.team_abbrev == game.home_team.team_abbrev
            else game.home_shift_trend_profile
        )

        points: List[str] = []
        sig = 60.0

        if w_ctx and l_ctx:
            points.append(
                f"Season Stage ({w_ctx.season_stage.replace('_', ' ')}): {w_ctx.stage_narrative}"
            )
            points.append(
                f"{w.team_name} Full-Season Profile: {w_ctx.wins}-{w_ctx.losses}-{w_ctx.ot_losses} "
                f"({w_ctx.points} pts, {w_ctx.points_pct*100:.1f}% pts, Goal Diff {w_ctx.goal_diff:+d}, "
                f"Pythagorean Expectation {w_ctx.pythagorean_pct*100:.1f}%, Season Corsi {w_tr.season_cf_pct if w_tr else 50.0:.1f}%)."
            )
            points.append(
                f"{l.team_name} Full-Season Profile: {l_ctx.wins}-{l_ctx.losses}-{l_ctx.ot_losses} "
                f"({l_ctx.points} pts, {l_ctx.points_pct*100:.1f}% pts, Goal Diff {l_ctx.goal_diff:+d}, "
                f"Pythagorean Expectation {l_ctx.pythagorean_pct*100:.1f}%, Season Corsi {l_tr.season_cf_pct if l_tr else 50.0:.1f}%)."
            )

        if w_tr and w_tr.l5_games > 0:
            points.append(
                f"{w.team_abbrev} Past 5 Games (L5) Rolling Trend ({w_tr.l5_form_status.replace('_', ' ')}): "
                f"{w_tr.l5_record} record over prior {w_tr.l5_games} games ({w_tr.l5_gf_per_game:.2f} GF/G, "
                f"{w_tr.l5_ga_per_game:.2f} GA/G, {w_tr.l5_cf_pct:.1f}% L5 CF%). "
                f"Tonight's {w_tr.game_cf_pct:.1f}% Corsi share was {w_tr.game_vs_l5_cf_delta:+.1f}% vs. their L5 average."
            )
            if abs(w_tr.game_vs_l5_cf_delta) >= 8.0 or w_tr.l5_form_status in ("HOT_SURGE", "COLD_SLUMP"):
                sig = 74.0

        if l_tr and l_tr.l5_games > 0:
            points.append(
                f"{l.team_abbrev} Past 5 Games (L5) Rolling Trend ({l_tr.l5_form_status.replace('_', ' ')}): "
                f"{l_tr.l5_record} record over prior {l_tr.l5_games} games ({l_tr.l5_gf_per_game:.2f} GF/G, "
                f"{l_tr.l5_ga_per_game:.2f} GA/G, {l_tr.l5_cf_pct:.1f}% L5 CF%). "
                f"Tonight's {l_tr.game_cf_pct:.1f}% Corsi share was {l_tr.game_vs_l5_cf_delta:+.1f}% vs. their L5 average."
            )

        headline = f"Season trajectory and L5 rolling form contrast between {w.team_abbrev} and {l.team_abbrev}"
        return EditorialSkillBrief(
            skill_name="FormAndSeasonTrajectorySkill",
            headline_insight=headline,
            significance_score=sig,
            narrative_points=points,
        )

    @classmethod
    def _run_adversity_and_defense_skill(cls, game: AnalyzedGame) -> EditorialSkillBrief:
        w = game.winner
        l = game.loser
        w_is_home = (w.team_abbrev == game.home_team.team_abbrev)
        w_tr = game.home_shift_trend_profile if w_is_home else game.away_shift_trend_profile
        l_tr = game.away_shift_trend_profile if w_is_home else game.home_shift_trend_profile
        w_def = game.home_defense_profile if w_is_home else game.away_defense_profile
        l_def = game.away_defense_profile if w_is_home else game.home_defense_profile
        w_g = game.winning_goalie_profile
        l_g = game.losing_goalie_profile

        points: List[str] = []
        sig = 58.0
        headline = f"{w.team_abbrev} defensive structure and netminding sealed the win"

        # Home/Away & Schedule Adversity
        venue_role = "at home (leveraging last-change matchup control)" if w_is_home else f"on the road at {game.venue}"
        adv_notes = []
        if w_tr:
            if not w_is_home and w_tr.consec_road_games >= 2:
                adv_notes.append(f"game #{w_tr.consec_road_games} of a road trip")
                sig = max(sig, 76.0)
            if w_tr.is_back_to_back:
                adv_notes.append("playing on the second night of a back-to-back (1 day rest)")
                sig = max(sig, 80.0)
            elif w_tr.rest_advantage < 0:
                adv_notes.append(f"overcoming a rest disadvantage ({w_tr.rest_days}d rest vs {l.team_abbrev}'s {w_tr.opponent_rest_days}d)")
                sig = max(sig, 72.0)

        adv_clause = f" while {', '.join(adv_notes)}" if adv_notes else ""
        points.append(f"Venue & Schedule Context: {w.team_name} won {venue_role}{adv_clause}.")

        if w_g:
            points.append(
                f"Winning Goaltender ({w_g.name}, {w.team_abbrev}): {w_g.saves}/{w_g.shots_against} saves "
                f"({w_g.save_pct*100:.1f}% SV%, {w_g.performance_rating}) — {w_g.assessment}"
            )
            if w_g.performance_rating in ("BRICK_WALL", "UNDER_SIEGE") or w_g.is_shutout:
                sig = max(sig, 86.0)
                headline = f"{w_g.name}'s {w_g.saves}-save clinic anchored {w.team_abbrev} under heavy pressure"

        if l_g:
            points.append(
                f"Opposing Goaltender ({l_g.name}, {l.team_abbrev}): {l_g.saves}/{l_g.shots_against} saves "
                f"({l_g.save_pct*100:.1f}% SV%, {l_g.performance_rating}) — {l_g.assessment}"
            )

        if w_def and l_def:
            points.append(
                f"Defensive Sacrifice & Physicality: {w.team_abbrev} recorded {w_def.blocked_shots} blocked shots "
                f"({w_def.blocks_rating}), {w_def.hits} hits, and a {w_def.turnover_diff:+d} turnover margin "
                f"(PP: {w.power_play_conv}); {l.team_abbrev} had {l_def.blocked_shots} blocks, {l_def.hits} hits, "
                f"and a {l_def.turnover_diff:+d} turnover margin (PP: {l.power_play_conv})."
            )

        return EditorialSkillBrief(
            skill_name="AdversityAndDefenseSkill",
            headline_insight=headline,
            significance_score=sig,
            narrative_points=points,
        )

    @classmethod
    def _select_primary_archetype(
        cls, game: AnalyzedGame, briefs: Dict[str, EditorialSkillBrief]
    ) -> Tuple[str, str, str]:
        w = game.winner
        l = game.loser
        w_is_home = (w.team_abbrev == game.home_team.team_abbrev)
        w_tr = game.home_shift_trend_profile if w_is_home else game.away_shift_trend_profile
        w_g = game.winning_goalie_profile
        flow = game.period_flow

        # Check for Goalie Heist Under Siege
        if w_g and (w_g.saves >= 32 or (w_tr and w_tr.game_cf_pct <= 44.5)) and w_g.save_pct >= 0.915:
            return (
                "GOALIE_HEIST_UNDER_SIEGE",
                "Goaltending Heist & Defensive Grit Under Siege",
                f"{w.team_name} absorbed heavy territorial pressure ({w_tr.game_cf_pct if w_tr else 45.0:.1f}% CF%) "
                f"but leaned on {w_g.name}'s {w_g.saves} saves and shot-blocking sacrifice to prevail.",
            )

        # Check for Tactical Line-Blender Spark
        if (
            w_tr
            and w_tr.best_line
            and w_tr.best_line.l5_trend_verdict == "LINE_CHANGE_IMPROVED"
            and (w_tr.best_line.goal_diff >= 1 or w_tr.best_line.cf_pct >= 58.0)
        ):
            bl = w_tr.best_line
            return (
                "TACTICAL_LINE_BLENDER_SPARK",
                "Tactical Line Adjustment & Chemistry Spark",
                f"Coach's decision to deploy {bl.unit_label} ({bl.unit_players}) paid immediate dividends "
                f"({bl.cf_pct:.1f}% CF%, {bl.gf}-{bl.ga} goals, {bl.delta_cf_pct_vs_l5:+.1f}% CF% over L5 baseline).",
            )

        # Check for Third-Period Comeback / Walk-off Drama
        if flow and flow.game_script_type in ("COMEBACK_WIN", "WALK_OFF_OT_SO", "TRUE_THRILLER"):
            return (
                "LATE_DRAMA_SURGE",
                "Late-Game Momentum Swing & Clutch Execution",
                f"{flow.script_summary} culminating in a {w.score}-{l.score} {game.period_type} finish.",
            )

        # Check for Road Adversity Overcome
        if w_tr and not w_is_home and (w_tr.is_back_to_back or w_tr.consec_road_games >= 2 or w_tr.rest_advantage < 0):
            return (
                "ROAD_SCHEDULE_GRIT",
                "Road Resilience Over Schedule Adversity",
                f"{w.team_name} overcame road schedule adversity at {game.venue} to secure two points.",
            )

        # Check for Top-Line Continuity Dominance
        if w_tr and w_tr.best_line and w_tr.best_line.performance_tier == "DOMINANT":
            bl = w_tr.best_line
            return (
                "LINE_CONTINUITY_DOMINANCE",
                "Five-on-Five Line Dominance",
                f"{w.team_name}'s {bl.unit_label} ({bl.unit_players}) controlled five-on-five play ({bl.cf_pct:.1f}% CF%, {bl.gf}-{bl.ga} goals).",
            )

        return (
            "TERRITORIAL_EXECUTION",
            "Balanced Territorial Execution",
            f"{w.team_name} combined structured five-on-five play with timely finishing to defeat {l.team_name} {w.score}-{l.score}.",
        )

    @classmethod
    def _compile_dossier_brief_text(
        cls,
        game: AnalyzedGame,
        archetype_code: str,
        archetype_label: str,
        rationale: str,
        briefs: Dict[str, EditorialSkillBrief],
    ) -> str:
        lines = [
            f"PRIMARY EDITORIAL ARCHETYPE: {archetype_label} ({archetype_code})",
            f"EDITORIAL ANGLE RATIONALE: {rationale}",
            "",
            "--- SKILL 1: GAME PROGRESSION & SEQUENTIAL MOMENTUM ---",
            *briefs["progression"].narrative_points,
            "",
            "--- SKILL 2: LINE SHIFT CHEMISTRY, BENCH MANAGEMENT & L5 LINE TRENDS ---",
            *briefs["line_shifts"].narrative_points,
            "",
            "--- SKILL 3: GOALTENDING, DEFENSIVE SACRIFICE & SCHEDULE ADVERSITY ---",
            *briefs["adversity_and_defense"].narrative_points,
            "",
            "--- SKILL 4: FULL-SEASON TRAJECTORY vs. PAST 5 GAMES (L5) ROLLING FORM ---",
            *briefs["form_and_season"].narrative_points,
        ]
        return "\n".join(lines)
