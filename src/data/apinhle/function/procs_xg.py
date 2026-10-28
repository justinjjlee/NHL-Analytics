"""
procs_xg.py — Canonical Expected Goals (xG) & Goals Saved Above Expected (GSAx)
Measurement and Estimation Module for NHL Analytics (Rinklytics).

Location: src/data/apinhle/function/procs_xg.py

Provides both:
1. Single-game NHL API JSON play-by-play evaluation (`evaluate_game_xg`) used by
   the Scoreboard Agent and live game report views.
2. Vectorized season-level and game-level estimation (`nhl_dataproc_pbp_xg` and
   `enrich_season_xg`) across `latest/play/{year}_playbyplay.csv`, enriching:
   - Team game & rolling statistics (`latest/box/{year}_box.csv`,
     `latest/box/{year}_box_team.csv`, `latest/box/{year}_box_gameStats.csv`)
   - Team season KPIs (`latest/team/season/{year}_team_season.csv`)
   - Player game & rolling + season-ending statistics
     (`latest/box/{year}_box_player.csv` and `latest/player/{year}_player.csv`)

Methodological Design & Dual-Stage Empirical Calibration:
- Unblocked shot attempts (Fenwick: `shot-on-goal`, `missed-shot`, `goal`) are
  projected into a direction-invariant attacking coordinate frame where the target
  goal center is at (+89.0, 0.0) feet.
- Because missed shots cannot score once off-target, a single probability cannot
  simultaneously be unbiased on all unblocked attempts (Fenwick, ~6.82% conversion)
  and on shots on goal (SOG, ~10.21% conversion). We therefore provide:
  * Fenwick Expected Goals (`xg` / `ixg`): P(Goal | Fenwick, x), calibrated across
    360,000+ NHL unblocked shot attempts (2023-2025) so League sum(xG) == sum(Goals).
  * Conditional On-Goal Expected Goals Against (`xga`): P(Goal | SOG, x), calibrated
    on non-empty-net Shots on Goal faced by goaltenders so League sum(GSAx) == 0.0.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Constants & Empirically Calibrated Model Parameters (2023-2025 NHL PBP)
# =============================================================================

NET_X = 89.0  # Goal line distance from center ice (feet)
RINK_LENGTH = 200.0
RINK_WIDTH = 85.0
HALF_WIDTH = 42.5
CREASE_CLAMP_FT = 10.5  # Effective minimum distance clamp for jam shots at the crease

# Stage 1: Fenwick Expected Goals (Unblocked Shot Attempts: SOG + Missed + Goal)
# Calibrated so E[sum(xG_F)] == sum(Goals) across High, Medium, and Low danger zones
BASE_LOG_ODDS_FENWICK = -1.25
BETA_DISTANCE_FENWICK = -0.0395
BETA_ANGLE_FENWICK = -0.0135
SLOT_CORRIDOR_FENWICK = 0.36  # Mid-slot release corridor (20 < d <= 35 ft, angle <= 50 deg)

# Stage 2: On-Goal Expected Goals Against (Non-Empty-Net Shots on Goal: SOG + Goal)
# Calibrated so E[sum(xGA_SOG)] == sum(Non-EN GA) -> League-wide sum(GSAx) == 0
BASE_LOG_ODDS_ON_GOAL = -0.82
BETA_DISTANCE_ON_GOAL = -0.0385
BETA_ANGLE_ON_GOAL = -0.0125
SLOT_CORRIDOR_ON_GOAL = 0.38

# Shared interaction & contextual log-odds adjustments
BETA_INTERACTION = -0.00012

SHOT_TYPE_WEIGHTS: Dict[str, float] = {
    "deflected": 0.22,
    "tip-in": 0.18,
    "bat": 0.25,
    "between-legs": 0.20,
    "poke": 0.15,
    "cradle": 0.15,
    "snap": 0.14,
    "slap": 0.06,
    "backhand": 0.05,
    "wrist": 0.00,
    "wrap-around": -0.45,
}

REBOUND_BOOST = 0.38       # Shot within 3.0s of a prior shot by same team
RUSH_BOOST = 0.22          # Transition shot within 4.0s of neutral/defensive zone event
POWER_PLAY_BOOST = 0.18    # Man-advantage lateral seam adjustment
SHORTHANDED_PENALTY = -0.15

# Backwards-compatible alias constants
BASE_LOG_ODDS = BASE_LOG_ODDS_FENWICK
BETA_DISTANCE = BETA_DISTANCE_FENWICK
BETA_ANGLE = BETA_ANGLE_FENWICK


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ShotAttempt:
    """Individual shot attempt with spatial telemetry and calculated expected goals."""
    event_id: int
    period: int
    time_in_period: str
    team_abbrev: str
    is_home: bool
    shooter_id: int
    shooter_name: str
    goalie_id: Optional[int]
    goalie_name: str
    x: float
    y: float
    distance: float
    angle: float
    shot_type: str
    danger_zone: str  # 'HIGH', 'MEDIUM', 'LOW'
    is_goal: bool
    is_on_goal: bool
    is_rebound: bool = False
    is_rush: bool = False
    is_power_play: bool = False
    is_empty_net: bool = False
    xg: float = 0.0
    xga_on_goal: float = 0.0


@dataclass
class GoalieXGSummary:
    """Goaltender quality evaluation isolating shot difficulty from defensive structure."""
    player_id: int
    name: str
    team_abbrev: str
    shots_against: int = 0
    goals_against: int = 0
    xga: float = 0.0         # Expected Goals Against (calibrated on SOG)
    gsax: float = 0.0        # Goals Saved Above Expected (xGA - GA)
    save_pct: float = 0.0    # Actual Save Percentage (0-100 scale)
    xsave_pct: float = 0.0   # Expected Save Percentage (1 - xGA/SA, 0-100 scale)
    dsave_pct: float = 0.0   # Delta Save % (save_pct - xsave_pct = gsax/SA)
    hd_shots_against: int = 0
    hd_goals_against: int = 0
    hd_save_pct: float = 0.0


@dataclass
class SkaterXGSummary:
    """Individual skater offensive generation and shot-quality breakdown."""
    player_id: int
    name: str
    team_abbrev: str
    shots: int = 0           # Unblocked shot attempts (Fenwick)
    sog: int = 0             # Shots on goal
    goals: int = 0
    ixg: float = 0.0         # Individual Expected Goals
    goals_above_expected: float = 0.0  # G - ixG
    hd_shots: int = 0        # High Danger Shots
    md_shots: int = 0        # Medium Danger Shots
    ld_shots: int = 0        # Low Danger Shots
    shooting_pct: float = 0.0


@dataclass
class GameXGReport:
    """Consolidated game-level expected goals analysis across teams, goalies, and skaters."""
    away_team_abbrev: str
    home_team_abbrev: str
    away_xg: float = 0.0
    home_xg: float = 0.0
    xg_differential: float = 0.0   # Home xG - Away xG
    away_xg_pct: float = 50.0      # Away xGF% (0-100)
    home_xg_pct: float = 50.0      # Home xGF% (0-100)
    away_shots: int = 0
    home_shots: int = 0
    away_goals: int = 0
    home_goals: int = 0
    period_xg: Dict[str, Dict[str, float]] = field(default_factory=dict)
    danger_zones: Dict[str, Dict[str, int]] = field(default_factory=dict)
    goalie_summaries: Dict[int, GoalieXGSummary] = field(default_factory=dict)
    skater_summaries: Dict[int, SkaterXGSummary] = field(default_factory=dict)
    all_shots: List[ShotAttempt] = field(default_factory=list)


# =============================================================================
# Geometric Transformations & Danger Zone Classification
# =============================================================================

def normalize_shot_geometry(
    x: float,
    y: float,
    zone_code: str = "O",
    attack_sign: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Computes Euclidean shot distance (feet) and shooting angle (degrees)
    relative to the center of the attacking goal mouth at (+89.0, 0.0).

    If `attack_sign` (+1 or -1) is known from period/team orientation,
    `x_att = x * attack_sign` maps all zones ('O', 'N', 'D') continuously onto
    [-100, +100]. Otherwise `zone_code` disambiguates offensive vs defensive end.
    """
    fx = float(x)
    fy = float(y)
    abs_x = abs(fx)
    abs_y = abs(fy)

    if attack_sign is not None and attack_sign != 0:
        x_att = fx * float(attack_sign)
        dx = abs(NET_X - x_att)
    else:
        z = str(zone_code or "O").upper().strip()
        if z == "D":
            dx = NET_X + abs_x
        elif z == "N":
            dx = NET_X + abs_x
        else:
            # In offensive zone ('O'), goal line is at |x| = 89.0
            dx = abs(NET_X - abs_x)

    dx_safe = max(dx, 0.5)
    distance = math.sqrt(dx**2 + abs_y**2)
    angle_rad = math.atan2(abs_y, dx_safe)
    angle_deg = math.degrees(angle_rad)

    return round(distance, 2), round(angle_deg, 2)


def classify_danger_zone(distance: float, angle: float) -> str:
    """
    Classifies a shot attempt into High, Medium, or Low Danger.
    - HIGH: Distance <= 20 ft and angle <= 45 deg (inner slot & crease).
    - MEDIUM: (20 < Distance <= 35 ft and angle <= 50 deg) OR (Distance <= 20 ft and angle > 45 deg).
    - LOW: Distance > 35 ft, perimeter point shots, or extreme wide-angle attempts.
    """
    if distance <= 20.0 and angle <= 45.0:
        return "HIGH"
    elif (distance <= 35.0 and angle <= 50.0) or (distance <= 20.0 and angle > 45.0):
        return "MEDIUM"
    else:
        return "LOW"


# =============================================================================
# Probability Calculation (Scalar & Vectorized)
# =============================================================================

def calculate_shot_xg(
    distance: float,
    angle: float,
    shot_type: str = "wrist",
    is_rebound: bool = False,
    is_rush: bool = False,
    is_power_play: bool = False,
    is_shorthanded: bool = False,
    is_empty_net: bool = False,
    on_goal_mode: bool = False,
) -> float:
    """
    Calculates calibrated Expected Goals probability for a single shot attempt.
    - When `on_goal_mode=False` (default): returns Fenwick xG P(Goal | Fenwick, x),
      calibrated across all unblocked shot attempts for team xG and skater ixG.
    - When `on_goal_mode=True`: returns On-Goal xGA P(Goal | SOG, x), calibrated
      for goaltender GSAx evaluation on shots on target.
    """
    if is_empty_net:
        if on_goal_mode:
            return 0.98
        if distance < 65.0:
            return 0.72
        elif distance < 115.0:
            return 0.55
        return 0.38

    d_eff = max(float(distance), CREASE_CLAMP_FT)
    ang = float(angle)

    if on_goal_mode:
        b0 = BASE_LOG_ODDS_ON_GOAL
        b_d = BETA_DISTANCE_ON_GOAL
        b_a = BETA_ANGLE_ON_GOAL
        slot_bonus = SLOT_CORRIDOR_ON_GOAL
    else:
        b0 = BASE_LOG_ODDS_FENWICK
        b_d = BETA_DISTANCE_FENWICK
        b_a = BETA_ANGLE_FENWICK
        slot_bonus = SLOT_CORRIDOR_FENWICK

    is_med = (
        ((distance <= 35.0 and ang <= 50.0) or (distance <= 20.0 and ang > 45.0))
        and not (distance <= 20.0 and ang <= 45.0)
    )

    z = (
        b0
        + (b_d * d_eff)
        + (b_a * ang)
        + (BETA_INTERACTION * d_eff * ang)
        + (slot_bonus if is_med else 0.0)
    )

    clean_type = str(shot_type or "wrist").strip().lower()
    z += SHOT_TYPE_WEIGHTS.get(clean_type, 0.0)

    if is_rebound:
        z += REBOUND_BOOST
    if is_rush:
        z += RUSH_BOOST
    if is_power_play:
        z += POWER_PLAY_BOOST
    if is_shorthanded:
        z += SHORTHANDED_PENALTY

    prob = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, z))))
    prob = max(0.001, min(0.85, prob))
    return round(prob, 4)


def calculate_shot_xg_vectorized(
    distance: np.ndarray,
    angle: np.ndarray,
    shot_type: np.ndarray,
    is_rebound: np.ndarray,
    is_rush: np.ndarray,
    is_empty_net: np.ndarray,
    on_goal_mode: bool = False,
) -> np.ndarray:
    """
    Vectorized Expected Goals probability calculation over NumPy arrays.
    """
    d_eff = np.maximum(distance, CREASE_CLAMP_FT)
    if on_goal_mode:
        b0 = BASE_LOG_ODDS_ON_GOAL
        b_d = BETA_DISTANCE_ON_GOAL
        b_a = BETA_ANGLE_ON_GOAL
        slot_bonus = SLOT_CORRIDOR_ON_GOAL
    else:
        b0 = BASE_LOG_ODDS_FENWICK
        b_d = BETA_DISTANCE_FENWICK
        b_a = BETA_ANGLE_FENWICK
        slot_bonus = SLOT_CORRIDOR_FENWICK

    is_high = (distance <= 20.0) & (angle <= 45.0)
    is_med = (((distance <= 35.0) & (angle <= 50.0)) | ((distance <= 20.0) & (angle > 45.0))) & (~is_high)
    slot_corridor = is_med.astype(float) * slot_bonus

    sw = np.array([SHOT_TYPE_WEIGHTS.get(str(s).lower().strip(), 0.0) for s in shot_type], dtype=float)
    z = (
        b0
        + b_d * d_eff
        + b_a * angle
        + BETA_INTERACTION * d_eff * angle
        + slot_corridor
        + sw
        + REBOUND_BOOST * is_rebound.astype(float)
        + RUSH_BOOST * is_rush.astype(float)
    )
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))
    p = np.clip(p, 0.001, 0.85)

    if on_goal_mode:
        p = np.where(is_empty_net, 0.98, p)
    else:
        en_p = np.where(distance < 65.0, 0.72, np.where(distance < 115.0, 0.55, 0.38))
        p = np.where(is_empty_net, en_p, p)

    return np.round(p, 4)


# =============================================================================
# Single-Game Play-by-Play JSON Evaluator (Scoreboard Agent & Live Recaps)
# =============================================================================

def evaluate_game_xg(pbp_data: Dict[str, Any]) -> GameXGReport:
    """
    Processes an NHL API play-by-play JSON payload to compute consolidated
    team xG, period xG, danger zone counts, skater ixG, and goalie GSAx.
    """
    if not pbp_data or not isinstance(pbp_data, dict):
        return GameXGReport(away_team_abbrev="AWAY", home_team_abbrev="HOME")

    away_team_info = pbp_data.get("awayTeam", {})
    home_team_info = pbp_data.get("homeTeam", {})
    away_abbrev = away_team_info.get("abbrev", "AWAY")
    home_abbrev = home_team_info.get("abbrev", "HOME")
    away_id = away_team_info.get("id")
    home_id = home_team_info.get("id")

    roster_map: Dict[int, Tuple[str, str]] = {}
    for spot in pbp_data.get("rosterSpots", []):
        pid = spot.get("playerId")
        if pid:
            first = spot.get("firstName", {}).get("default", "")
            last = spot.get("lastName", {}).get("default", "")
            name = f"{first} {last}".strip()
            tid = spot.get("teamId")
            tabbrev = home_abbrev if tid == home_id else away_abbrev
            roster_map[int(pid)] = (name, tabbrev)

    plays = pbp_data.get("plays", [])
    all_shots: List[ShotAttempt] = []

    period_xg: Dict[str, Dict[str, float]] = {}
    danger_counts = {
        away_abbrev: {"HIGH": 0, "MEDIUM": 0, "LOW": 0},
        home_abbrev: {"HIGH": 0, "MEDIUM": 0, "LOW": 0},
    }
    goalie_map: Dict[int, GoalieXGSummary] = {}
    skater_map: Dict[int, SkaterXGSummary] = {}

    prev_shot_period = -1
    prev_shot_seconds = -999.0
    prev_shot_team = None

    prev_ev_period = -1
    prev_ev_seconds = -999.0
    prev_ev_zone = None

    away_xg_total = 0.0
    home_xg_total = 0.0
    away_shots_total = 0
    home_shots_total = 0
    away_goals_total = 0
    home_goals_total = 0

    def parse_time_seconds(time_str: str) -> float:
        try:
            parts = str(time_str).split(":")
            return float(parts[0]) * 60.0 + float(parts[1])
        except Exception:
            return 0.0

    for p in plays:
        desc = p.get("typeDescKey")
        details = p.get("details", {}) or {}
        period_desc = p.get("periodDescriptor", {}) or {}
        p_num = int(period_desc.get("number", 1))
        time_str = p.get("timeInPeriod", "00:00")
        current_sec = parse_time_seconds(time_str)
        zone_code = details.get("zoneCode", "O")

        if desc not in ("shot-on-goal", "goal", "missed-shot"):
            prev_ev_period = p_num
            prev_ev_seconds = current_sec
            prev_ev_zone = zone_code
            continue

        x = details.get("xCoord")
        y = details.get("yCoord")
        if x is None or y is None:
            prev_ev_period = p_num
            prev_ev_seconds = current_sec
            prev_ev_zone = zone_code
            continue

        owner_team_id = details.get("eventOwnerTeamId")
        is_home_shot = (owner_team_id == home_id)
        team_abbrev = home_abbrev if is_home_shot else away_abbrev

        # Determine attacking sign if homeTeamDefendingSide is available
        def_side = p.get("homeTeamDefendingSide")
        attack_sign = None
        if def_side in ("left", "right"):
            # 'left' means home defends negative x (-89) and attacks positive x (+89)
            home_att = 1.0 if def_side == "left" else -1.0
            attack_sign = home_att if is_home_shot else -home_att

        dist, angle = normalize_shot_geometry(x, y, zone_code=zone_code, attack_sign=attack_sign)
        danger_zone = classify_danger_zone(dist, angle)

        shooter_id = details.get("shootingPlayerId") or details.get("scoringPlayerId") or 0
        shooter_name, _ = roster_map.get(int(shooter_id), (f"Player {shooter_id}", team_abbrev))

        goalie_id = details.get("goalieInNetId")
        opp_abbrev = away_abbrev if is_home_shot else home_abbrev
        goalie_name = ""
        if goalie_id:
            goalie_name, _ = roster_map.get(int(goalie_id), (f"Goalie {goalie_id}", opp_abbrev))

        p_label = f"P{p_num}" if p_num <= 3 else "OT"

        sit_code = str(p.get("situationCode", "1551"))
        is_pp = False
        is_sh = False
        is_en = (goalie_id is None)
        if len(sit_code) == 4 and sit_code.isdigit():
            away_g, away_sk, home_sk, home_g = int(sit_code[0]), int(sit_code[1]), int(sit_code[2]), int(sit_code[3])
            if is_home_shot:
                is_pp = home_sk > away_sk
                is_sh = home_sk < away_sk
                is_en = (away_g == 0)
            else:
                is_pp = away_sk > home_sk
                is_sh = away_sk < home_sk
                is_en = (home_g == 0)

        is_rebound = (
            p_num == prev_shot_period
            and 0.0 <= (current_sec - prev_shot_seconds) <= 3.0
            and prev_shot_team == team_abbrev
        )
        is_rush = (
            not is_rebound
            and p_num == prev_ev_period
            and 0.0 <= (current_sec - prev_ev_seconds) <= 4.0
            and prev_ev_zone in ("N", "D")
            and zone_code == "O"
        )

        stype = details.get("shotType", "wrist")
        xg = calculate_shot_xg(
            distance=dist,
            angle=angle,
            shot_type=stype,
            is_rebound=is_rebound,
            is_rush=is_rush,
            is_power_play=is_pp,
            is_shorthanded=is_sh,
            is_empty_net=is_en,
            on_goal_mode=False,
        )
        xga_sog = calculate_shot_xg(
            distance=dist,
            angle=angle,
            shot_type=stype,
            is_rebound=is_rebound,
            is_rush=is_rush,
            is_power_play=is_pp,
            is_shorthanded=is_sh,
            is_empty_net=is_en,
            on_goal_mode=True,
        )

        is_goal = (desc == "goal")
        is_on_goal = (desc in ("shot-on-goal", "goal"))

        prev_shot_period = p_num
        prev_shot_seconds = current_sec
        prev_shot_team = team_abbrev
        prev_ev_period = p_num
        prev_ev_seconds = current_sec
        prev_ev_zone = zone_code

        if p_label not in period_xg:
            period_xg[p_label] = {away_abbrev: 0.0, home_abbrev: 0.0}
        period_xg[p_label][team_abbrev] = round(period_xg[p_label][team_abbrev] + xg, 4)

        danger_counts[team_abbrev][danger_zone] += 1

        if is_home_shot:
            home_xg_total += xg
            if is_on_goal:
                home_shots_total += 1
                if is_goal:
                    home_goals_total += 1
        else:
            away_xg_total += xg
            if is_on_goal:
                away_shots_total += 1
                if is_goal:
                    away_goals_total += 1

        if shooter_id:
            sid = int(shooter_id)
            if sid not in skater_map:
                skater_map[sid] = SkaterXGSummary(
                    player_id=sid,
                    name=shooter_name,
                    team_abbrev=team_abbrev,
                )
            skater_map[sid].shots += 1
            if is_on_goal:
                skater_map[sid].sog += 1
            skater_map[sid].ixg = round(skater_map[sid].ixg + xg, 4)
            if is_goal:
                skater_map[sid].goals += 1
            skater_map[sid].goals_above_expected = round(skater_map[sid].goals - skater_map[sid].ixg, 2)
            if danger_zone == "HIGH":
                skater_map[sid].hd_shots += 1
            elif danger_zone == "MEDIUM":
                skater_map[sid].md_shots += 1
            else:
                skater_map[sid].ld_shots += 1
            if skater_map[sid].shots > 0:
                skater_map[sid].shooting_pct = round((skater_map[sid].goals / skater_map[sid].shots) * 100.0, 1)

        if goalie_id and is_on_goal and not is_en:
            gid = int(goalie_id)
            if gid not in goalie_map:
                goalie_map[gid] = GoalieXGSummary(
                    player_id=gid,
                    name=goalie_name,
                    team_abbrev=opp_abbrev,
                )
            goalie_map[gid].shots_against += 1
            goalie_map[gid].xga = round(goalie_map[gid].xga + xga_sog, 4)
            if is_goal:
                goalie_map[gid].goals_against += 1
            if danger_zone == "HIGH":
                goalie_map[gid].hd_shots_against += 1
                if is_goal:
                    goalie_map[gid].hd_goals_against += 1

        all_shots.append(
            ShotAttempt(
                event_id=int(p.get("eventId", 0)),
                period=p_num,
                time_in_period=time_str,
                team_abbrev=team_abbrev,
                is_home=is_home_shot,
                shooter_id=int(shooter_id),
                shooter_name=shooter_name,
                goalie_id=int(goalie_id) if goalie_id else None,
                goalie_name=goalie_name,
                x=float(x),
                y=float(y),
                distance=dist,
                angle=angle,
                shot_type=stype,
                danger_zone=danger_zone,
                is_goal=is_goal,
                is_on_goal=is_on_goal,
                is_rebound=is_rebound,
                is_rush=is_rush,
                is_power_play=is_pp,
                is_empty_net=is_en,
                xg=xg,
                xga_on_goal=xga_sog if is_on_goal else 0.0,
            )
        )

    for g in goalie_map.values():
        g.gsax = round(g.xga - g.goals_against, 2)
        if g.shots_against > 0:
            saves = g.shots_against - g.goals_against
            g.save_pct = round((saves / g.shots_against) * 100.0, 1)
            g.xsave_pct = round((1.0 - (g.xga / g.shots_against)) * 100.0, 1)
            g.dsave_pct = round(g.save_pct - g.xsave_pct, 1)
        if g.hd_shots_against > 0:
            hd_saves = g.hd_shots_against - g.hd_goals_against
            g.hd_save_pct = round((hd_saves / g.hd_shots_against) * 100.0, 1)

    tot_xg = away_xg_total + home_xg_total
    away_pct = round((away_xg_total / tot_xg * 100.0), 1) if tot_xg > 0 else 50.0
    home_pct = round((home_xg_total / tot_xg * 100.0), 1) if tot_xg > 0 else 50.0

    return GameXGReport(
        away_team_abbrev=away_abbrev,
        home_team_abbrev=home_abbrev,
        away_xg=round(away_xg_total, 2),
        home_xg=round(home_xg_total, 2),
        xg_differential=round(home_xg_total - away_xg_total, 2),
        away_xg_pct=away_pct,
        home_xg_pct=home_pct,
        away_shots=away_shots_total,
        home_shots=home_shots_total,
        away_goals=away_goals_total,
        home_goals=home_goals_total,
        period_xg=period_xg,
        danger_zones=danger_counts,
        goalie_summaries=goalie_map,
        skater_summaries=skater_map,
        all_shots=all_shots,
    )


# =============================================================================
# Vectorized Season Play-by-Play Processor (Team, Skater & Goalie Measurement)
# =============================================================================

def _parse_time_col(s: pd.Series) -> np.ndarray:
    """Vectorized MM:SS to elapsed seconds converter."""
    parts = s.fillna("00:00").astype(str).str.split(":", n=1, expand=True)
    if parts.shape[1] < 2:
        return np.zeros(len(s), dtype=float)
    mins = pd.to_numeric(parts[0], errors="coerce").fillna(0.0).values
    secs = pd.to_numeric(parts[1], errors="coerce").fillna(0.0).values
    return mins * 60.0 + secs


def nhl_dataproc_pbp_xg(
    df_pbp: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Vectorized play-by-play expected goals processor.
    Given a season's play-by-play DataFrame (`{year}_playbyplay.csv`), computes
    shot-level xG and aggregates into three measurement tables:
      1. `team_game_xg`: one row per (gameid, team_tri)
      2. `skater_game_xg`: one row per (gameid, playerId, team_tri)
      3. `goalie_game_xg`: one row per (gameid, playerId, team_tri)
    """
    if df_pbp is None or df_pbp.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = df_pbp.copy()
    df["sec"] = _parse_time_col(df["timeInPeriod"])
    df = df.sort_values(["gameid", "periodDescriptor.number", "sec", "eventId"]).reset_index(drop=True)

    # Track prior event across all plays for rush detection
    df["prev_game_all"] = df["gameid"].shift(1)
    df["prev_per_all"] = df["periodDescriptor.number"].shift(1)
    df["prev_sec_all"] = df["sec"].shift(1)
    df["prev_zone_all"] = df["details.zoneCode"].shift(1)

    # Filter unblocked shot attempts (Fenwick)
    shots = df[df["typeDescKey"].isin(["shot-on-goal", "goal", "missed-shot"])].copy()
    shots = shots.dropna(subset=["details.xCoord", "details.yCoord", "details.eventOwnerTeam"]).copy()
    if shots.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    shots["is_goal"] = (shots["typeDescKey"] == "goal").astype(int)
    shots["is_sog"] = shots["typeDescKey"].isin(["shot-on-goal", "goal"]).astype(int)
    shots["is_en"] = shots["details.goalieInNetId"].isna()

    # Track prior shot for rebound detection (shot within 3.0s by same team in same period)
    shots["prev_game_shot"] = shots["gameid"].shift(1)
    shots["prev_per_shot"] = shots["periodDescriptor.number"].shift(1)
    shots["prev_team_shot"] = shots["details.eventOwnerTeam"].shift(1)
    shots["prev_sec_shot"] = shots["sec"].shift(1)
    dt_shot = shots["sec"] - shots["prev_sec_shot"]

    shots["is_rebound"] = (
        (shots["gameid"] == shots["prev_game_shot"])
        & (shots["periodDescriptor.number"] == shots["prev_per_shot"])
        & (shots["details.eventOwnerTeam"] == shots["prev_team_shot"])
        & (dt_shot >= 0.0)
        & (dt_shot <= 3.0)
    )

    dt_all = shots["sec"] - shots["prev_sec_all"]
    shots["is_rush"] = (
        (~shots["is_rebound"])
        & (shots["gameid"] == shots["prev_game_all"])
        & (shots["periodDescriptor.number"] == shots["prev_per_all"])
        & (dt_all >= 0.0)
        & (dt_all <= 4.0)
        & (shots["prev_zone_all"].isin(["N", "D"]))
        & (shots["details.zoneCode"] == "O")
    )

    # Coordinate geometry
    x = pd.to_numeric(shots["details.xCoord"], errors="coerce").fillna(0.0).values
    y = pd.to_numeric(shots["details.yCoord"], errors="coerce").fillna(0.0).values
    zone = shots["details.zoneCode"].fillna("O").astype(str).values
    abs_x = np.abs(x)
    abs_y = np.abs(y)

    dx = np.where(
        zone == "D",
        NET_X + abs_x,
        np.where(zone == "N", NET_X + abs_x, np.abs(NET_X - abs_x)),
    )
    dx_safe = np.maximum(dx, 0.5)
    dist = np.sqrt(dx**2 + abs_y**2)
    angle = np.degrees(np.arctan2(abs_y, dx_safe))

    shots["dist"] = np.round(dist, 2)
    shots["angle"] = np.round(angle, 2)

    is_high = (dist <= 20.0) & (angle <= 45.0)
    is_med = (((dist <= 35.0) & (angle <= 50.0)) | ((dist <= 20.0) & (angle > 45.0))) & (~is_high)
    shots["hd_shot"] = is_high.astype(int)
    shots["md_shot"] = is_med.astype(int)
    shots["ld_shot"] = (~is_high & ~is_med).astype(int)

    stype = shots["details.shotType"].fillna("wrist").values
    shots["xg"] = calculate_shot_xg_vectorized(
        distance=dist,
        angle=angle,
        shot_type=stype,
        is_rebound=shots["is_rebound"].values,
        is_rush=shots["is_rush"].values,
        is_empty_net=shots["is_en"].values,
        on_goal_mode=False,
    )
    shots["xga_sog"] = np.where(
        (shots["is_sog"] == 1) & (~shots["is_en"]),
        calculate_shot_xg_vectorized(
            distance=dist,
            angle=angle,
            shot_type=stype,
            is_rebound=shots["is_rebound"].values,
            is_rush=shots["is_rush"].values,
            is_empty_net=shots["is_en"].values,
            on_goal_mode=True,
        ),
        0.0,
    )
    shots["goal_non_en"] = np.where((shots["is_goal"] == 1) & (~shots["is_en"]), 1, 0)

    # -------------------------------------------------------------------------
    # 1. Team-Game Aggregation
    # -------------------------------------------------------------------------
    team_for = (
        shots.groupby(["gameid", "details.eventOwnerTeam"], as_index=False)
        .agg(
            xg_for=("xg", "sum"),
            fenwick_pbp_for=("xg", "count"),
            hdcf_for=("hd_shot", "sum"),
            mdcf_for=("md_shot", "sum"),
            ldcf_for=("ld_shot", "sum"),
            rebound_shots_for=("is_rebound", "sum"),
            rush_shots_for=("is_rush", "sum"),
            xga_on_goal_for=("xga_sog", "sum"),
            ga_non_en_for=("goal_non_en", "sum"),
        )
        .rename(columns={"details.eventOwnerTeam": "team_tri"})
    )
    team_for["xg_for"] = team_for["xg_for"].round(3)

    # Pair each team with its opponent in the same gameid
    team_opp = team_for.rename(
        columns={
            "team_tri": "team_tri_against",
            "xg_for": "xg_against",
            "fenwick_pbp_for": "fenwick_pbp_against",
            "hdcf_for": "hdcf_against",
            "mdcf_for": "mdcf_against",
            "ldcf_for": "ldcf_against",
            "rebound_shots_for": "rebound_shots_against",
            "rush_shots_for": "rush_shots_against",
            "xga_on_goal_for": "xga_faced",
            "ga_non_en_for": "ga_non_en_faced",
        }
    )
    team_game_xg = team_for.merge(team_opp, on="gameid", how="inner")
    team_game_xg = team_game_xg[team_game_xg["team_tri"] != team_game_xg["team_tri_against"]].copy()

    team_game_xg["xg_lvl"] = (team_game_xg["xg_for"] - team_game_xg["xg_against"]).round(3)
    tot_xg = team_game_xg["xg_for"] + team_game_xg["xg_against"]
    team_game_xg["xg_pct"] = np.where(tot_xg > 0, (team_game_xg["xg_for"] / tot_xg).round(4), 0.5)
    tot_hd = team_game_xg["hdcf_for"] + team_game_xg["hdcf_against"]
    team_game_xg["hdcf_pct"] = np.where(tot_hd > 0, (team_game_xg["hdcf_for"] / tot_hd).round(4), 0.5)

    # Team goaltending GSAx in that game = xGA faced on SOG - non-EN Goals Against faced
    team_game_xg["gsax_for"] = (team_game_xg["xga_faced"] - team_game_xg["ga_non_en_faced"]).round(3)
    team_game_xg["gsax_against"] = (team_game_xg["xga_on_goal_for"] - team_game_xg["ga_non_en_for"]).round(3)

    # Build lookup map (gameid, attacking_team) -> defending_team for goalie team attribution
    opp_lookup = dict(
        zip(
            zip(team_game_xg["gameid"], team_game_xg["team_tri"]),
            team_game_xg["team_tri_against"],
        )
    )

    # -------------------------------------------------------------------------
    # 2. Skater-Game Aggregation
    # -------------------------------------------------------------------------
    shots["shooter_id"] = shots["details.shootingPlayerId"].fillna(shots["details.scoringPlayerId"])
    sk_shots = shots.dropna(subset=["shooter_id"]).copy()
    sk_shots["playerId"] = sk_shots["shooter_id"].astype(int)

    skater_game_xg = (
        sk_shots.groupby(["gameid", "playerId", "details.eventOwnerTeam"], as_index=False)
        .agg(
            ixg=("xg", "sum"),
            fenwick_shots=("xg", "count"),
            hd_shots=("hd_shot", "sum"),
            md_shots=("md_shot", "sum"),
            ld_shots=("ld_shot", "sum"),
            rebound_shots=("is_rebound", "sum"),
            rush_shots=("is_rush", "sum"),
            pbp_goals=("is_goal", "sum"),
        )
        .rename(columns={"details.eventOwnerTeam": "team_tri"})
    )
    skater_game_xg["ixg"] = skater_game_xg["ixg"].round(3)
    skater_game_xg["idx_season_type"] = ((skater_game_xg["gameid"] // 10000) % 100).astype(int)

    # -------------------------------------------------------------------------
    # 3. Goalie-Game Aggregation
    # -------------------------------------------------------------------------
    gl_shots = shots[(shots["is_sog"] == 1) & (~shots["is_en"]) & (shots["details.goalieInNetId"].notna())].copy()
    gl_shots["playerId"] = gl_shots["details.goalieInNetId"].astype(int)
    gl_shots["team_tri"] = [
        opp_lookup.get((gid, att_tm), "")
        for gid, att_tm in zip(gl_shots["gameid"], gl_shots["details.eventOwnerTeam"])
    ]
    gl_shots["hd_ga"] = np.where((gl_shots["hd_shot"] == 1) & (gl_shots["is_goal"] == 1), 1, 0)

    goalie_game_xg = (
        gl_shots.groupby(["gameid", "playerId", "team_tri"], as_index=False)
        .agg(
            xga=("xga_sog", "sum"),
            pbp_sa=("is_sog", "sum"),
            pbp_ga=("is_goal", "sum"),
            hd_shots_against=("hd_shot", "sum"),
            hd_goals_against=("hd_ga", "sum"),
        )
    )
    goalie_game_xg["xga"] = goalie_game_xg["xga"].round(3)
    goalie_game_xg["gsax"] = (goalie_game_xg["xga"] - goalie_game_xg["pbp_ga"]).round(3)
    goalie_game_xg["xsave_pct"] = np.where(
        goalie_game_xg["pbp_sa"] > 0,
        (1.0 - goalie_game_xg["xga"] / goalie_game_xg["pbp_sa"]).round(4),
        np.nan,
    )
    goalie_game_xg["hd_save_pct"] = np.where(
        goalie_game_xg["hd_shots_against"] > 0,
        (1.0 - goalie_game_xg["hd_goals_against"] / goalie_game_xg["hd_shots_against"]).round(4),
        np.nan,
    )
    goalie_game_xg["idx_season_type"] = ((goalie_game_xg["gameid"] // 10000) % 100).astype(int)

    return team_game_xg, skater_game_xg, goalie_game_xg


def _kpinorm_series(s: pd.Series) -> pd.Series:
    """Min-max [0, 1] KPI normalizer matching procs_boxscore.kpinorm."""
    vmin = s.min()
    vmax = s.max()
    if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
        return pd.Series(0.5, index=s.index)
    return ((s - vmin) / (vmax - vmin)).round(4)


def enrich_season_xg(year: int | str, base_dir: str = "./latest") -> Dict[str, int]:
    """
    Computes and merges xG, ixG, Danger Zone, and GSAx metrics for a given season `year`
    across all team and player measurement CSVs in `base_dir`:
      - `{base_dir}/box/{year}_box.csv`
      - `{base_dir}/box/{year}_box_team.csv`
      - `{base_dir}/box/{year}_box_gameStats.csv`
      - `{base_dir}/team/season/{year}_team_season.csv`
      - `{base_dir}/box/{year}_box_player.csv`
      - `{base_dir}/player/{year}_player.csv`
    Returns a summary dict of updated row counts.
    """
    yr = str(year)
    pbp_path = os.path.join(base_dir, "play", f"{yr}_playbyplay.csv")
    if not os.path.exists(pbp_path):
        return {"status": 0}

    df_pbp = pd.read_csv(pbp_path, low_memory=False)
    team_game_xg, skater_game_xg, goalie_game_xg = nhl_dataproc_pbp_xg(df_pbp)
    if team_game_xg.empty:
        return {"status": 0}

    counts: Dict[str, int] = {"year": int(yr)}

    # -------------------------------------------------------------------------
    # A. Enrich `{year}_box.csv` (Scoreboard game-level records)
    # -------------------------------------------------------------------------
    box_path = os.path.join(base_dir, "box", f"{yr}_box.csv")
    if os.path.exists(box_path):
        df_box = pd.read_csv(box_path)
        xg_Cols = ["xg_for", "xg_against", "xg_diff", "xg_pct_for", "hdcf_for", "hdcf_against", "gsax_for", "gsax_against"]
        df_box = df_box.drop(columns=[c for c in xg_Cols if c in df_box.columns], errors="ignore")
        tg_sub = team_game_xg[
            ["gameid", "team_tri", "xg_for", "xg_against", "xg_lvl", "xg_pct", "hdcf_for", "hdcf_against", "gsax_for", "gsax_against"]
        ].rename(
            columns={
                "team_tri": "tricode_for",
                "xg_lvl": "xg_diff",
                "xg_pct": "xg_pct_for",
            }
        )
        df_box = df_box.merge(tg_sub, on=["gameid", "tricode_for"], how="left")
        df_box.to_csv(box_path, index=False)
        counts["box_rows"] = len(df_box)

    # -------------------------------------------------------------------------
    # B. Enrich `{year}_box_team.csv`
    # -------------------------------------------------------------------------
    box_team_path = os.path.join(base_dir, "box", f"{yr}_box_team.csv")
    if os.path.exists(box_team_path):
        df_bt = pd.read_csv(box_team_path)
        bt_drop = ["xg", "xga", "xg_lvl", "xg_pct", "hdcf", "hdca", "gsax"]
        df_bt = df_bt.drop(columns=[c for c in bt_drop if c in df_bt.columns], errors="ignore")
        tg_bt = team_game_xg[
            ["gameid", "team_tri", "xg_for", "xg_against", "xg_lvl", "xg_pct", "hdcf_for", "hdcf_against", "gsax_for"]
        ].rename(
            columns={
                "xg_for": "xg",
                "xg_against": "xga",
                "hdcf_for": "hdcf",
                "hdcf_against": "hdca",
                "gsax_for": "gsax",
            }
        )
        df_bt = df_bt.merge(tg_bt, on=["gameid", "team_tri"], how="left")
        df_bt.to_csv(box_team_path, index=False)
        counts["box_team_rows"] = len(df_bt)

    # -------------------------------------------------------------------------
    # C. Enrich `{year}_box_gameStats.csv` (Game + Rolling Season-to-Date)
    # -------------------------------------------------------------------------
    gs_path = os.path.join(base_dir, "box", f"{yr}_box_gameStats.csv")
    if os.path.exists(gs_path):
        df_gs = pd.read_csv(gs_path, low_memory=False)
        gs_xg_cols = [
            "xg_for", "xg_against", "xg_lvl", "xg_pct",
            "hdcf_for", "hdcf_against", "hdcf_pct",
            "mdcf_for", "mdcf_against", "ldcf_for", "ldcf_against",
            "gsax_for", "gsax_against",
            "rxg_for", "rxg_against", "rxg_lvl", "rxg_pct",
            "rhdcf_for", "rhdcf_against", "rhdcf_pct", "rgsax_for",
        ]
        df_gs = df_gs.drop(columns=[c for c in gs_xg_cols if c in df_gs.columns], errors="ignore")

        tg_gs = team_game_xg[
            [
                "gameid", "team_tri",
                "xg_for", "xg_against", "xg_lvl", "xg_pct",
                "hdcf_for", "hdcf_against", "hdcf_pct",
                "mdcf_for", "mdcf_against", "ldcf_for", "ldcf_against",
                "gsax_for", "gsax_against",
            ]
        ].rename(columns={"gameid": "gameid_for", "team_tri": "team_tri_for"})

        df_gs = df_gs.merge(tg_gs, on=["gameid_for", "team_tri_for"], how="left")

        # Sort chronologically per team to compute cumulative rolling season-to-date metrics
        if "gameDate_for" in df_gs.columns:
            df_gs = df_gs.sort_values(["gameDate_for", "gameid_for"]).reset_index(drop=True)

        for col in ["xg_for", "xg_against", "hdcf_for", "hdcf_against", "gsax_for"]:
            df_gs[col] = df_gs[col].fillna(0.0)

        df_gs["rxg_for"] = df_gs.groupby("team_tri_for")["xg_for"].cumsum().round(3)
        df_gs["rxg_against"] = df_gs.groupby("team_tri_for")["xg_against"].cumsum().round(3)
        df_gs["rxg_lvl"] = (df_gs["rxg_for"] - df_gs["rxg_against"]).round(3)
        rtot_xg = df_gs["rxg_for"] + df_gs["rxg_against"]
        df_gs["rxg_pct"] = np.where(rtot_xg > 0, (df_gs["rxg_for"] / rtot_xg).round(4), 0.5)

        df_gs["rhdcf_for"] = df_gs.groupby("team_tri_for")["hdcf_for"].cumsum().astype(int)
        df_gs["rhdcf_against"] = df_gs.groupby("team_tri_for")["hdcf_against"].cumsum().astype(int)
        rtot_hd = df_gs["rhdcf_for"] + df_gs["rhdcf_against"]
        df_gs["rhdcf_pct"] = np.where(rtot_hd > 0, (df_gs["rhdcf_for"] / rtot_hd).round(4), 0.5)
        df_gs["rgsax_for"] = df_gs.groupby("team_tri_for")["gsax_for"].cumsum().round(3)

        df_gs.to_csv(gs_path, index=False)
        counts["game_stats_rows"] = len(df_gs)

    # -------------------------------------------------------------------------
    # D. Enrich `{year}_team_season.csv` (Team Season Aggregates & KPIs)
    # -------------------------------------------------------------------------
    ts_path = os.path.join(base_dir, "team", "season", f"{yr}_team_season.csv")
    if os.path.exists(ts_path):
        df_ts = pd.read_csv(ts_path)
        ts_drop = [
            "xg_for", "xg_against", "xg_lvl", "xg_pct",
            "hdcf_for", "hdcf_against", "hdcf_pct",
            "gsax", "kpi_xg", "kpi_gsax",
        ]
        df_ts = df_ts.drop(columns=[c for c in ts_drop if c in df_ts.columns], errors="ignore")

        team_season_xg = (
            team_game_xg.groupby("team_tri", as_index=False)
            .agg(
                xg_for=("xg_for", "sum"),
                xg_against=("xg_against", "sum"),
                hdcf_for=("hdcf_for", "sum"),
                hdcf_against=("hdcf_against", "sum"),
                gsax=("gsax_for", "sum"),
            )
            .rename(columns={"team_tri": "team_tri_for"})
        )
        team_season_xg["xg_for"] = team_season_xg["xg_for"].round(2)
        team_season_xg["xg_against"] = team_season_xg["xg_against"].round(2)
        team_season_xg["xg_lvl"] = (team_season_xg["xg_for"] - team_season_xg["xg_against"]).round(2)
        tot_s_xg = team_season_xg["xg_for"] + team_season_xg["xg_against"]
        team_season_xg["xg_pct"] = np.where(tot_s_xg > 0, (team_season_xg["xg_for"] / tot_s_xg).round(4), 0.5)
        tot_s_hd = team_season_xg["hdcf_for"] + team_season_xg["hdcf_against"]
        team_season_xg["hdcf_pct"] = np.where(tot_s_hd > 0, (team_season_xg["hdcf_for"] / tot_s_hd).round(4), 0.5)
        team_season_xg["gsax"] = team_season_xg["gsax"].round(2)
        team_season_xg["kpi_xg"] = _kpinorm_series(team_season_xg["xg_lvl"])
        team_season_xg["kpi_gsax"] = _kpinorm_series(team_season_xg["gsax"])

        df_ts = df_ts.merge(team_season_xg, on="team_tri_for", how="left")
        df_ts.to_csv(ts_path, index=False)
        counts["team_season_rows"] = len(df_ts)

    # -------------------------------------------------------------------------
    # E. Enrich `{year}_box_player.csv` (Player Game + Rolling Season-to-Date)
    # -------------------------------------------------------------------------
    bp_path = os.path.join(base_dir, "box", f"{yr}_box_player.csv")
    if os.path.exists(bp_path):
        df_bp = pd.read_csv(bp_path, low_memory=False)
        bp_xg_cols = [
            "ixg", "g_minus_ixg", "fenwick_shots", "hd_shots", "md_shots", "ld_shots",
            "rebound_shots", "rush_shots",
            "xga", "gsax", "xsave_pct", "dsave_pct",
            "hd_shots_against", "hd_goals_against", "hd_save_pct",
            "cum_gp", "cum_goals", "cum_assists", "cum_points",
            "cum_ixg", "cum_g_minus_ixg", "cum_hd_shots",
            "cum_xga", "cum_gsax", "cum_sv_pct", "cum_xsv_pct", "cum_dsv_pct",
            "is_season_end",
        ]
        df_bp = df_bp.drop(columns=[c for c in bp_xg_cols if c in df_bp.columns], errors="ignore")

        # Ensure existing box_player table has strictly unique (gameid, playerId) rows
        df_bp = df_bp.drop_duplicates(subset=["gameid", "playerId"]).reset_index(drop=True)

        sk_cols = ["ixg", "fenwick_shots", "hd_shots", "md_shots", "ld_shots", "rebound_shots", "rush_shots"]
        sk_sub = (
            skater_game_xg.groupby(["gameid", "playerId"], as_index=False)[sk_cols]
            .sum()
        )
        sk_sub["ixg"] = sk_sub["ixg"].round(3)

        gl_sub = (
            goalie_game_xg.groupby(["gameid", "playerId"], as_index=False)
            .agg(
                xga=("xga", "sum"),
                gsax=("gsax", "sum"),
                xsave_pct=("xsave_pct", "mean"),
                hd_shots_against=("hd_shots_against", "sum"),
                hd_goals_against=("hd_goals_against", "sum"),
                hd_save_pct=("hd_save_pct", "mean"),
            )
        )
        gl_sub["xga"] = gl_sub["xga"].round(3)
        gl_sub["gsax"] = gl_sub["gsax"].round(3)
        gl_sub["xsave_pct"] = gl_sub["xsave_pct"].round(4)
        gl_sub["hd_save_pct"] = gl_sub["hd_save_pct"].round(4)

        df_bp = df_bp.merge(sk_sub, on=["gameid", "playerId"], how="left")
        df_bp = df_bp.merge(gl_sub, on=["gameid", "playerId"], how="left")
        df_bp = df_bp.drop_duplicates(subset=["gameid", "playerId"]).reset_index(drop=True)

        is_goalie = df_bp["position"] == "G"
        for col in ["ixg", "fenwick_shots", "hd_shots", "md_shots", "ld_shots", "rebound_shots", "rush_shots"]:
            df_bp[col] = np.where(~is_goalie, df_bp[col].fillna(0.0), np.nan)

        df_bp["g_minus_ixg"] = np.where(~is_goalie, (df_bp["goals"].fillna(0.0) - df_bp["ixg"]).round(3), np.nan)

        # For goalies, align gsax and dsave_pct with official box score goalsAgainst / shotsAgainst when available
        sa_col = pd.to_numeric(df_bp.get("shotsAgainst"), errors="coerce")
        ga_col = pd.to_numeric(df_bp.get("goalsAgainst"), errors="coerce")
        sv_col = pd.to_numeric(df_bp.get("savePctg"), errors="coerce")

        df_bp["xga"] = np.where(is_goalie, df_bp["xga"].fillna(0.0), np.nan)
        df_bp["gsax"] = np.where(
            is_goalie,
            (df_bp["xga"] - ga_col.fillna(0.0)).round(3),
            np.nan,
        )
        df_bp["xsave_pct"] = np.where(
            is_goalie & (sa_col > 0),
            (1.0 - df_bp["xga"] / sa_col).round(4),
            df_bp["xsave_pct"],
        )
        df_bp["dsave_pct"] = np.where(
            is_goalie & sv_col.notna() & df_bp["xsave_pct"].notna(),
            (sv_col - df_bp["xsave_pct"]).round(4),
            np.nan,
        )

        # Sort chronologically per player to compute rolling season-to-date columns
        df_bp = df_bp.sort_values(["gameDate", "gameid", "playerId"]).reset_index(drop=True)
        df_bp["cum_gp"] = df_bp.groupby("playerId").cumcount() + 1

        # Skater cumulative rolling season-to-date
        df_bp["cum_goals"] = np.where(~is_goalie, df_bp.groupby("playerId")["goals"].cumsum(), np.nan)
        df_bp["cum_assists"] = np.where(~is_goalie, df_bp.groupby("playerId")["assists"].cumsum(), np.nan)
        df_bp["cum_points"] = np.where(~is_goalie, df_bp.groupby("playerId")["points"].cumsum(), np.nan)
        df_bp["cum_ixg"] = np.where(~is_goalie, df_bp.groupby("playerId")["ixg"].cumsum().round(3), np.nan)
        df_bp["cum_g_minus_ixg"] = np.where(~is_goalie, (df_bp["cum_goals"] - df_bp["cum_ixg"]).round(3), np.nan)
        df_bp["cum_hd_shots"] = np.where(~is_goalie, df_bp.groupby("playerId")["hd_shots"].cumsum(), np.nan)

        # Goalie cumulative rolling season-to-date
        df_bp["_sa_tmp"] = np.where(is_goalie, sa_col.fillna(0.0), 0.0)
        df_bp["_ga_tmp"] = np.where(is_goalie, ga_col.fillna(0.0), 0.0)
        df_bp["_xga_tmp"] = np.where(is_goalie, df_bp["xga"].fillna(0.0), 0.0)

        cum_sa = df_bp.groupby("playerId")["_sa_tmp"].cumsum()
        cum_ga = df_bp.groupby("playerId")["_ga_tmp"].cumsum()
        cum_xga = df_bp.groupby("playerId")["_xga_tmp"].cumsum().round(3)

        df_bp["cum_xga"] = np.where(is_goalie, cum_xga, np.nan)
        df_bp["cum_gsax"] = np.where(is_goalie, (cum_xga - cum_ga).round(3), np.nan)
        df_bp["cum_sv_pct"] = np.where(is_goalie & (cum_sa > 0), (1.0 - cum_ga / cum_sa).round(4), np.nan)
        df_bp["cum_xsv_pct"] = np.where(is_goalie & (cum_sa > 0), (1.0 - cum_xga / cum_sa).round(4), np.nan)
        df_bp["cum_dsv_pct"] = np.where(
            is_goalie & df_bp["cum_sv_pct"].notna() & df_bp["cum_xsv_pct"].notna(),
            (df_bp["cum_sv_pct"] - df_bp["cum_xsv_pct"]).round(4),
            np.nan,
        )
        df_bp = df_bp.drop(columns=["_sa_tmp", "_ga_tmp", "_xga_tmp"])

        # Mark each player's latest/final game of the season (`is_season_end`)
        last_idx = df_bp.groupby("playerId").tail(1).index
        df_bp["is_season_end"] = df_bp.index.isin(last_idx)

        df_bp.to_csv(bp_path, index=False)
        counts["box_player_rows"] = len(df_bp)

    # -------------------------------------------------------------------------
    # F. Enrich `{year}_player.csv` (Player Season-to-Date / Season-Ending)
    # -------------------------------------------------------------------------
    pl_path = os.path.join(base_dir, "player", f"{yr}_player.csv")
    if os.path.exists(pl_path):
        df_pl = pd.read_csv(pl_path, low_memory=False)
        pl_xg_cols = [
            "ixg", "ixg_per_game", "goals_above_expected", "fenwick_shots",
            "hd_shots", "md_shots", "ld_shots", "rebound_shots", "rush_shots",
            "xga", "gsax", "gsax_per_game", "xsave_pct", "dsave_pct",
            "hd_shots_against", "hd_goals_against", "hd_save_pct",
        ]
        df_pl = df_pl.drop(columns=[c for c in pl_xg_cols if c in df_pl.columns], errors="ignore")

        # Repair missing idx_season_type (2=Regular Season, 3=Playoffs) per team_tri block
        if "idx_season_type" not in df_pl.columns:
            df_pl["idx_season_type"] = np.nan
        if df_pl["idx_season_type"].isna().any():
            for tm, grp in df_pl.groupby("team_tri", sort=False):
                pos = grp["positionCode"].tolist()
                idxs = grp.index.tolist()
                trans = [i for i in range(1, len(pos)) if pos[i] == "G" and pos[i - 1] != "G"]
                if trans:
                    split_i = trans[0]
                    reg_idxs = idxs[:split_i]
                    po_idxs = idxs[split_i:]
                    df_pl.loc[reg_idxs, "idx_season_type"] = df_pl.loc[reg_idxs, "idx_season_type"].fillna(2)
                    df_pl.loc[po_idxs, "idx_season_type"] = df_pl.loc[po_idxs, "idx_season_type"].fillna(3)
                else:
                    df_pl.loc[idxs, "idx_season_type"] = df_pl.loc[idxs, "idx_season_type"].fillna(2)
        df_pl["idx_season_type"] = pd.to_numeric(df_pl["idx_season_type"], errors="coerce").fillna(2).astype(int)

        # Aggregate skater xG by (playerId, team_tri, idx_season_type)
        sk_season_tm = (
            skater_game_xg.groupby(["playerId", "team_tri", "idx_season_type"], as_index=False)
            .agg(
                ixg=("ixg", "sum"),
                fenwick_shots=("fenwick_shots", "sum"),
                hd_shots=("hd_shots", "sum"),
                md_shots=("md_shots", "sum"),
                ld_shots=("ld_shots", "sum"),
                rebound_shots=("rebound_shots", "sum"),
                rush_shots=("rush_shots", "sum"),
                pbp_goals=("pbp_goals", "sum"),
            )
        )
        # Fallback aggregate by (playerId, idx_season_type) for single-team players
        sk_season_all = (
            skater_game_xg.groupby(["playerId", "idx_season_type"], as_index=False)
            .agg(
                ixg_all=("ixg", "sum"),
                fenwick_shots_all=("fenwick_shots", "sum"),
                hd_shots_all=("hd_shots", "sum"),
                md_shots_all=("md_shots", "sum"),
                ld_shots_all=("ld_shots", "sum"),
                rebound_shots_all=("rebound_shots", "sum"),
                rush_shots_all=("rush_shots", "sum"),
                pbp_goals_all=("pbp_goals", "sum"),
            )
        )

        # Aggregate goalie xGA by (playerId, team_tri, idx_season_type) and fallback by (playerId, idx_season_type)
        gl_season_tm = (
            goalie_game_xg.groupby(["playerId", "team_tri", "idx_season_type"], as_index=False)
            .agg(
                xga=("xga", "sum"),
                pbp_sa=("pbp_sa", "sum"),
                pbp_ga=("pbp_ga", "sum"),
                hd_shots_against=("hd_shots_against", "sum"),
                hd_goals_against=("hd_goals_against", "sum"),
            )
        )
        gl_season_all = (
            goalie_game_xg.groupby(["playerId", "idx_season_type"], as_index=False)
            .agg(
                xga_all=("xga", "sum"),
                pbp_sa_all=("pbp_sa", "sum"),
                pbp_ga_all=("pbp_ga", "sum"),
                hd_shots_against_all=("hd_shots_against", "sum"),
                hd_goals_against_all=("hd_goals_against", "sum"),
            )
        )

        df_pl = df_pl.merge(sk_season_tm, on=["playerId", "team_tri", "idx_season_type"], how="left")
        df_pl = df_pl.merge(sk_season_all, on=["playerId", "idx_season_type"], how="left")
        df_pl = df_pl.merge(gl_season_tm, on=["playerId", "team_tri", "idx_season_type"], how="left")
        df_pl = df_pl.merge(gl_season_all, on=["playerId", "idx_season_type"], how="left")

        # Count how many rows a player has in df_pl within the same season_type (1 = single team, >1 = traded)
        pl_team_cnt = df_pl.groupby(["playerId", "idx_season_type"])["team_tri"].transform("count")

        for col in ["ixg", "fenwick_shots", "hd_shots", "md_shots", "ld_shots", "rebound_shots", "rush_shots", "pbp_goals"]:
            df_pl[col] = np.where(
                df_pl[col].notna(),
                df_pl[col],
                np.where(pl_team_cnt == 1, df_pl[f"{col}_all"], 0.0),
            )
            df_pl = df_pl.drop(columns=[f"{col}_all"])

        for col in ["xga", "pbp_sa", "pbp_ga", "hd_shots_against", "hd_goals_against"]:
            df_pl[col] = np.where(
                df_pl[col].notna(),
                df_pl[col],
                np.where(pl_team_cnt == 1, df_pl[f"{col}_all"], np.nan),
            )
            df_pl = df_pl.drop(columns=[f"{col}_all"])

        is_g_pl = df_pl["positionCode"] == "G"
        gp_pl = pd.to_numeric(df_pl["gamesPlayed"], errors="coerce").replace(0, np.nan)

        # Skater derived columns (exact play-by-play matched goals - ixg)
        df_pl["ixg"] = np.where(~is_g_pl, df_pl["ixg"].fillna(0.0).round(2), np.nan)
        df_pl["ixg_per_game"] = np.where(~is_g_pl & gp_pl.notna(), (df_pl["ixg"] / gp_pl).round(3), np.nan)
        df_pl["goals_above_expected"] = np.where(
            ~is_g_pl,
            (df_pl["pbp_goals"].fillna(0.0) - df_pl["ixg"]).round(2),
            np.nan,
        )
        for col in ["fenwick_shots", "hd_shots", "md_shots", "ld_shots", "rebound_shots", "rush_shots"]:
            df_pl[col] = np.where(~is_g_pl, df_pl[col].fillna(0.0), np.nan)

        # Goalie derived columns (exact play-by-play matched xga - pbp_ga and pbp_sa denominator)
        pbp_sa = pd.to_numeric(df_pl["pbp_sa"], errors="coerce")
        pbp_ga = pd.to_numeric(df_pl["pbp_ga"], errors="coerce")

        df_pl["xga"] = np.where(is_g_pl, df_pl["xga"].round(2), np.nan)
        df_pl["gsax"] = np.where(is_g_pl & df_pl["xga"].notna() & pbp_ga.notna(), (df_pl["xga"] - pbp_ga).round(2), np.nan)
        df_pl["gsax_per_game"] = np.where(is_g_pl & df_pl["gsax"].notna() & gp_pl.notna(), (df_pl["gsax"] / gp_pl).round(3), np.nan)
        df_pl["xsave_pct"] = np.where(
            is_g_pl & df_pl["xga"].notna() & (pbp_sa > 0),
            (1.0 - df_pl["xga"] / pbp_sa).round(4),
            np.nan,
        )
        df_pl["dsave_pct"] = np.where(
            is_g_pl & df_pl["gsax"].notna() & (pbp_sa > 0),
            (df_pl["gsax"] / pbp_sa).round(4),
            np.nan,
        )
        df_pl["hd_save_pct"] = np.where(
            is_g_pl & (df_pl["hd_shots_against"] > 0),
            (1.0 - df_pl["hd_goals_against"] / df_pl["hd_shots_against"]).round(4),
            np.nan,
        )
        df_pl = df_pl.drop(columns=["pbp_goals", "pbp_sa", "pbp_ga"], errors="ignore")

        df_pl.to_csv(pl_path, index=False)
        counts["player_season_rows"] = len(df_pl)

    return counts
