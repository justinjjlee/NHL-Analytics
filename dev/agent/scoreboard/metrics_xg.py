"""
metrics_xg.py — Expected Goals (xG) & Goals Saved Above Expected (GSAx) Engine.

Canonical implementation lives in `src/data/apinhle/function/procs_xg.py`.
This module re-exports all data structures, constants, and functions for backward
compatibility with the Scoreboard Agent (`dev/agent/scoreboard/`).
"""
from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.apinhle.function.procs_xg import (  # noqa: E402,F401
    BASE_LOG_ODDS,
    BASE_LOG_ODDS_FENWICK,
    BASE_LOG_ODDS_ON_GOAL,
    BETA_ANGLE,
    BETA_ANGLE_FENWICK,
    BETA_ANGLE_ON_GOAL,
    BETA_DISTANCE,
    BETA_DISTANCE_FENWICK,
    BETA_DISTANCE_ON_GOAL,
    BETA_INTERACTION,
    CREASE_CLAMP_FT,
    HALF_WIDTH,
    NET_X,
    POWER_PLAY_BOOST,
    REBOUND_BOOST,
    RINK_LENGTH,
    RINK_WIDTH,
    RUSH_BOOST,
    SHORTHANDED_PENALTY,
    SHOT_TYPE_WEIGHTS,
    SLOT_CORRIDOR_FENWICK,
    SLOT_CORRIDOR_ON_GOAL,
    GameXGReport,
    GoalieXGSummary,
    ShotAttempt,
    SkaterXGSummary,
    calculate_shot_xg,
    calculate_shot_xg_vectorized,
    classify_danger_zone,
    enrich_season_xg,
    evaluate_game_xg,
    nhl_dataproc_pbp_xg,
    normalize_shot_geometry,
)
