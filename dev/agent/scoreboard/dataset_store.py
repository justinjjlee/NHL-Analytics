"""Hugging Face Dataset Store for NHL Scoreboard Agent.

Provides persistent record-keeping for generated game recaps, slate summaries,
and analytical narrations inside the Hugging Face dataset `jjerlee/nhl-analytics`.

Features:
- Idempotent execution: Only generates recaps when new games or dates are unrecorded.
- Parquet storage: High-performance columnar storage with fast date and game_id lookups.
- Cloud synchronization: Uploads new/updated records to Hugging Face using HF_TOKEN.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Safe import for huggingface_hub
try:
    from huggingface_hub import HfApi, hf_hub_download, get_token
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

try:
    from .narrator import NarratorEngine
    from .env_utils import discover_hf_token, is_local_streamlit
except (ImportError, ValueError):
    from narrator import NarratorEngine
    from env_utils import discover_hf_token, is_local_streamlit


DEFAULT_HF_REPO = "jjerlee/nhl-analytics"
DEFAULT_HF_SUBDIR = "insights/scoreboard"
HF_BASE_URL = f"https://huggingface.co/datasets/{DEFAULT_HF_REPO}/resolve/main"

# Column schemas
GAME_RECAP_COLUMNS = [
    "game_id",
    "game_date",
    "season",
    "away_team",
    "away_score",
    "home_team",
    "home_score",
    "period_ending",
    "headline",
    "narrative_recap",
    "full_report_markdown",
    "milestones_json",
    "three_stars_json",
    "away_stats_json",
    "home_stats_json",
    "game_script_type",
    "period_flow_json",
    "defensive_effort_json",
    "goalie_performance_json",
    "season_standings_json",
    "validation_report_json",
    "generated_at",
    "generator_model",
]

SLATE_RECAP_COLUMNS = [
    "game_date",
    "total_games",
    "total_goals",
    "avg_goals_per_game",
    "home_wins",
    "away_wins",
    "overtime_games",
    "three_stars_json",
    "milestones_json",
    "slate_headline",
    "slate_narrative",
    "generated_at",
    "generator_model",
]

_CORRUPT_MARKERS = (
    "alright, i need",
    "okay, i need to",
    "i need to write",
    "let me start by",
    "thinking process:",
    "<think>",
    "</think>",
    "in the reg period",
)


def _discover_hf_token() -> Optional[str]:
    """Helper to discover Hugging Face token across environment, streamlit, and .env file."""
    return discover_hf_token()


def get_model_priority(model_name: Optional[str]) -> int:
    """Returns priority tier for generator models:
    Tier 2: High-fidelity LLM synthesis (e.g. 'lm-studio/*', 'hf-space/*', 'meta-llama/*', 'remote-endpoint/*')
    Tier 1: Deterministic Algorithmic Guardrail Engine ('algorithmic-guardrails-v1', 'algorithmic*')
    Tier 0: Blank, missing, or legacy placeholder ('scoreboard-agent-v1', None, '')
    """
    if not model_name:
        return 0
    m = str(model_name).strip().lower()
    if not m or m in ("none", "null", "nan", "unknown", "undefined"):
        return 0
    if "algorithmic" in m:
        return 1
    if m in ("scoreboard-agent-v1", "default", "legacy"):
        return 0
    # Any neural / LLM model
    return 2


def _clean_game_recaps_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Filters out any malformed, empty, or reasoning-leaked rows from game_recaps DataFrame,
    prioritizing LLM-generated models over algorithmic fallbacks.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=GAME_RECAP_COLUMNS)
    out = df.copy()
    if "narrative_recap" in out.columns:
        narr_low = out["narrative_recap"].fillna("").astype(str).str.lower()
        valid_mask = narr_low.str.len() >= 80
        for marker in _CORRUPT_MARKERS:
            valid_mask = valid_mask & (~narr_low.str.contains(marker, regex=False))
        out = out[valid_mask]
    if "game_id" in out.columns and not out.empty:
        # Precedence: Tier 2 (LLM) > Tier 1 (Algorithmic) > Tier 0 (Blank/Legacy)
        # Ties within same tier broken by latest generated_at timestamp
        if "generator_model" in out.columns:
            out["_model_prio"] = out["generator_model"].apply(get_model_priority)
        else:
            out["_model_prio"] = 0

        sort_keys = ["_model_prio"]
        if "generated_at" in out.columns:
            sort_keys.append("generated_at")

        out = out.sort_values(sort_keys, ascending=True)
        out = out.drop_duplicates(subset=["game_id"], keep="last")
        out = out.drop(columns=["_model_prio"], errors="ignore")
    return out.reset_index(drop=True)


def _clean_slate_recaps_df(df: Optional[pd.DataFrame], valid_game_dates: Optional[set] = None) -> pd.DataFrame:
    """Filters out empty (0-game) or reasoning-leaked rows from slate_recaps DataFrame,
    prioritizing LLM-generated models over algorithmic fallbacks.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=SLATE_RECAP_COLUMNS)
    out = df.copy()
    if "total_games" in out.columns:
        out = out[pd.to_numeric(out["total_games"], errors="coerce").fillna(0) > 0]
    if "slate_narrative" in out.columns and not out.empty:
        narr_low = out["slate_narrative"].fillna("").astype(str).str.lower()
        valid_mask = narr_low.str.len() >= 80
        for marker in _CORRUPT_MARKERS:
            valid_mask = valid_mask & (~narr_low.str.contains(marker, regex=False))
        out = out[valid_mask]
    if valid_game_dates is not None and "game_date" in out.columns and not out.empty:
        out = out[out["game_date"].astype(str).isin(valid_game_dates)]
    if "game_date" in out.columns and not out.empty:
        # Precedence: Tier 2 (LLM) > Tier 1 (Algorithmic) > Tier 0 (Blank/Legacy)
        # Ties within same tier broken by latest generated_at timestamp
        if "generator_model" in out.columns:
            out["_model_prio"] = out["generator_model"].apply(get_model_priority)
        else:
            out["_model_prio"] = 0

        sort_keys = ["_model_prio"]
        if "generated_at" in out.columns:
            sort_keys.append("generated_at")

        out = out.sort_values(sort_keys, ascending=True)
        out = out.drop_duplicates(subset=["game_date"], keep="last")
        out = out.drop(columns=["_model_prio"], errors="ignore")
    return out.reset_index(drop=True)


class HFScoreboardStore:
    """Manages reading, caching, record-keeping and syncing of generated texts to Hugging Face."""

    def __init__(
        self,
        local_dir: Optional[Path | str] = None,
        repo_id: str = DEFAULT_HF_REPO,
        token: Optional[str] = None,
        remote_subdir: str = DEFAULT_HF_SUBDIR,
    ):
        self.repo_id = repo_id
        self.remote_subdir = remote_subdir.strip("/")
        self.token = discover_hf_token(token)
        
        if local_dir:
            self.local_dir = Path(local_dir)
        else:
            # External cache outside git repository (~/.cache/rinklytics/insights/scoreboard)
            cache_base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
            self.local_dir = cache_base / "rinklytics" / "insights" / "scoreboard"
            # Migrate from legacy ~/.cache/rinklytics/scoreboard if it exists
            legacy_dir = cache_base / "rinklytics" / "scoreboard"
            if legacy_dir.exists() and not self.local_dir.exists():
                try:
                    import shutil
                    self.local_dir.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(legacy_dir, self.local_dir)
                except Exception:
                    pass
            
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.game_recaps_file = self.local_dir / "game_recaps.parquet"
        self.slate_recaps_file = self.local_dir / "slate_recaps.parquet"

        self._game_recaps_df: Optional[pd.DataFrame] = None
        self._slate_recaps_df: Optional[pd.DataFrame] = None
        
        # Load tables (local cache or remote dataset)
        self._load_tables()

    def _fetch_remote_parquet(self, filename: str) -> Optional[pd.DataFrame]:
        """Downloads a parquet table from the remote Hugging Face dataset if reachable."""
        urls_to_try = [
            f"{HF_BASE_URL}/{self.remote_subdir}/{filename}",
            f"{HF_BASE_URL}/scoreboard/{filename}",
        ]
        for remote_url in urls_to_try:
            try:
                return pd.read_parquet(remote_url)
            except Exception:
                continue
        return None

    def _load_tables(self, refresh_remote: bool = False) -> None:
        """Loads and cleans game and slate recaps from local cache and/or remote HF dataset."""
        local_games = None
        if self.game_recaps_file.exists():
            try:
                local_games = pd.read_parquet(self.game_recaps_file)
            except Exception as e:
                print(f"[HFScoreboardStore] Warning loading local game recaps: {e}", file=sys.stderr)

        remote_games = None
        if refresh_remote or local_games is None or local_games.empty or not is_local_streamlit():
            remote_games = self._fetch_remote_parquet("game_recaps.parquet")

        if local_games is not None and remote_games is not None:
            combined_games = pd.concat([remote_games, local_games], ignore_index=True)
        elif local_games is not None:
            combined_games = local_games
        elif remote_games is not None:
            combined_games = remote_games
        else:
            combined_games = pd.DataFrame(columns=GAME_RECAP_COLUMNS)

        self._game_recaps_df = _clean_game_recaps_df(combined_games)
        if not self._game_recaps_df.empty:
            try:
                self._game_recaps_df.to_parquet(self.game_recaps_file, index=False)
            except Exception:
                pass

        valid_game_dates = (
            set(self._game_recaps_df["game_date"].astype(str).unique())
            if not self._game_recaps_df.empty and "game_date" in self._game_recaps_df.columns
            else set()
        )

        # 2. Slate Recaps
        local_slates = None
        if self.slate_recaps_file.exists():
            try:
                local_slates = pd.read_parquet(self.slate_recaps_file)
            except Exception as e:
                print(f"[HFScoreboardStore] Warning loading local slate recaps: {e}", file=sys.stderr)

        remote_slates = None
        if refresh_remote or local_slates is None or local_slates.empty or not is_local_streamlit():
            remote_slates = self._fetch_remote_parquet("slate_recaps.parquet")

        if local_slates is not None and remote_slates is not None:
            combined_slates = pd.concat([remote_slates, local_slates], ignore_index=True)
        elif local_slates is not None:
            combined_slates = local_slates
        elif remote_slates is not None:
            combined_slates = remote_slates
        else:
            combined_slates = pd.DataFrame(columns=SLATE_RECAP_COLUMNS)

        self._slate_recaps_df = _clean_slate_recaps_df(combined_slates, valid_game_dates=valid_game_dates)
        if not self._slate_recaps_df.empty:
            try:
                self._slate_recaps_df.to_parquet(self.slate_recaps_file, index=False)
            except Exception:
                pass

    # =========================================================================
    # Idempotent Querying
    # =========================================================================

    def is_slate_recorded(self, game_date: str) -> bool:
        """Returns True if both the slate overview and its game recaps for `game_date` (YYYY-MM-DD) are cleanly recorded."""
        if self._slate_recaps_df is None or self._slate_recaps_df.empty:
            return False
        if self._game_recaps_df is None or self._game_recaps_df.empty:
            return False
        d_str = str(game_date)
        has_slate = (self._slate_recaps_df["game_date"].astype(str) == d_str).any()
        has_games = (self._game_recaps_df["game_date"].astype(str) == d_str).any()
        return bool(has_slate and has_games)

    def is_game_recorded(self, game_id: int) -> bool:
        """Returns True if `game_id` is already recorded in the dataset."""
        if self._game_recaps_df is None or self._game_recaps_df.empty:
            return False
        return (self._game_recaps_df["game_id"] == int(game_id)).any()

    def get_slate_record(self, game_date: str) -> Optional[Dict[str, Any]]:
        """Retrieves recorded slate recap dictionary for `game_date`."""
        if self._slate_recaps_df is None or self._slate_recaps_df.empty:
            return None
        match = self._slate_recaps_df[self._slate_recaps_df["game_date"] == str(game_date)]
        if match.empty:
            return None
        return match.iloc[0].to_dict()

    def is_slate_llm_generated(self, game_date: str) -> bool:
        """Returns True if the recorded slate for `game_date` was generated by an LLM (Tier 2)."""
        record = self.get_slate_record(game_date)
        if not record:
            return False
        return get_model_priority(record.get("generator_model")) >= 2

    def get_slate_generator_model(self, game_date: str) -> Optional[str]:
        """Returns the generator_model string for `game_date` if recorded, else None."""
        record = self.get_slate_record(game_date)
        if not record:
            return None
        return record.get("generator_model")

    def get_game_record(self, game_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves recorded game recap dictionary for `game_id`."""
        if self._game_recaps_df is None or self._game_recaps_df.empty:
            return None
        match = self._game_recaps_df[self._game_recaps_df["game_id"] == int(game_id)]
        if match.empty:
            return None
        return match.iloc[0].to_dict()

    def get_games_for_date(self, game_date: str) -> List[Dict[str, Any]]:
        """Retrieves all recorded game recaps for a specific date."""
        if self._game_recaps_df is None or self._game_recaps_df.empty:
            return []
        matches = self._game_recaps_df[self._game_recaps_df["game_date"] == str(game_date)]
        return matches.to_dict(orient="records")

    def get_recorded_dates(self) -> List[Dict[str, Any]]:
        """Returns all dates currently archived in the slate_recaps table."""
        if self._slate_recaps_df is None or self._slate_recaps_df.empty:
            return []
        sorted_df = self._slate_recaps_df.sort_values(by="game_date", ascending=False)
        return sorted_df[["game_date", "total_games", "total_goals", "slate_headline"]].to_dict(orient="records")

    def get_available_historical_dates(self, limit: int = 40) -> List[str]:
        """Returns recent dates that have games available in the Lakehouse or archive."""
        recorded = [d["game_date"] for d in self.get_recorded_dates()]
        
        # Also try pulling recent dates from data/games.parquet if needed
        try:
            url = f"{HF_BASE_URL}/data/games.parquet"
            df = pd.read_parquet(url, columns=["game_date"])
            dates = df["game_date"].dropna().unique().tolist()
            dates.sort(reverse=True)
            combined = list(dict.fromkeys(recorded + dates[:limit]))
            return combined[:limit]
        except Exception:
            return recorded

    # =========================================================================
    # Record-Keeping & Persistence
    # =========================================================================

    def record_slate(
        self,
        analyzed_slate,
        generator_model: str = "scoreboard-agent-v1",
        force: bool = False,
    ) -> None:
        """Records or updates a full slate analysis into `slate_recaps.parquet`.
        
        Model Precedence:
        - LLM-generated recaps (Tier 2) overwrite algorithmic summaries (Tier 1).
        - Algorithmic summaries will NOT overwrite existing LLM recaps unless force=True.
        """
        if int(getattr(analyzed_slate, "total_games", 0)) <= 0:
            return
        timestamp = datetime.now(timezone.utc).isoformat()
        
        three_stars_list = [
            {
                "star_num": s.star,
                "player_id": s.player_id,
                "name": s.name,
                "team": s.team_abbrev,
                "position": s.position,
                "stat_summary": s.stat_summary,
            }
            for s in getattr(analyzed_slate, "slate_three_stars", [])
        ]
        
        milestones_list = [
            {
                "category": m.category,
                "title": m.title,
                "player_or_team": m.entity,
                "description": m.description,
                "importance_score": getattr(m, "significance", "MEDIUM"),
            }
            for m in getattr(analyzed_slate, "slate_notables", [])
        ]

        headline = getattr(analyzed_slate, "executive_headline", "") or f"NHL Night in Review: {analyzed_slate.total_games} Games, {analyzed_slate.total_goals} Goals on {analyzed_slate.date}"
        narrator = NarratorEngine()
        narrative = getattr(analyzed_slate, "executive_summary", "") or narrator.generate_slate_markdown(analyzed_slate)
        actual_model = getattr(analyzed_slate, "generator_model", None) or generator_model

        row = {
            "game_date": str(analyzed_slate.date),
            "total_games": int(analyzed_slate.total_games),
            "total_goals": int(analyzed_slate.total_goals),
            "avg_goals_per_game": float(analyzed_slate.avg_goals_per_game),
            "home_wins": int(analyzed_slate.home_wins),
            "away_wins": int(analyzed_slate.away_wins),
            "overtime_games": int(analyzed_slate.ot_so_games),
            "three_stars_json": json.dumps(three_stars_list, ensure_ascii=False),
            "milestones_json": json.dumps(milestones_list, ensure_ascii=False),
            "slate_headline": headline,
            "slate_narrative": narrative,
            "generated_at": timestamp,
            "generator_model": actual_model,
        }

        new_row_df = pd.DataFrame([row])
        target_date_str = str(analyzed_slate.date)
        incoming_prio = get_model_priority(actual_model)

        if self._slate_recaps_df is None or self._slate_recaps_df.empty:
            self._slate_recaps_df = new_row_df
        else:
            existing_match = self._slate_recaps_df[self._slate_recaps_df["game_date"].astype(str) == target_date_str]
            if not existing_match.empty and not force:
                existing_model = str(existing_match.iloc[0].get("generator_model", ""))
                existing_prio = get_model_priority(existing_model)
                if incoming_prio < existing_prio:
                    print(
                        f"[HFScoreboardStore] Preserving existing higher-priority LLM slate recap for {target_date_str} "
                        f"({existing_model}) over incoming algorithmic summary ({actual_model}).",
                        file=sys.stderr,
                    )
                    return

            filtered = self._slate_recaps_df[self._slate_recaps_df["game_date"].astype(str) != target_date_str]
            self._slate_recaps_df = pd.concat([filtered, new_row_df], ignore_index=True)

        self._slate_recaps_df = _clean_slate_recaps_df(self._slate_recaps_df)
        self._slate_recaps_df.to_parquet(self.slate_recaps_file, index=False)
        if self.token and HF_HUB_AVAILABLE:
            try:
                self.sync_to_huggingface()
            except Exception as e:
                print(f"[HFScoreboardStore] Auto-sync slate warning: {e}", file=sys.stderr)

    def record_games(
        self,
        analyzed_games: list,
        generator_model: str = "scoreboard-agent-v1",
        force: bool = False,
    ) -> None:
        """Records or updates analyzed games into `game_recaps.parquet`.
        
        Model Precedence:
        - LLM-generated recaps (Tier 2) overwrite algorithmic summaries (Tier 1).
        - Algorithmic summaries will NOT overwrite existing LLM recaps unless force=True.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        narrator = NarratorEngine()
        rows = []

        for g in analyzed_games:
            full_md = getattr(g, "full_report_markdown", "") or narrator.generate_single_game_markdown(g)
            milestones_list = [
                {
                    "category": m.category,
                    "title": m.title,
                    "player_or_team": m.entity,
                    "description": m.description,
                    "importance_score": getattr(m, "significance", "MEDIUM"),
                }
                for m in getattr(g, "notables", [])
            ]
            three_stars_list = [
                {
                    "star_num": s.star,
                    "player_id": s.player_id,
                    "name": s.name,
                    "team": s.team_abbrev,
                    "position": s.position,
                    "stat_summary": s.stat_summary,
                }
                for s in getattr(g, "three_stars", [])
            ]
            away_stats_dict = {
                "sog": getattr(g.away_team, "sog", 0),
                "power_play_pct": getattr(g.away_team, "power_play_pct", 0.0),
                "faceoff_pct": getattr(g.away_team, "faceoff_pct", 0.0),
                "hits": getattr(g.away_team, "hits", 0),
                "blocked_shots": getattr(g.away_team, "blocked_shots", 0),
                "giveaways": getattr(g.away_team, "giveaways", 0),
                "takeaways": getattr(g.away_team, "takeaways", 0),
            }
            home_stats_dict = {
                "sog": getattr(g.home_team, "sog", 0),
                "power_play_pct": getattr(g.home_team, "power_play_pct", 0.0),
                "faceoff_pct": getattr(g.home_team, "faceoff_pct", 0.0),
                "hits": getattr(g.home_team, "hits", 0),
                "blocked_shots": getattr(g.home_team, "blocked_shots", 0),
                "giveaways": getattr(g.home_team, "giveaways", 0),
                "takeaways": getattr(g.home_team, "takeaways", 0),
            }

            # Period flow serialization
            flow_dict = {}
            if getattr(g, "period_flow", None):
                flow_dict = {
                    "lead_changes": g.period_flow.lead_changes,
                    "ties_count": g.period_flow.ties_count,
                    "max_winner_lead": g.period_flow.max_winner_lead,
                    "max_loser_lead": g.period_flow.max_loser_lead,
                    "score_after_p1": g.period_flow.score_after_p1,
                    "score_after_p2": g.period_flow.score_after_p2,
                    "score_entering_p3": g.period_flow.score_entering_p3,
                    "winner_trailed_entering_p3": g.period_flow.winner_trailed_entering_p3,
                    "game_script_type": g.period_flow.game_script_type,
                    "script_summary": g.period_flow.script_summary,
                }

            # Defense effort serialization
            def_dict = {}
            if getattr(g, "away_defense_profile", None) and getattr(g, "home_defense_profile", None):
                def_dict = {
                    "away": {
                        "team": g.away_defense_profile.team_abbrev,
                        "blocked_shots": g.away_defense_profile.blocked_shots,
                        "blocks_rating": g.away_defense_profile.blocks_rating,
                        "hits": g.away_defense_profile.hits,
                        "hits_rating": g.away_defense_profile.hits_rating,
                        "giveaways": g.away_defense_profile.giveaways,
                        "takeaways": g.away_defense_profile.takeaways,
                        "turnover_diff": g.away_defense_profile.turnover_diff,
                        "shots_allowed": g.away_defense_profile.shots_allowed,
                        "shot_suppression": g.away_defense_profile.shot_suppression,
                    },
                    "home": {
                        "team": g.home_defense_profile.team_abbrev,
                        "blocked_shots": g.home_defense_profile.blocked_shots,
                        "blocks_rating": g.home_defense_profile.blocks_rating,
                        "hits": g.home_defense_profile.hits,
                        "hits_rating": g.home_defense_profile.hits_rating,
                        "giveaways": g.home_defense_profile.giveaways,
                        "takeaways": g.home_defense_profile.takeaways,
                        "turnover_diff": g.home_defense_profile.turnover_diff,
                        "shots_allowed": g.home_defense_profile.shots_allowed,
                        "shot_suppression": g.home_defense_profile.shot_suppression,
                    }
                }

            # Goalie performance serialization
            goalie_dict = {}
            if getattr(g, "winning_goalie_profile", None):
                goalie_dict["winner"] = {
                    "name": g.winning_goalie_profile.name,
                    "team": g.winning_goalie_profile.team_abbrev,
                    "decision": g.winning_goalie_profile.decision,
                    "saves": g.winning_goalie_profile.saves,
                    "shots_against": g.winning_goalie_profile.shots_against,
                    "goals_against": g.winning_goalie_profile.goals_against,
                    "save_pct": g.winning_goalie_profile.save_pct,
                    "rating": g.winning_goalie_profile.performance_rating,
                    "assessment": g.winning_goalie_profile.assessment,
                }
            if getattr(g, "losing_goalie_profile", None):
                goalie_dict["loser"] = {
                    "name": g.losing_goalie_profile.name,
                    "team": g.losing_goalie_profile.team_abbrev,
                    "decision": g.losing_goalie_profile.decision,
                    "saves": g.losing_goalie_profile.saves,
                    "shots_against": g.losing_goalie_profile.shots_against,
                    "goals_against": g.losing_goalie_profile.goals_against,
                    "save_pct": g.losing_goalie_profile.save_pct,
                    "rating": g.losing_goalie_profile.performance_rating,
                    "assessment": g.losing_goalie_profile.assessment,
                }

            # Season standings context serialization
            season_dict = {}
            for side, ctx in [("away", getattr(g, "away_season_context", None)), ("home", getattr(g, "home_season_context", None))]:
                if ctx:
                    season_dict[side] = {
                        "team": ctx.team_abbrev,
                        "gp": ctx.games_played,
                        "record": f"{ctx.wins}-{ctx.losses}-{ctx.ot_losses}",
                        "pts": ctx.points,
                        "pts_pct": ctx.points_pct,
                        "gf": ctx.goals_for,
                        "ga": ctx.goals_against,
                        "goal_diff": ctx.goal_diff,
                        "pyth_pct": ctx.pythagorean_pct,
                        "pyth_diff": ctx.pythagorean_diff,
                        "stage": ctx.season_stage,
                        "stage_narrative": ctx.stage_narrative,
                    }

            # Validation report serialization
            val_dict = {}
            if getattr(g, "validation_report", None):
                val_dict = {
                    "is_valid": g.validation_report.is_valid,
                    "goals_reconciled": g.validation_report.goals_reconciled,
                    "shots_reconciled": g.validation_report.shots_reconciled,
                    "period_math_reconciled": g.validation_report.period_math_reconciled,
                    "audit_notes": g.validation_report.audit_notes,
                }

            rows.append({
                "game_id": int(g.game_id),
                "game_date": str(g.game_date),
                "season": getattr(g, "season", 20262027),
                "away_team": str(g.away_team.team_abbrev),
                "away_score": int(g.away_team.score),
                "home_team": str(g.home_team.team_abbrev),
                "home_score": int(g.home_team.score),
                "period_ending": str(getattr(g, "period_type", "REG")),
                "headline": str(g.headline),
                "narrative_recap": str(g.recap_narrative),
                "full_report_markdown": str(full_md),
                "milestones_json": json.dumps(milestones_list, ensure_ascii=False),
                "three_stars_json": json.dumps(three_stars_list, ensure_ascii=False),
                "away_stats_json": json.dumps(away_stats_dict, ensure_ascii=False),
                "home_stats_json": json.dumps(home_stats_dict, ensure_ascii=False),
                "game_script_type": flow_dict.get("game_script_type", "REGULATION"),
                "period_flow_json": json.dumps(flow_dict, ensure_ascii=False),
                "defensive_effort_json": json.dumps(def_dict, ensure_ascii=False),
                "goalie_performance_json": json.dumps(goalie_dict, ensure_ascii=False),
                "season_standings_json": json.dumps(season_dict, ensure_ascii=False),
                "validation_report_json": json.dumps(val_dict, ensure_ascii=False),
                "generated_at": timestamp,
                "generator_model": getattr(g, "generator_model", None) or generator_model,
            })

        if not rows:
            return

        new_df = pd.DataFrame(rows)
        if self._game_recaps_df is None or self._game_recaps_df.empty:
            self._game_recaps_df = new_df
        else:
            existing_prios: Dict[int, int] = {}
            if "game_id" in self._game_recaps_df.columns:
                existing_models = (
                    self._game_recaps_df.set_index("game_id")["generator_model"].to_dict()
                    if "generator_model" in self._game_recaps_df.columns
                    else {}
                )
                existing_prios = {int(gid): get_model_priority(m) for gid, m in existing_models.items()}

            rows_to_keep_from_new = []
            for _, r in new_df.iterrows():
                gid = int(r["game_id"])
                inc_prio = get_model_priority(r.get("generator_model"))
                ext_prio = existing_prios.get(gid, -1)
                if not force and ext_prio > inc_prio:
                    # Existing has higher priority (e.g. LLM vs algorithmic)
                    print(
                        f"[HFScoreboardStore] Preserving existing LLM recap for game {gid} "
                        f"over incoming lower-priority summary.",
                        file=sys.stderr,
                    )
                    continue
                rows_to_keep_from_new.append(r)

            if rows_to_keep_from_new:
                accepted_new_df = pd.DataFrame(rows_to_keep_from_new)
                gids_to_replace = set(accepted_new_df["game_id"].astype(int))
                filtered_existing = self._game_recaps_df[~self._game_recaps_df["game_id"].astype(int).isin(gids_to_replace)]
                self._game_recaps_df = pd.concat([filtered_existing, accepted_new_df], ignore_index=True)

        self._game_recaps_df = _clean_game_recaps_df(self._game_recaps_df)
        self._game_recaps_df.to_parquet(self.game_recaps_file, index=False)
        if self.token and HF_HUB_AVAILABLE:
            try:
                self.sync_to_huggingface()
            except Exception as e:
                print(f"[HFScoreboardStore] Auto-sync games warning: {e}", file=sys.stderr)

    # =========================================================================
    # Hugging Face Synchronization & Diagnostics
    # =========================================================================

    def ping_huggingface(self, token: Optional[str] = None) -> Dict[str, Any]:
        """Pings Hugging Face API to verify connectivity, dataset existence, and token auth.
        
        Credentials are never logged or returned directly.
        """
        auth_token = token or self.token
        result: Dict[str, Any] = {
            "status": "unknown",
            "repo_id": self.repo_id,
            "has_token": bool(auth_token),
            "authenticated_user": None,
            "repo_exists": False,
            "can_write": False,
            "error": None,
        }

        if not HF_HUB_AVAILABLE:
            result["status"] = "error"
            result["error"] = "huggingface_hub package is not installed."
            return result

        try:
            api = HfApi(token=auth_token)
            
            # Check authentication
            if auth_token:
                try:
                    user_info = api.whoami()
                    result["authenticated_user"] = user_info.get("name")
                    result["can_write"] = True
                except Exception as e:
                    result["error"] = f"Token validation failed: {e}"

            # Check repo existence
            dataset_info = api.dataset_info(repo_id=self.repo_id)
            result["repo_exists"] = True
            result["last_modified"] = str(dataset_info.lastModified)
            result["status"] = "ready" if result["can_write"] else "read_only"
            return result

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            return result

    def sync_to_huggingface(
        self,
        token: Optional[str] = None,
        commit_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Pushes local `game_recaps.parquet` and `slate_recaps.parquet` to the Hugging Face dataset."""
        auth_token = token or self.token
        if not auth_token:
            return {
                "success": False,
                "error": "No HF_TOKEN provided. Cannot sync to Hugging Face dataset.",
            }

        if not HF_HUB_AVAILABLE:
            return {
                "success": False,
                "error": "huggingface_hub package is not installed.",
            }

        api = HfApi(token=auth_token)
        uploaded_files = []
        msg = commit_message or f"Update Scoreboard recaps: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"

        try:
            if self.game_recaps_file.exists():
                path_in_repo = f"{self.remote_subdir}/game_recaps.parquet"
                api.upload_file(
                    path_or_fileobj=str(self.game_recaps_file),
                    path_in_repo=path_in_repo,
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    commit_message=f"{msg} (game_recaps.parquet)",
                )
                uploaded_files.append(path_in_repo)

            if self.slate_recaps_file.exists():
                path_in_repo = f"{self.remote_subdir}/slate_recaps.parquet"
                api.upload_file(
                    path_or_fileobj=str(self.slate_recaps_file),
                    path_in_repo=path_in_repo,
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    commit_message=f"{msg} (slate_recaps.parquet)",
                )
                uploaded_files.append(path_in_repo)

            return {
                "success": True,
                "repo_id": self.repo_id,
                "uploaded_files": uploaded_files,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }
