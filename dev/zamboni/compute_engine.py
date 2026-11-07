"""
compute_engine.py — Deterministic Python / Pandas / DuckDB Compute Skills for Agent Zamboni.

Queries the Hugging Face Dataset Lakehouse (`hf://datasets/jjerlee/nhl-analytics`)
with seamless local fallback (`latest/*` and `dev/*`) to answer quantitative user
questions with 100% mathematical accuracy and zero LLM mental-math hallucination.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .config import HF_DATASET_BASE
from .env_utils import get_project_root
from .models import ComputeSkillResult

# Ensure app/exe is importable for unified data_loader & player_stats reuse
_REPO_ROOT = get_project_root()
_APP_EXE = _REPO_ROOT / "app" / "exe"
if str(_APP_EXE) not in sys.path:
    sys.path.insert(0, str(_APP_EXE))
if str(_REPO_ROOT / "app") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "app"))


class ZamboniComputeEngine:
    """
    Executes deterministic data queries and statistical calculations over
    Rinklytics Hugging Face datasets (`jjerlee/nhl-analytics`) and local panels.
    """

    def __init__(self):
        self.root: Path = get_project_root()
        self._skaters_cache: Optional[pd.DataFrame] = None
        self._goalies_cache: Optional[pd.DataFrame] = None
        self._team_season_cache: Optional[pd.DataFrame] = None

    # ── Data Access Helpers ───────────────────────────────────────────────────

    def _get_player_season(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self._skaters_cache is not None and self._goalies_cache is not None:
            return self._skaters_cache, self._goalies_cache

        try:
            from data_loader import load_player_season_dataset
            sk, gl = load_player_season_dataset()
            if sk is not None and not sk.empty:
                self._skaters_cache, self._goalies_cache = sk, gl
                return sk, gl
        except Exception:
            pass

        # Direct HF / Local fallback
        try:
            df = pd.read_parquet(f"{HF_DATASET_BASE}/measurement/player/player_season.parquet")
        except Exception:
            import glob
            files = sorted(glob.glob(str(self.root / "latest" / "player" / "*_player.csv")))
            frames = []
            for f in files:
                yr = os.path.basename(f).split("_")[0]
                tdf = pd.read_csv(f)
                tdf["season_year"] = int(yr)
                tdf["season_label"] = f"{yr}-{str(int(yr)+1)[2:]}"
                frames.append(tdf)
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        if "season_year" not in df.columns:
            yr_col = "yr_season" if "yr_season" in df.columns else "idx_season"
            df["season_year"] = df[yr_col].astype(int)
            df["season_label"] = df["season_year"].astype(str) + "-" + (df["season_year"] + 1).astype(str).str[2:]

        df.rename(columns={"firstName.default": "firstName", "lastName.default": "lastName"}, inplace=True)
        if "fullName" not in df.columns:
            df["fullName"] = (df.get("firstName", pd.Series(dtype=str)).fillna("") + " " + df.get("lastName", pd.Series(dtype=str)).fillna("")).str.strip()

        if "idx_season_type" in df.columns:
            df = df[df["idx_season_type"].fillna(2) == 2].copy()

        for col in [
            "gamesPlayed", "goals", "assists", "points", "plusMinus", "shots", "shootingPctg",
            "avgTimeOnIcePerGame", "ixg", "ixg_per_game", "goals_above_expected", "hd_shots",
            "gamesStarted", "wins", "losses", "overtimeLosses", "goalsAgainstAverage",
            "savePercentage", "saves", "shotsAgainst", "shutouts", "xga", "gsax", "xsave_pct", "dsave_pct", "hd_save_pct",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        sk = df[df["positionCode"].isin(["C", "L", "R", "D"])].copy()
        gl = df[df["positionCode"] == "G"].copy()

        sk["points_per_game"] = sk["points"] / sk["gamesPlayed"].replace(0, np.nan)
        sk["goals_per_game"] = sk["goals"] / sk["gamesPlayed"].replace(0, np.nan)
        sk["assists_per_game"] = sk["assists"] / sk["gamesPlayed"].replace(0, np.nan)
        sk["shots_per_game"] = sk["shots"] / sk["gamesPlayed"].replace(0, np.nan)
        if "ixg" in sk.columns and "ixg_per_game" not in sk.columns:
            sk["ixg_per_game"] = sk["ixg"] / sk["gamesPlayed"].replace(0, np.nan)

        gl["saves_per_game"] = gl["saves"] / gl["gamesStarted"].replace(0, np.nan)
        gl["win_pct"] = gl["wins"] / gl["gamesPlayed"].replace(0, np.nan)

        self._skaters_cache, self._goalies_cache = sk, gl
        return sk, gl

    def _get_team_season(self) -> pd.DataFrame:
        if self._team_season_cache is not None and not self._team_season_cache.empty:
            return self._team_season_cache
        try:
            from data_loader import load_team_season_dataset
            df = load_team_season_dataset()
            if df is not None and not df.empty:
                self._team_season_cache = df
                return df
        except Exception:
            pass
        return pd.DataFrame()

    # ── Skill 1: Skater PPI & Advanced Stats ──────────────────────────────────

    def query_skater_stats(self, entities: Dict[str, Any]) -> Optional[ComputeSkillResult]:
        skaters, _ = self._get_player_season()
        if skaters is None or skaters.empty:
            return None

        avail_years = sorted(skaters["season_year"].dropna().unique().astype(int).tolist())
        target_yr = entities.get("season_year")
        if target_yr not in avail_years:
            target_yr = avail_years[-1]

        teams = entities.get("teams", [])
        raw_q = (entities.get("raw_query") or "").lower()
        top_n = min(entities.get("top_n", 10), 15)

        try:
            from player_stats import compute_skater_ppi
            ppi_df = compute_skater_ppi(skaters, season_year=target_yr, min_games=15)
        except Exception:
            ppi_df = pd.DataFrame()

        if ppi_df.empty:
            ppi_df = skaters[skaters["season_year"] == target_yr].copy()

        # Check if specific player name is mentioned in the query
        matched_players = []
        for name in ppi_df["fullName"].dropna().unique():
            last = name.split()[-1].lower()
            if len(last) >= 4 and last in raw_q:
                matched_players.append(name)

        if matched_players:
            sub = ppi_df[ppi_df["fullName"].isin(matched_players)].copy()
        elif teams:
            sub = ppi_df[ppi_df["team_tri"].isin(teams)].copy()
        else:
            sub = ppi_df.copy()

        metric = entities.get("metric", "PPI")
        sort_col = metric if metric in sub.columns else ("PPI" if "PPI" in sub.columns else "points")
        sub = sub.sort_values(sort_col, ascending=False).head(top_n).reset_index(drop=True)

        cols = [
            c for c in [
                "fullName", "team_tri", "positionCode", "gamesPlayed",
                "goals", "assists", "points", "ixg", "goals_above_expected", "hd_shots", "PPI",
            ]
            if c in sub.columns
        ]
        sub_disp = sub[cols].copy()
        for c in ["ixg", "goals_above_expected", "PPI"]:
            if c in sub_disp.columns:
                sub_disp[c] = pd.to_numeric(sub_disp[c], errors="coerce").round(2)

        season_str = f"{target_yr}-{str(target_yr + 1)[2:]}"
        lines = [f"**Skater Analytics & PPI Leaderboard ({season_str}, sorted by `{sort_col}`):**"]
        for idx, row in sub_disp.head(8).iterrows():
            ppi_str = f" | PPI: `{row['PPI']:.3f}`" if "PPI" in row and pd.notna(row["PPI"]) else ""
            ixg_str = f" | ixG: `{row['ixg']:.2f}` (G-ixG: `{row.get('goals_above_expected', 0):+.2f}`)" if "ixg" in row and pd.notna(row["ixg"]) else ""
            lines.append(
                f"{idx + 1}. **{row['fullName']}** ({row.get('team_tri', '')}, {row.get('positionCode', '')}) — "
                f"GP: `{int(row.get('gamesPlayed', 0))}`, G: `{int(row.get('goals', 0))}`, "
                f"A: `{int(row.get('assists', 0))}`, PTS: `{int(row.get('points', 0))}`{ixg_str}{ppi_str}"
            )

        return ComputeSkillResult(
            skill_name="skill_skater_analytics",
            title=f"NHL Skater Season & PPI Telemetry ({season_str})",
            dataset_source="hf://datasets/jjerlee/nhl-analytics/measurement/player/player_season.parquet",
            summary_markdown="\n".join(lines),
            records=sub_disp.to_dict(orient="records"),
            columns=cols,
            query_params={"season_year": target_yr, "sort_by": sort_col, "teams": teams},
        )

    # ── Skill 2: Goalie PPI & GSAx Crease Analytics ───────────────────────────

    def query_goalie_stats(self, entities: Dict[str, Any]) -> Optional[ComputeSkillResult]:
        _, goalies = self._get_player_season()
        if goalies is None or goalies.empty:
            return None

        avail_years = sorted(goalies["season_year"].dropna().unique().astype(int).tolist())
        target_yr = entities.get("season_year")
        if target_yr not in avail_years:
            target_yr = avail_years[-1]

        teams = entities.get("teams", [])
        raw_q = (entities.get("raw_query") or "").lower()
        top_n = min(entities.get("top_n", 10), 15)

        try:
            from player_stats import compute_goalie_ppi
            ppi_df = compute_goalie_ppi(goalies, season_year=target_yr, min_games=10)
        except Exception:
            ppi_df = pd.DataFrame()

        if ppi_df.empty:
            ppi_df = goalies[goalies["season_year"] == target_yr].copy()

        matched_goalies = []
        for name in ppi_df["fullName"].dropna().unique():
            last = name.split()[-1].lower()
            if len(last) >= 4 and (last in raw_q or last.replace("š", "s") in raw_q):
                matched_goalies.append(name)

        if matched_goalies:
            sub = ppi_df[ppi_df["fullName"].isin(matched_goalies)].copy()
        elif teams:
            sub = ppi_df[ppi_df["team_tri"].isin(teams)].copy()
        else:
            sub = ppi_df.copy()

        metric = entities.get("metric", "PPI")
        sort_col = metric if metric in sub.columns else ("gsax" if "gsax" in sub.columns and "gsax" in raw_q else ("PPI" if "PPI" in sub.columns else "savePercentage"))
        sub = sub.sort_values(sort_col, ascending=False).head(top_n).reset_index(drop=True)

        cols = [
            c for c in [
                "fullName", "team_tri", "gamesPlayed", "wins", "savePercentage",
                "goalsAgainstAverage", "xga", "gsax", "xsave_pct", "dsave_pct", "PPI",
            ]
            if c in sub.columns
        ]
        sub_disp = sub[cols].copy()

        season_str = f"{target_yr}-{str(target_yr + 1)[2:]}"
        lines = [f"**Goaltender GSAx & PPI Leaderboard ({season_str}, sorted by `{sort_col}`):**"]
        for idx, row in sub_disp.head(8).iterrows():
            gsax_str = f" | GSAx: `{row['gsax']:+.2f}` (dSV%: `{row.get('dsave_pct', 0):+.3f}`)" if "gsax" in row and pd.notna(row["gsax"]) else ""
            ppi_str = f" | PPI: `{row['PPI']:.3f}`" if "PPI" in row and pd.notna(row["PPI"]) else ""
            sv_val = row.get("savePercentage", 0.0)
            gaa_val = row.get("goalsAgainstAverage", 0.0)
            lines.append(
                f"{idx + 1}. **{row['fullName']}** ({row.get('team_tri', '')}) — "
                f"GP: `{int(row.get('gamesPlayed', 0))}`, W: `{int(row.get('wins', 0))}`, "
                f"SV%: `{sv_val:.3f}`, GAA: `{gaa_val:.2f}`{gsax_str}{ppi_str}"
            )

        return ComputeSkillResult(
            skill_name="skill_goalie_analytics",
            title=f"NHL Goaltender GSAx & PPI Telemetry ({season_str})",
            dataset_source="hf://datasets/jjerlee/nhl-analytics/measurement/player/player_season.parquet",
            summary_markdown="\n".join(lines),
            records=sub_disp.to_dict(orient="records"),
            columns=cols,
            query_params={"season_year": target_yr, "sort_by": sort_col, "teams": teams},
        )

    # ── Skill 3: Team Season KPIs (RPE, RPI, Pairwise, xG, GSAx) ──────────────

    def query_team_season_kpis(self, entities: Dict[str, Any]) -> Optional[ComputeSkillResult]:
        df = self._get_team_season()
        if df is None or df.empty:
            return None

        avail_seasons = sorted(df["Season"].astype(str).unique().tolist())
        req_yr = str(entities.get("season_year")) if entities.get("season_year") else avail_seasons[-1]
        if req_yr not in avail_seasons:
            req_yr = avail_seasons[-1]

        sub = df[df["Season"].astype(str) == req_yr].copy()
        teams = entities.get("teams", [])
        if teams:
            sub = sub[sub["team"].isin(teams)].copy()

        metric = entities.get("metric", "PTS")
        sort_col = metric if metric in sub.columns else ("PTS" if "PTS" in sub.columns else "wp")
        ascending = True if sort_col == "pairwise_win" else False
        sub = sub.sort_values(sort_col, ascending=ascending).head(entities.get("top_n", 10)).reset_index(drop=True)

        cols = [
            c for c in [
                "team", "GP", "W", "L", "OTL", "PTS", "wp", "rpe", "rwin_fittedPE",
                "rpi", "pairwise_win", "xg_pct", "hdcf_pct", "gsax", "kpi_pe", "kpi_pairwise", "kpi_xg",
            ]
            if c in sub.columns
        ]
        sub_disp = sub[cols].copy()

        season_str = f"{req_yr}-{str(int(req_yr) + 1)[2:]}"
        lines = [f"**Team Season KPIs & Econometric Benchmarks ({season_str}, sorted by `{sort_col}`):**"]
        for idx, row in sub_disp.head(8).iterrows():
            xg_str = f" | xGF%: `{row['xg_pct']:.3f}` | GSAx: `{row.get('gsax', 0):+.1f}`" if "xg_pct" in row and pd.notna(row["xg_pct"]) else ""
            lines.append(
                f"{idx + 1}. **{row['team']}** — PTS: `{int(row.get('PTS', 0))}` ({int(row.get('W', 0))}-{int(row.get('L', 0))}-{int(row.get('OTL', 0))}), "
                f"WP: `{row.get('wp', 0):.3f}`, RPE (Pythagorean): `{row.get('rpe', 0):.3f}`, "
                f"Pairwise Rank: `#{int(row.get('pairwise_win', 0))}`{xg_str}"
            )

        return ComputeSkillResult(
            skill_name="skill_team_season_kpis",
            title=f"Team Season Success KPIs ({season_str})",
            dataset_source="hf://datasets/jjerlee/nhl-analytics/measurement/box/team_season.parquet",
            summary_markdown="\n".join(lines),
            records=sub_disp.to_dict(orient="records"),
            columns=cols,
            query_params={"season_year": req_yr, "sort_by": sort_col, "teams": teams},
        )

    # ── Skill 4: Franchise Rebuild Tracker (2000–2025) ────────────────────────

    def query_rebuild_regimes(self, entities: Dict[str, Any]) -> Optional[ComputeSkillResult]:
        rebuild_tables = self.root / "dev" / "player_team_success" / "rebuild" / "results" / "tables"
        processed_dir = self.root / "dev" / "player_team_success" / "rebuild" / "data" / "processed"

        # Check MTL vs CHI comparison table or panel
        panel_file = processed_dir / "panel_franchise_season.csv"
        comp_file = rebuild_tables / "table4_mtl_vs_chi_rebuild_comparison.csv"
        levers_file = rebuild_tables / "table7_gm_actionable_policy_levers.csv"

        teams = entities.get("teams", [])
        raw_q = (entities.get("raw_query") or "").lower()

        if ("lever" in raw_q or "policy" in raw_q or "simulator" in raw_q) and levers_file.exists():
            df_l = pd.read_csv(levers_file)
            lines = ["**GM Actionable Policy Levers (25-Year Rebuild Study):**"]
            for _, r in df_l.head(6).iterrows():
                lines.append(
                    f"- **Rank #{r.get('priority_rank', '')}: {r.get('gm_lever_name', '')}** ({r.get('quirky_rule_name', '')}) — "
                    f"Low setting peak odds: `{r.get('low_setting_peak_odds_pct', '')}%` -> Optimized 90th-pct odds: `{r.get('high_setting_peak_odds_pct', '')}%`"
                )
            return ComputeSkillResult(
                skill_name="skill_rebuild_econometrics",
                title="Franchise Rebuild GM Policy Levers (2000–2025)",
                dataset_source="dev/player_team_success/rebuild/results/tables/table7_gm_actionable_policy_levers.csv",
                summary_markdown="\n".join(lines),
                records=df_l.to_dict(orient="records"),
                columns=list(df_l.columns),
            )

        if panel_file.exists():
            df_p = pd.read_csv(panel_file)
            if teams:
                df_sub = df_p[df_p["franchise_id"].isin(teams)].sort_values(["franchise_id", "season"], ascending=[True, False]).head(10)
            else:
                latest_s = df_p["season"].max()
                df_sub = df_p[df_p["season"] == latest_s].sort_values("pts_pct", ascending=False).head(10)

            cols = [
                c for c in [
                    "franchise_id", "season", "pts_pct", "macro_regime", "rebuild_regime",
                    "ali_score", "yis_score", "s_b1_elc_0_3", "s_b2_prime_4_6",
                    "shannon_exp_entropy", "spsi_style_synergy", "surplus_value_total",
                ]
                if c in df_sub.columns
            ]
            sub_disp = df_sub[cols].copy()
            lines = ["**Franchise Rebuild Regime & Cohort Telemetry (2000–2025 Panel):**"]
            for _, r in sub_disp.head(8).iterrows():
                lines.append(
                    f"- **{r.get('franchise_id')} ({r.get('season')})**: Macro=`{r.get('macro_regime')}`, "
                    f"State=`{r.get('rebuild_regime')}`, Pts%=`{r.get('pts_pct', 0):.3f}`, "
                    f"YIS=`{r.get('yis_score', 0):.2f}`, ALI=`{r.get('ali_score', 0):.2f}`, "
                    f"Cap Surplus=`${r.get('surplus_value_total', 0):.1f}M`"
                )
            return ComputeSkillResult(
                skill_name="skill_rebuild_econometrics",
                title="Franchise Rebuild & Equilibrium Regime Panel",
                dataset_source="dev/player_team_success/rebuild/data/processed/panel_franchise_season.csv",
                summary_markdown="\n".join(lines),
                records=sub_disp.to_dict(orient="records"),
                columns=cols,
                query_params={"teams": teams},
            )

        if comp_file.exists():
            df_c = pd.read_csv(comp_file)
            return ComputeSkillResult(
                skill_name="skill_rebuild_econometrics",
                title="Montréal vs. Chicago Rebuild Comparison",
                dataset_source=str(comp_file),
                summary_markdown="Loaded Montréal vs. Chicago structural rebuild comparison table.",
                records=df_c.to_dict(orient="records"),
                columns=list(df_c.columns),
            )
        return None

    # ── Skill 5: Hit on Impact Sequential & Spatial Analytics ─────────────────

    def query_hit_impact(self, entities: Dict[str, Any]) -> Optional[ComputeSkillResult]:
        try:
            from data_loader import load_hit_analytics_datasets
            res = load_hit_analytics_datasets()
            df_zone_kpi, df_seq, df_goals, df_grid, df_team_kpi = res[0], res[1], res[2], res[3], res[4]
        except Exception:
            df_zone_kpi, df_team_kpi = None, None

        if df_zone_kpi is not None and not df_zone_kpi.empty:
            lines = ["**Hit on Impact — Zone Summary KPIs (`hit_zone_summary_kpi.parquet`):**"]
            for _, r in df_zone_kpi.head(6).iterrows():
                row_str = ", ".join(f"{k}: `{v}`" for k, v in r.items() if pd.notna(v))
                lines.append(f"- {row_str}")
            return ComputeSkillResult(
                skill_name="skill_hit_impact",
                title="Hit on Impact — Zone & Sequence KPIs",
                dataset_source="hf://datasets/jjerlee/nhl-analytics/measurement/action/hits/hit_zone_summary_kpi.parquet",
                summary_markdown="\n".join(lines),
                records=df_zone_kpi.head(10).to_dict(orient="records"),
                columns=list(df_zone_kpi.columns),
            )
        return None

    # ── Skill 6: Shot Blocking Sequential Transitions ─────────────────────────

    def query_blocking_shots(self) -> Optional[ComputeSkillResult]:
        try:
            from data_loader import load_blocking_shots_dataset
            df = load_blocking_shots_dataset()
        except Exception:
            df = pd.DataFrame()

        if df is None or df.empty:
            return None

        total = df["seasonIdx"].sum()
        by_owner = df.groupby(["blocker_team_lag1_now", "eventOwnerTeam_sequence"])["seasonIdx"].sum().reset_index()
        by_owner["share_pct"] = (by_owner["seasonIdx"] / max(total, 1) * 100).round(2)

        lines = [f"**Shot Blocking Sequential Transitions ($N = {int(total):,}$ blocked shot sequences):**"]
        for _, r in by_owner.iterrows():
            lines.append(
                f"- Prior Zone: **{r['blocker_team_lag1_now']}** $\\to$ Next Event Owner: **{r['eventOwnerTeam_sequence']}** — "
                f"`{int(r['seasonIdx']):,}` sequences (`{r['share_pct']:.2f}%` of total)"
            )
        return ComputeSkillResult(
            skill_name="skill_blocking_shots",
            title="Shot Blocking Downstream Possession Transitions",
            dataset_source="dev/decision_science/plays_blockingShots/result/table - sequence blocked shot stats.csv",
            summary_markdown="\n".join(lines),
            records=by_owner.to_dict(orient="records"),
            columns=list(by_owner.columns),
        )

    # ── Skill 7: Draft Lifecycles & Travel Map ────────────────────────────────

    def query_draft_and_map(self, entities: Dict[str, Any], domains: List[str]) -> Optional[ComputeSkillResult]:
        if "map_travel" in domains:
            try:
                from data_loader import load_map_dataset
                df_map = load_map_dataset()
            except Exception:
                df_map = None
            if df_map is not None and not df_map.empty:
                teams = entities.get("teams", [])
                if len(teams) >= 2:
                    sub = df_map[
                        (df_map["tricode_home"] == teams[0]) & (df_map["tricode_away"] == teams[1])
                    ].copy()
                elif len(teams) == 1:
                    sub = df_map[df_map["tricode_home"] == teams[0]].sort_values("distance_meters", ascending=False).head(8)
                else:
                    sub = df_map.sort_values("distance_meters", ascending=False).head(8)

                sub["distance_km"] = (sub["distance_meters"] / 1000.0).round(1)
                sub["distance_miles"] = (sub["distance_meters"] * 0.000621371).round(1)
                cols = [c for c in ["tricode_home", "home_team_city", "tricode_away", "away_team_city", "distance_km", "distance_miles"] if c in sub.columns]
                lines = ["**NHL Inter-Arena Geodesic Travel Distances:**"]
                for _, r in sub[cols].head(6).iterrows():
                    lines.append(
                        f"- **{r.get('tricode_home')} ({r.get('home_team_city')}) $\\leftrightarrow$ {r.get('tricode_away')} ({r.get('away_team_city')})**: "
                        f"`{r.get('distance_km'):,.1f} km` (`{r.get('distance_miles'):,.1f} miles`)"
                    )
                return ComputeSkillResult(
                    skill_name="skill_map_travel",
                    title="NHL Arena Travel Distance Matrix",
                    dataset_source="app/data/teamlist_locations_distance_meters.csv",
                    summary_markdown="\n".join(lines),
                    records=sub[cols].to_dict(orient="records"),
                    columns=cols,
                )

        if "draft" in domains:
            try:
                from data_loader import load_draft_dataset
                ok, d_dict = load_draft_dataset()
            except Exception:
                ok, d_dict = False, {}
            if ok and "exceptional_players" in d_dict:
                df_exc = d_dict["exceptional_players"].sort_values("exceptional_seasons", ascending=False).head(10)
                lines = ["**1st-Round Draft Picks with Most 95th-Percentile Exceptional Seasons (`Magnifiques Patineuses`):**"]
                for idx, r in df_exc.iterrows():
                    lines.append(
                        f"- **{r.get('firstName', '')} {r.get('lastName', '')}** (Overall Pick `#{r.get('overallPick', '')}`) — "
                        f"`{int(r.get('exceptional_seasons', 0))}` exceptional seasons (>= 95th percentile)"
                    )
                return ComputeSkillResult(
                    skill_name="skill_draft_lifecycle",
                    title="1st-Round Exceptional Draft Performers (95th Percentile)",
                    dataset_source="dev/player/player_draft/draftee_lifecycle/results/multi_exceptional_players.csv",
                    summary_markdown="\n".join(lines),
                    records=df_exc.to_dict(orient="records"),
                    columns=list(df_exc.columns),
                )
        return None

    # ── Skill 8: Scoreboard & Game Recaps ─────────────────────────────────────

    def query_scoreboard_and_recaps(self, entities: Dict[str, Any]) -> Optional[ComputeSkillResult]:
        try:
            from data_loader import load_scoreboard_box_and_odds
            box_df, odds_df, avail_dates = load_scoreboard_box_and_odds()
        except Exception:
            box_df, odds_df, avail_dates = None, None, []

        if box_df is None or box_df.empty:
            return None

        target_date = entities.get("date_str")
        teams = entities.get("teams", [])

        sub = box_df.copy()
        if target_date:
            sub = sub[sub["date"].dt.strftime("%Y-%m-%d") == target_date]
        elif teams:
            sub = sub[(sub["tricode_for"].isin(teams)) | (sub["tricode_against"].isin(teams))]
        else:
            latest_dt = sub["date"].max()
            sub = sub[sub["date"] == latest_dt]

        sub = sub.head(10).copy()
        sub["date_str"] = sub["date"].dt.strftime("%Y-%m-%d")
        cols = [c for c in ["gameid", "date_str", "tricode_against", "metric_score_against", "tricode_for", "metric_score_for", "period_ending"] if c in sub.columns]
        lines = ["**NHL Scoreboard Game Results (`data/games.parquet`):**"]
        for _, r in sub[cols].head(8).iterrows():
            lines.append(
                f"- `{r.get('date_str')}`: **{r.get('tricode_against')} {int(r.get('metric_score_against', 0))} @ {r.get('tricode_for')} {int(r.get('metric_score_for', 0))}** ({r.get('period_ending', 'REG')})"
            )
        return ComputeSkillResult(
            skill_name="skill_scoreboard_and_recaps",
            title="NHL Scoreboard & Game Telemetry",
            dataset_source="hf://datasets/jjerlee/nhl-analytics/data/games.parquet",
            summary_markdown="\n".join(lines),
            records=sub[cols].to_dict(orient="records"),
            columns=cols,
        )

    # ── Master Dispatch Method ────────────────────────────────────────────────

    def execute_skills(self, intent: str, domains: List[str], entities: Dict[str, Any]) -> List[ComputeSkillResult]:
        """Dispatches relevant deterministic compute skills based on routed domains and intent."""
        results: List[ComputeSkillResult] = []

        # Always attach live data examples for DATA_QUERY and HYBRID, or top leaderboard context
        if "player_ppi" in domains or ("xg_gsax" in domains and intent in ("DATA_QUERY", "HYBRID")):
            res = self.query_skater_stats(entities)
            if res:
                results.append(res)

        if "goalie_ppi" in domains or ("xg_gsax" in domains and "gsax" in (entities.get("raw_query") or "").lower()):
            res = self.query_goalie_stats(entities)
            if res:
                results.append(res)

        if "team_season" in domains:
            res = self.query_team_season_kpis(entities)
            if res:
                results.append(res)

        if "rebuild" in domains:
            res = self.query_rebuild_regimes(entities)
            if res:
                results.append(res)

        if "hits" in domains:
            res = self.query_hit_impact(entities)
            if res:
                results.append(res)

        if "blocking_shots" in domains:
            res = self.query_blocking_shots()
            if res:
                results.append(res)

        if "draft" in domains or "map_travel" in domains:
            res = self.query_draft_and_map(entities, domains)
            if res:
                results.append(res)

        if "scoreboard_recaps" in domains:
            res = self.query_scoreboard_and_recaps(entities)
            if res:
                results.append(res)

        return results
