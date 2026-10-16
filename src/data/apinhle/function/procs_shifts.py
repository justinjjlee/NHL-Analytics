"""
procs_shifts.py — Play-by-Play Shift & Line Chemistry Analytics Processor.

Reconstructs on-ice Forward Trios (3F) and Defense Pairings (2D) from NHL shift chart
telemetry, intersects shift stints with play-by-play events (Corsi, Fenwick, Shots,
Goals, Hits, Blocks, Turnovers, Zone Faceoffs), tracks sequential in-game shift
momentum and period-by-period bench adjustments, and computes rolling 5-game (L5)
line continuity and performance progression/regression metrics.
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("ShiftLineAnalyticsProcessor")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _mmss_to_sec(val: Any) -> int:
    """Convert MM:SS string to integer seconds."""
    if pd.isna(val):
        return 0
    s = str(val).strip()
    if ":" not in s:
        try:
            return int(float(s))
        except Exception:
            return 0
    parts = s.split(":")
    try:
        return int(parts[0]) * 60 + int(parts[1])
    except Exception:
        return 0


def _sec_to_mmss(sec: float) -> str:
    """Convert seconds to MM:SS string."""
    s = max(0, int(round(sec)))
    return f"{s // 60}:{s % 60:02d}"


class ShiftLineAnalyticsProcessor:
    """Processes shift charts, play-by-play events, and rolling 5-game (L5) line & team trends."""

    def __init__(
        self,
        play_dir: str = "./latest/play",
        box_dir: str = "./latest/box",
        team_dir: str = "./latest/team",
        output_dir: str = "./latest/action/shifts",
    ):
        self.play_dir = play_dir
        self.box_dir = box_dir
        self.team_dir = team_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def get_available_seasons(self) -> List[str]:
        """Return sorted list of season years that have shift or box data."""
        shift_files = sorted(glob.glob(os.path.join(self.play_dir, "*_playbyplay_shift.csv")))
        seasons = []
        for f in shift_files:
            base = os.path.basename(f)
            yr = base.split("_")[0]
            if yr.isdigit():
                seasons.append(yr)
        return sorted(set(seasons))

    def _build_player_position_lookup(self, season: str) -> Dict[int, Tuple[str, str, str]]:
        """Build mapping of player_id -> (short_name, pos_group ['F'|'D'|'G'], pos_code)."""
        pos_map: Dict[int, Tuple[str, str, str]] = {}

        pbp_p_path = os.path.join(self.play_dir, f"{season}_playbyplay_player.csv")
        if os.path.exists(pbp_p_path):
            try:
                df_pp = pd.read_csv(pbp_p_path, low_memory=False)
                for _, r in df_pp.drop_duplicates("playerId").iterrows():
                    if pd.isna(r.get("playerId")):
                        continue
                    pid = int(r["playerId"])
                    fn = str(r.get("firstName.default", "")).strip()
                    ln = str(r.get("lastName.default", "")).strip()
                    short_name = f"{fn[0]}. {ln}" if fn else ln
                    pc = str(r.get("positionCode", "F")).strip().upper()
                    pg = "F" if pc in ("C", "L", "R", "LW", "RW", "F") else ("D" if pc == "D" else "G")
                    pos_map[pid] = (short_name, pg, pc)
            except Exception as e:
                logger.warning("Could not load %s: %s", pbp_p_path, e)

        box_p_path = os.path.join(self.box_dir, f"{season}_box_player.csv")
        if os.path.exists(box_p_path):
            try:
                df_bp = pd.read_csv(box_p_path, low_memory=False)
                for _, r in df_bp.drop_duplicates("playerId").iterrows():
                    if pd.isna(r.get("playerId")):
                        continue
                    pid = int(r["playerId"])
                    if pid not in pos_map:
                        name = str(r.get("name.default", "Unknown")).strip()
                        pc = str(r.get("position", "F")).strip().upper()
                        pg = "F" if pc in ("C", "L", "R", "LW", "RW", "F") else ("D" if pc == "D" else "G")
                        pos_map[pid] = (name, pg, pc)
            except Exception as e:
                logger.warning("Could not load %s: %s", box_p_path, e)

        return pos_map

    def process_season(self, season: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Process a single season's shift charts, L5 line trends, and L5 team trends."""
        shift_path = os.path.join(self.play_dir, f"{season}_playbyplay_shift.csv")
        pbp_path = os.path.join(self.play_dir, f"{season}_playbyplay.csv")
        box_path = os.path.join(self.box_dir, f"{season}_box.csv")
        gamestats_path = os.path.join(self.box_dir, f"{season}_box_gameStats.csv")

        if not os.path.exists(shift_path):
            logger.warning("Shift file missing for season %s: %s", season, shift_path)
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        logger.info("Loading shift and play-by-play data for season %s...", season)
        df_shifts = pd.read_csv(shift_path, low_memory=False)
        df_shifts.columns = [str(c).lower() for c in df_shifts.columns]

        df_pbp = pd.read_csv(pbp_path, low_memory=False) if os.path.exists(pbp_path) else pd.DataFrame()
        df_box = pd.read_csv(box_path, low_memory=False) if os.path.exists(box_path) else pd.DataFrame()
        df_gs = pd.read_csv(gamestats_path, low_memory=False) if os.path.exists(gamestats_path) else pd.DataFrame()

        pos_map = self._build_player_position_lookup(season)

        # Build game metadata map: gameid -> (game_date, home_team, away_team)
        game_meta: Dict[int, Dict[str, Any]] = {}
        if not df_box.empty and "gameid" in df_box.columns:
            for _, r in df_box.iterrows():
                gid = int(r["gameid"])
                game_meta[gid] = {
                    "game_date": str(r.get("date", "")),
                    "home_team": str(r.get("tricode_for", "")),
                    "away_team": str(r.get("tricode_against", "")),
                    "home_score": int(r.get("metric_score_for", 0)) if pd.notna(r.get("metric_score_for")) else 0,
                    "away_score": int(r.get("metric_score_against", 0)) if pd.notna(r.get("metric_score_against")) else 0,
                }

        # Filter to actual player shift rows (typecode == 517)
        if "typecode" in df_shifts.columns:
            df_shifts = df_shifts[df_shifts["typecode"] == 517].copy()

        df_shifts = df_shifts.dropna(subset=["gameid", "playerid", "period", "starttime", "endtime"]).copy()
        df_shifts["gameid"] = df_shifts["gameid"].astype(int)
        df_shifts["playerid"] = df_shifts["playerid"].astype(int)
        df_shifts["period"] = df_shifts["period"].astype(int)
        df_shifts["s_sec"] = df_shifts["starttime"].map(_mmss_to_sec)
        df_shifts["e_sec"] = df_shifts["endtime"].map(_mmss_to_sec)
        df_shifts = df_shifts[df_shifts["e_sec"] > df_shifts["s_sec"]].copy()

        # Attach player names and position groups
        def _resolve_name(row: pd.Series) -> str:
            pid = int(row["playerid"])
            if pid in pos_map:
                return pos_map[pid][0]
            fn = str(row.get("firstname", "")).strip()
            ln = str(row.get("lastname", "")).strip()
            return f"{fn[0]}. {ln}" if fn else ln

        def _resolve_pg(pid: int) -> str:
            return pos_map.get(int(pid), ("Unknown", "F", "F"))[1]

        df_shifts["pname"] = df_shifts.apply(_resolve_name, axis=1)
        df_shifts["pg"] = df_shifts["playerid"].map(_resolve_pg)

        # Prepare play-by-play events
        if not df_pbp.empty and "gameid" in df_pbp.columns:
            df_pbp = df_pbp.dropna(subset=["gameid", "periodDescriptor.number", "timeInPeriod"]).copy()
            df_pbp["gameid"] = df_pbp["gameid"].astype(int)
            df_pbp["period"] = df_pbp["periodDescriptor.number"].astype(int)
            df_pbp["ev_sec"] = df_pbp["timeInPeriod"].map(_mmss_to_sec)

        # 1. Reconstruct Game-by-Game Line Shifts (3F & 2D) + Individual Skater On-Ice Stats
        line_rows: List[Dict[str, Any]] = []
        skater_game_onice: List[Dict[str, Any]] = []

        pbp_by_game = {gid: grp for gid, grp in df_pbp.groupby("gameid")} if not df_pbp.empty else {}

        for gid, g_shifts in df_shifts.groupby("gameid"):
            meta = game_meta.get(int(gid), {})
            g_date = meta.get("game_date", "")
            home_team = meta.get("home_team", "")
            away_team = meta.get("away_team", "")
            g_pbp = pbp_by_game.get(int(gid), pd.DataFrame())
            if not g_date and not g_pbp.empty and "gameDate" in g_pbp.columns:
                g_date = str(g_pbp["gameDate"].iloc[0])

            teams_in_game = sorted(g_shifts["teamabbrev"].dropna().unique())
            for team in teams_in_game:
                opp = away_team if team == home_team else (home_team if team == away_team else "")
                if not opp and len(teams_in_game) == 2:
                    opp = [t for t in teams_in_game if t != team][0]
                home_away = "HOME" if team == home_team else "AWAY"

                t_shifts = g_shifts[g_shifts["teamabbrev"] == team]
                unit_records, skater_records = self._analyze_team_game_shifts(
                    season=int(season),
                    gameid=int(gid),
                    game_date=g_date,
                    team=str(team),
                    opponent=str(opp),
                    home_away=home_away,
                    t_shifts=t_shifts,
                    g_pbp=g_pbp,
                )
                line_rows.extend(unit_records)
                skater_game_onice.extend(skater_records)

        df_lines = pd.DataFrame(line_rows)
        df_skater_onice = pd.DataFrame(skater_game_onice)

        if not df_lines.empty:
            df_lines = df_lines.sort_values(
                by=["game_date", "gameid", "team", "unit_type", "unit_rank"]
            ).reset_index(drop=True)

        # 2. Compute Rolling L5 Line Trends (Line Continuity, Line Changes, Improvement/Regression)
        df_line_trends_l5 = self._compute_line_trends_l5(df_lines, df_skater_onice)

        # 3. Compute Team-Level Rolling L5 & Full-Season Context Table
        df_team_trends_l5 = self._compute_team_game_trends_l5(
            season=int(season),
            df_gs=df_gs,
            df_box=df_box,
            df_line_trends_l5=df_line_trends_l5,
        )

        # Save outputs
        out_lines = os.path.join(self.output_dir, f"{season}_game_line_shifts.csv")
        out_line_l5 = os.path.join(self.output_dir, f"{season}_game_line_trends_l5.csv")
        out_team_l5 = os.path.join(self.output_dir, f"{season}_team_game_trends_l5.csv")

        if not df_lines.empty:
            df_lines.to_csv(out_lines, index=False)
        if not df_line_trends_l5.empty:
            df_line_trends_l5.to_csv(out_line_l5, index=False)
        if not df_team_trends_l5.empty:
            df_team_trends_l5.to_csv(out_team_l5, index=False)

        # Also write master un-prefixed tables for direct lookup
        if not df_lines.empty:
            df_lines.to_csv(os.path.join(self.output_dir, "game_line_shifts.csv"), index=False)
        if not df_line_trends_l5.empty:
            df_line_trends_l5.to_csv(os.path.join(self.output_dir, "game_line_trends_l5.csv"), index=False)
        if not df_team_trends_l5.empty:
            df_team_trends_l5.to_csv(os.path.join(self.output_dir, "team_game_trends_l5.csv"), index=False)

        logger.info(
            "Season %s shift processing complete: %d line records, %d L5 line trend rows, %d team L5 trend rows.",
            season, len(df_lines), len(df_line_trends_l5), len(df_team_trends_l5)
        )
        return df_lines, df_line_trends_l5, df_team_trends_l5

    def _analyze_team_game_shifts(
        self,
        season: int,
        gameid: int,
        game_date: str,
        team: str,
        opponent: str,
        home_away: str,
        t_shifts: pd.DataFrame,
        g_pbp: pd.DataFrame,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Reconstruct 3F and 2D stints for one team in one game and intersect with PBP events."""
        unit_stats: Dict[Tuple[str, str], Dict[str, Any]] = {}
        skater_stats: Dict[int, Dict[str, Any]] = {}

        total_team_toi = {"3F": 0, "2D": 0}
        per_team_toi = {
            "3F": {1: 0, 2: 0, 3: 0, 4: 0},
            "2D": {1: 0, 2: 0, 3: 0, 4: 0},
        }

        for per in sorted(t_shifts["period"].unique()):
            p_shifts = t_shifts[t_shifts["period"] == per]
            p_ev = g_pbp[g_pbp["period"] == per] if not g_pbp.empty else pd.DataFrame()

            s_arr = p_shifts["s_sec"].to_numpy()
            e_arr = p_shifts["e_sec"].to_numpy()
            pid_arr = p_shifts["playerid"].to_numpy()
            pname_arr = p_shifts["pname"].to_numpy()
            pg_arr = p_shifts["pg"].to_numpy()

            bounds = sorted(set(s_arr).union(set(e_arr)))
            if len(bounds) < 2:
                continue

            if not p_ev.empty:
                ev_sec_arr = p_ev["ev_sec"].to_numpy()
                ev_type_arr = p_ev["typeDescKey"].astype(str).to_numpy()
                ev_owner_arr = p_ev["details.eventOwnerTeam"].astype(str).to_numpy()
                ev_zc_arr = p_ev["details.zoneCode"].astype(str).to_numpy()
            else:
                ev_sec_arr = np.array([], dtype=int)
                ev_type_arr = np.array([], dtype=str)
                ev_owner_arr = np.array([], dtype=str)
                ev_zc_arr = np.array([], dtype=str)

            for i in range(len(bounds) - 1):
                t1, t2 = int(bounds[i]), int(bounds[i + 1])
                dt = t2 - t1
                if dt <= 0:
                    continue

                mask_act = (s_arr <= t1) & (e_arr >= t2)
                if not np.any(mask_act):
                    continue

                act_pids = pid_arr[mask_act]
                act_pnames = pname_arr[mask_act]
                act_pgs = pg_arr[mask_act]

                # Count event metrics for this elementary slice [t1, t2]
                cf = ca = ff = fa = sf = sa = gf = ga = 0
                hf = ha = bf = ba = tk = gv = 0
                oz_fo = nz_fo = dz_fo = 0

                if len(ev_sec_arr) > 0:
                    ev_mask = (ev_sec_arr >= t1) & (ev_sec_arr <= t2) if t1 == 0 else (ev_sec_arr > t1) & (ev_sec_arr <= t2)
                    if np.any(ev_mask):
                        for et, owner in zip(ev_type_arr[ev_mask], ev_owner_arr[ev_mask]):
                            is_for = (owner == team)
                            if et in ("shot-on-goal", "missed-shot", "blocked-shot", "goal"):
                                if is_for:
                                    cf += 1
                                else:
                                    ca += 1
                            if et in ("shot-on-goal", "missed-shot", "goal"):
                                if is_for:
                                    ff += 1
                                else:
                                    fa += 1
                            if et in ("shot-on-goal", "goal"):
                                if is_for:
                                    sf += 1
                                else:
                                    sa += 1
                            if et == "goal":
                                if is_for:
                                    gf += 1
                                else:
                                    ga += 1
                            elif et == "hit":
                                if is_for:
                                    hf += 1
                                else:
                                    ha += 1
                            elif et == "blocked-shot":
                                if is_for:
                                    ba += 1
                                else:
                                    bf += 1
                            elif et == "takeaway" and is_for:
                                tk += 1
                            elif et == "giveaway" and is_for:
                                gv += 1

                    fo_mask = (ev_sec_arr == t1) & (ev_type_arr == "faceoff")
                    if np.any(fo_mask):
                        for owner, zc in zip(ev_owner_arr[fo_mask], ev_zc_arr[fo_mask]):
                            if owner == team:
                                if zc == "O":
                                    oz_fo += 1
                                elif zc == "D":
                                    dz_fo += 1
                                else:
                                    nz_fo += 1
                            else:
                                if zc == "O":
                                    dz_fo += 1
                                elif zc == "D":
                                    oz_fo += 1
                                else:
                                    nz_fo += 1

                # Unique active skaters by playerid
                seen_skaters: Dict[int, Tuple[str, str]] = {}
                fwds_list: List[Tuple[str, int]] = []
                defs_list: List[Tuple[str, int]] = []
                for pid_val, pn_val, pg_val in zip(act_pids, act_pnames, act_pgs):
                    pid_int = int(pid_val)
                    if pid_int in seen_skaters:
                        continue
                    seen_skaters[pid_int] = (str(pn_val), str(pg_val))
                    if pg_val == "F":
                        fwds_list.append((str(pn_val), pid_int))
                    elif pg_val == "D":
                        defs_list.append((str(pn_val), pid_int))

                if len(fwds_list) + len(defs_list) == 5:
                    for pid_int, (pn_str, pg_str) in seen_skaters.items():
                        if pg_str not in ("F", "D"):
                            continue
                        sk = skater_stats.setdefault(
                            pid_int,
                            {
                                "season": season,
                                "gameid": gameid,
                                "game_date": game_date,
                                "team": team,
                                "playerid": pid_int,
                                "pname": pn_str,
                                "pg": pg_str,
                                "toi_sec": 0,
                                "cf": 0,
                                "ca": 0,
                                "sf": 0,
                                "sa": 0,
                                "gf": 0,
                                "ga": 0,
                            },
                        )
                        sk["toi_sec"] += dt
                        sk["cf"] += cf
                        sk["ca"] += ca
                        sk["sf"] += sf
                        sk["sa"] += sa
                        sk["gf"] += gf
                        sk["ga"] += ga

                active_units: List[Tuple[str, str, str, str]] = []
                if len(fwds_list) == 3:
                    fwds_list.sort(key=lambda x: x[0])
                    sorted_pids = sorted(x[1] for x in fwds_list)
                    u_key = "_".join(str(x) for x in sorted_pids)
                    u_players = " - ".join(x[0] for x in fwds_list)
                    u_pids = ",".join(str(x) for x in sorted_pids)
                    active_units.append(("3F", u_key, u_players, u_pids))
                    total_team_toi["3F"] += dt
                    per_team_toi["3F"][min(per, 4)] = per_team_toi["3F"].get(min(per, 4), 0) + dt

                if len(defs_list) == 2:
                    defs_list.sort(key=lambda x: x[0])
                    sorted_pids = sorted(x[1] for x in defs_list)
                    u_key = "_".join(str(x) for x in sorted_pids)
                    u_players = " - ".join(x[0] for x in defs_list)
                    u_pids = ",".join(str(x) for x in sorted_pids)
                    active_units.append(("2D", u_key, u_players, u_pids))
                    total_team_toi["2D"] += dt
                    per_team_toi["2D"][min(per, 4)] = per_team_toi["2D"].get(min(per, 4), 0) + dt

                for u_type, u_key, u_players, u_pids in active_units:
                    rec = unit_stats.setdefault(
                        (u_type, u_key),
                        {
                            "unit_type": u_type,
                            "unit_key": u_key,
                            "unit_players": u_players,
                            "player_ids": u_pids,
                            "toi_sec": 0,
                            "p1_toi_sec": 0,
                            "p2_toi_sec": 0,
                            "p3_toi_sec": 0,
                            "ot_toi_sec": 0,
                            "cf": 0,
                            "ca": 0,
                            "ff": 0,
                            "fa": 0,
                            "sf": 0,
                            "sa": 0,
                            "gf": 0,
                            "ga": 0,
                            "hits_for": 0,
                            "hits_against": 0,
                            "blocks_for": 0,
                            "blocks_against": 0,
                            "takeaways": 0,
                            "giveaways": 0,
                            "oz_faceoffs": 0,
                            "nz_faceoffs": 0,
                            "dz_faceoffs": 0,
                            "p1_cf": 0,
                            "p1_ca": 0,
                            "p2_cf": 0,
                            "p2_ca": 0,
                            "p3_cf": 0,
                            "p3_ca": 0,
                            "stints": [],  # list of [per, start, end, dt, cf, ca, gf, ga]
                        },
                    )
                    rec["toi_sec"] += dt
                    if per == 1:
                        rec["p1_toi_sec"] += dt
                        rec["p1_cf"] += cf
                        rec["p1_ca"] += ca
                    elif per == 2:
                        rec["p2_toi_sec"] += dt
                        rec["p2_cf"] += cf
                        rec["p2_ca"] += ca
                    elif per == 3:
                        rec["p3_toi_sec"] += dt
                        rec["p3_cf"] += cf
                        rec["p3_ca"] += ca
                    else:
                        rec["ot_toi_sec"] += dt

                    rec["cf"] += cf
                    rec["ca"] += ca
                    rec["ff"] += ff
                    rec["fa"] += fa
                    rec["sf"] += sf
                    rec["sa"] += sa
                    rec["gf"] += gf
                    rec["ga"] += ga
                    rec["hits_for"] += hf
                    rec["hits_against"] += ha
                    rec["blocks_for"] += bf
                    rec["blocks_against"] += ba
                    rec["takeaways"] += tk
                    rec["giveaways"] += gv
                    rec["oz_faceoffs"] += oz_fo
                    rec["nz_faceoffs"] += nz_fo
                    rec["dz_faceoffs"] += dz_fo

                    # Merge contiguous slices into full stints
                    if rec["stints"] and rec["stints"][-1][0] == per and rec["stints"][-1][2] == t1:
                        last_s = rec["stints"][-1]
                        last_s[2] = t2
                        last_s[3] += dt
                        last_s[4] += cf
                        last_s[5] += ca
                        last_s[6] += gf
                        last_s[7] += ga
                    else:
                        rec["stints"].append([per, t1, t2, dt, cf, ca, gf, ga])

        # Format top 6 3F units and top 4 2D units (min 45 seconds together)
        unit_rows: List[Dict[str, Any]] = []
        for u_type, max_units, min_sec, prefix in [("3F", 6, 45, "Line"), ("2D", 4, 45, "Pair")]:
            candidates = [
                v for (ut, _), v in unit_stats.items()
                if ut == u_type and v["toi_sec"] >= min_sec
            ]
            candidates.sort(key=lambda x: x["toi_sec"], reverse=True)
            tot_u_toi = max(1, total_team_toi[u_type])
            p1_tot = max(1, per_team_toi[u_type].get(1, 0))
            p2_tot = max(1, per_team_toi[u_type].get(2, 0))
            p3_tot = max(1, per_team_toi[u_type].get(3, 0))

            for rank_idx, c in enumerate(candidates[:max_units], start=1):
                toi = c["toi_sec"]
                stints = c["stints"]
                stints_cnt = max(1, len(stints))
                avg_stint = round(toi / stints_cnt, 1)

                # Sequential stint metrics
                dom_stints = 0
                pin_stints = 0
                max_consec_pos = 0
                cur_consec = 0
                for st in stints:
                    _, _, _, s_dt, s_cf, s_ca, s_gf, s_ga = st
                    if (s_cf >= 2 and s_ca == 0) or s_gf > 0:
                        dom_stints += 1
                    if s_dt >= 35 and s_ca >= 2 and s_cf == 0:
                        pin_stints += 1
                    if (s_cf >= s_ca and s_cf > 0) or s_gf > s_ga:
                        cur_consec += 1
                        if cur_consec > max_consec_pos:
                            max_consec_pos = cur_consec
                    elif s_ca > s_cf or s_ga > s_gf:
                        cur_consec = 0

                cf, ca = c["cf"], c["ca"]
                ff, fa = c["ff"], c["fa"]
                sf, sa = c["sf"], c["sa"]
                gf, ga = c["gf"], c["ga"]

                cf_pct = round(100.0 * cf / (cf + ca), 1) if (cf + ca) > 0 else 50.0
                ff_pct = round(100.0 * ff / (ff + fa), 1) if (ff + fa) > 0 else 50.0
                sf_pct = round(100.0 * sf / (sf + sa), 1) if (sf + sa) > 0 else 50.0

                p1_cf_pct = round(100.0 * c["p1_cf"] / (c["p1_cf"] + c["p1_ca"]), 1) if (c["p1_cf"] + c["p1_ca"]) > 0 else 50.0
                p2_cf_pct = round(100.0 * c["p2_cf"] / (c["p2_cf"] + c["p2_ca"]), 1) if (c["p2_cf"] + c["p2_ca"]) > 0 else 50.0
                p3_cf_pct = round(100.0 * c["p3_cf"] / (c["p3_cf"] + c["p3_ca"]), 1) if (c["p3_cf"] + c["p3_ca"]) > 0 else 50.0

                oz, dz = c["oz_faceoffs"], c["dz_faceoffs"]
                ozs_pct = round(100.0 * oz / (oz + dz), 1) if (oz + dz) > 0 else 50.0

                toi_share = round(100.0 * toi / tot_u_toi, 1)
                p1_share = round(100.0 * c["p1_toi_sec"] / p1_tot, 1)
                p2_share = round(100.0 * c["p2_toi_sec"] / p2_tot, 1)
                p3_share = round(100.0 * c["p3_toi_sec"] / p3_tot, 1)
                p3_shift_delta = round(p3_share - 0.5 * (p1_share + p2_share), 1)

                hours = toi / 3600.0
                cf60 = round(cf / hours, 1) if hours > 0 else 0.0
                ca60 = round(ca / hours, 1) if hours > 0 else 0.0
                sf60 = round(sf / hours, 1) if hours > 0 else 0.0
                sa60 = round(sa / hours, 1) if hours > 0 else 0.0
                gf60 = round(gf / hours, 2) if hours > 0 else 0.0
                ga60 = round(ga / hours, 2) if hours > 0 else 0.0

                # Composite Line Impact Score (0-100, centered at 50)
                # Weight sample confidence by min(1.0, toi / 300.0) so 1-minute cameos don't skew extremes
                sample_w = min(1.0, toi / 300.0)
                raw_delta = (
                    0.45 * (cf_pct - 50.0)
                    + 0.30 * (sf_pct - 50.0)
                    + 14.0 * (gf - ga)
                    + 2.5 * (dom_stints - pin_stints)
                )
                impact_score = round(float(np.clip(50.0 + sample_w * raw_delta, 5.0, 98.0)), 1)

                if impact_score >= 65.0 or ((gf - ga) >= 2 and cf_pct >= 50.0):
                    perf_tier = "DOMINANT"
                elif impact_score >= 55.0:
                    perf_tier = "STRONG"
                elif impact_score >= 45.0:
                    perf_tier = "STEADY"
                elif impact_score >= 35.0:
                    perf_tier = "STRUGGLED"
                else:
                    perf_tier = "CAVED_IN"

                unit_rows.append(
                    {
                        "season": season,
                        "gameid": gameid,
                        "game_date": game_date,
                        "team": team,
                        "opponent": opponent,
                        "home_away": home_away,
                        "unit_type": u_type,
                        "unit_rank": rank_idx,
                        "unit_label": f"{prefix} {rank_idx}",
                        "unit_key": c["unit_key"],
                        "unit_players": c["unit_players"],
                        "player_ids": c["player_ids"],
                        "toi_sec": toi,
                        "toi_formatted": _sec_to_mmss(toi),
                        "toi_share_pct": toi_share,
                        "p1_toi_sec": c["p1_toi_sec"],
                        "p2_toi_sec": c["p2_toi_sec"],
                        "p3_toi_sec": c["p3_toi_sec"],
                        "ot_toi_sec": c["ot_toi_sec"],
                        "p1_toi_share_pct": p1_share,
                        "p2_toi_share_pct": p2_share,
                        "p3_toi_share_pct": p3_share,
                        "p3_deployment_shift_pct": p3_shift_delta,
                        "stints_count": stints_cnt,
                        "avg_stint_sec": avg_stint,
                        "cf": cf,
                        "ca": ca,
                        "cf_diff": cf - ca,
                        "cf_pct": cf_pct,
                        "ff": ff,
                        "fa": fa,
                        "ff_diff": ff - fa,
                        "ff_pct": ff_pct,
                        "sf": sf,
                        "sa": sa,
                        "sf_diff": sf - sa,
                        "sf_pct": sf_pct,
                        "gf": gf,
                        "ga": ga,
                        "goal_diff": gf - ga,
                        "cf_per60": cf60,
                        "ca_per60": ca60,
                        "sf_per60": sf60,
                        "sa_per60": sa60,
                        "gf_per60": gf60,
                        "ga_per60": ga60,
                        "hits_for": c["hits_for"],
                        "hits_against": c["hits_against"],
                        "blocks_for": c["blocks_for"],
                        "blocks_against": c["blocks_against"],
                        "takeaways": c["takeaways"],
                        "giveaways": c["giveaways"],
                        "turnover_diff": c["takeaways"] - c["giveaways"],
                        "oz_faceoffs": oz,
                        "nz_faceoffs": c["nz_faceoffs"],
                        "dz_faceoffs": dz,
                        "ozs_pct": ozs_pct,
                        "dominant_stints": dom_stints,
                        "pinned_stints": pin_stints,
                        "max_consec_positive_stints": max_consec_pos,
                        "p1_cf_pct": p1_cf_pct,
                        "p2_cf_pct": p2_cf_pct,
                        "p3_cf_pct": p3_cf_pct,
                        "line_impact_score": impact_score,
                        "performance_tier": perf_tier,
                    }
                )

        return unit_rows, list(skater_stats.values())

    def _compute_line_trends_l5(
        self, df_lines: pd.DataFrame, df_skater_onice: pd.DataFrame
    ) -> pd.DataFrame:
        """Enrich game line shifts with rolling past 5 games (L5) continuity, line changes, and deltas."""
        if df_lines.empty:
            return pd.DataFrame()

        enriched_rows: List[Dict[str, Any]] = []

        # Index skater on-ice history by (team, gameid)
        skater_by_team_game: Dict[Tuple[str, int], pd.DataFrame] = {}
        if not df_skater_onice.empty:
            for (tm, gid), grp in df_skater_onice.groupby(["team", "gameid"]):
                skater_by_team_game[(str(tm), int(gid))] = grp

        for team, t_df in df_lines.groupby("team"):
            # Chronological order of games for this team
            game_order = (
                t_df[["gameid", "game_date"]]
                .drop_duplicates("gameid")
                .sort_values(["game_date", "gameid"])["gameid"]
                .tolist()
            )
            lines_by_game = {int(gid): grp for gid, grp in t_df.groupby("gameid")}

            for idx, gid in enumerate(game_order):
                prior_gids = game_order[max(0, idx - 5) : idx]
                l5_avail = len(prior_gids)
                cur_game_df = lines_by_game[int(gid)]

                prior_lines_df = (
                    pd.concat([lines_by_game[pg] for pg in prior_gids], ignore_index=True)
                    if prior_gids
                    else pd.DataFrame()
                )
                prior_skaters_df = (
                    pd.concat(
                        [
                            skater_by_team_game[(str(team), int(pg))]
                            for pg in prior_gids
                            if (str(team), int(pg)) in skater_by_team_game
                        ],
                        ignore_index=True,
                    )
                    if prior_gids
                    else pd.DataFrame()
                )

                for _, row in cur_game_df.iterrows():
                    r_dict = row.to_dict()
                    u_key = str(row["unit_key"])
                    pids = [int(x) for x in str(row["player_ids"]).split(",") if x.strip().isdigit()]

                    # 1. Exact unit history over L5
                    l5_unit_games = 0
                    l5_unit_toi = 0
                    l5_unit_cf = 0
                    l5_unit_ca = 0
                    l5_unit_sf = 0
                    l5_unit_sa = 0
                    l5_unit_gf = 0
                    l5_unit_ga = 0
                    l5_unit_impact = 50.0

                    if not prior_lines_df.empty:
                        u_hist = prior_lines_df[prior_lines_df["unit_key"] == u_key]
                        if not u_hist.empty:
                            l5_unit_games = int((u_hist["toi_sec"] >= 90).sum())
                            l5_unit_toi = int(u_hist["toi_sec"].sum())
                            l5_unit_cf = int(u_hist["cf"].sum())
                            l5_unit_ca = int(u_hist["ca"].sum())
                            l5_unit_sf = int(u_hist["sf"].sum())
                            l5_unit_sa = int(u_hist["sa"].sum())
                            l5_unit_gf = int(u_hist["gf"].sum())
                            l5_unit_ga = int(u_hist["ga"].sum())
                            if l5_unit_toi > 0:
                                l5_unit_impact = round(
                                    float(np.average(u_hist["line_impact_score"], weights=u_hist["toi_sec"])),
                                    1,
                                )

                    l5_unit_cf_pct = (
                        round(100.0 * l5_unit_cf / (l5_unit_cf + l5_unit_ca), 1)
                        if (l5_unit_cf + l5_unit_ca) > 0
                        else np.nan
                    )
                    l5_unit_sf_pct = (
                        round(100.0 * l5_unit_sf / (l5_unit_sf + l5_unit_sa), 1)
                        if (l5_unit_sf + l5_unit_sa) > 0
                        else np.nan
                    )

                    # 2. Individual members' L5 baseline (crucial when evaluating a brand-new line change!)
                    mem_cf_pct = 50.0
                    mem_sf_pct = 50.0
                    mem_gf60 = 0.0
                    mem_impact = 50.0

                    if not prior_skaters_df.empty and pids:
                        m_hist = prior_skaters_df[prior_skaters_df["playerid"].isin(pids)]
                        if not m_hist.empty and m_hist["toi_sec"].sum() > 0:
                            m_toi = float(m_hist["toi_sec"].sum())
                            m_cf = float(m_hist["cf"].sum())
                            m_ca = float(m_hist["ca"].sum())
                            m_sf = float(m_hist["sf"].sum())
                            m_sa = float(m_hist["sa"].sum())
                            m_gf = float(m_hist["gf"].sum())
                            m_ga = float(m_hist["ga"].sum())
                            mem_cf_pct = round(100.0 * m_cf / (m_cf + m_ca), 1) if (m_cf + m_ca) > 0 else 50.0
                            mem_sf_pct = round(100.0 * m_sf / (m_sf + m_sa), 1) if (m_sf + m_sa) > 0 else 50.0
                            mem_gf60 = round(m_gf / (m_toi / 3600.0), 2) if m_toi > 0 else 0.0
                            mem_impact = round(
                                float(
                                    np.clip(
                                        50.0
                                        + 0.45 * (mem_cf_pct - 50.0)
                                        + 0.30 * (mem_sf_pct - 50.0)
                                        + 6.0 * ((m_gf - m_ga) / max(1.0, l5_avail)),
                                        15.0,
                                        85.0,
                                    )
                                ),
                                1,
                            )

                    # 3. Classify Line Change Status
                    p1_s = int(row["p1_toi_sec"])
                    p23_s = int(row["p2_toi_sec"]) + int(row["p3_toi_sec"])
                    if l5_avail == 0:
                        change_status = "SEASON_DEBUT_BASELINE"
                    elif p1_s < 45 and p23_s >= 150 and l5_unit_toi < 180:
                        change_status = "IN_GAME_SHUFFLE"
                    elif l5_unit_toi < 120:
                        change_status = "NEW_LINE_COMBO"
                    elif l5_unit_games == 1:
                        change_status = "REUNITED_COMBO"
                    else:
                        change_status = "ESTABLISHED_UNIT"

                    # 4. Compute Improvement / Regression Delta vs L5 Baseline
                    ref_cf_pct = l5_unit_cf_pct if (pd.notna(l5_unit_cf_pct) and l5_unit_toi >= 180) else mem_cf_pct
                    ref_sf_pct = l5_unit_sf_pct if (pd.notna(l5_unit_sf_pct) and l5_unit_toi >= 180) else mem_sf_pct
                    ref_impact = l5_unit_impact if l5_unit_toi >= 180 else mem_impact

                    delta_cf = round(float(row["cf_pct"]) - float(ref_cf_pct), 1) if l5_avail > 0 else 0.0
                    delta_sf = round(float(row["sf_pct"]) - float(ref_sf_pct), 1) if l5_avail > 0 else 0.0
                    delta_imp = round(float(row["line_impact_score"]) - float(ref_impact), 1) if l5_avail > 0 else 0.0

                    is_new_or_shuffled = change_status in ("NEW_LINE_COMBO", "IN_GAME_SHUFFLE")
                    cur_gd = int(row["goal_diff"])
                    cur_cf_pct = float(row["cf_pct"])

                    if l5_avail == 0:
                        verdict = "SEASON_DEBUT"
                    elif is_new_or_shuffled:
                        if delta_imp >= 5.0 or delta_cf >= 6.0 or (cur_gd >= 1 and cur_cf_pct >= 52.0):
                            verdict = "LINE_CHANGE_IMPROVED"
                        elif delta_imp <= -5.0 or delta_cf <= -6.0 or (cur_gd <= -1 and cur_cf_pct <= 47.0):
                            verdict = "LINE_CHANGE_REGRESSED"
                        else:
                            verdict = "LINE_CHANGE_NEUTRAL"
                    else:
                        if delta_imp >= 6.0 or delta_cf >= 7.0 or (cur_gd >= 2 and cur_cf_pct >= 53.0):
                            verdict = "ESTABLISHED_SURGE"
                        elif delta_imp <= -6.0 or delta_cf <= -7.0 or (cur_gd <= -2 and cur_cf_pct <= 46.0):
                            verdict = "ESTABLISHED_REGRESSION"
                        else:
                            verdict = "ESTABLISHED_STEADY"

                    # 5. Build Human-Readable Narrative Note for this Line/Pair
                    story = self._build_line_story(
                        row=row,
                        change_status=change_status,
                        verdict=verdict,
                        l5_avail=l5_avail,
                        l5_unit_games=l5_unit_games,
                        ref_cf_pct=ref_cf_pct,
                        delta_cf=delta_cf,
                        delta_imp=delta_imp,
                    )

                    r_dict.update(
                        {
                            "l5_team_games_available": l5_avail,
                            "l5_exact_unit_games": l5_unit_games,
                            "l5_exact_unit_toi_sec": l5_unit_toi,
                            "l5_exact_unit_cf_pct": l5_unit_cf_pct if pd.notna(l5_unit_cf_pct) else ref_cf_pct,
                            "l5_exact_unit_sf_pct": l5_unit_sf_pct if pd.notna(l5_unit_sf_pct) else ref_sf_pct,
                            "l5_exact_unit_gf": l5_unit_gf,
                            "l5_exact_unit_ga": l5_unit_ga,
                            "l5_exact_unit_impact": l5_unit_impact,
                            "l5_members_baseline_cf_pct": mem_cf_pct,
                            "l5_members_baseline_sf_pct": mem_sf_pct,
                            "l5_members_baseline_gf60": mem_gf60,
                            "l5_members_baseline_impact": mem_impact,
                            "line_change_status": change_status,
                            "delta_cf_pct_vs_l5": delta_cf,
                            "delta_sf_pct_vs_l5": delta_sf,
                            "delta_impact_vs_l5": delta_imp,
                            "l5_trend_verdict": verdict,
                            "line_story_summary": story,
                        }
                    )
                    enriched_rows.append(r_dict)

        return pd.DataFrame(enriched_rows)

    def _build_line_story(
        self,
        row: pd.Series,
        change_status: str,
        verdict: str,
        l5_avail: int,
        l5_unit_games: int,
        ref_cf_pct: float,
        delta_cf: float,
        delta_imp: float,
    ) -> str:
        """Construct a concise analytical sentence describing the unit's game & L5 trajectory."""
        players = str(row["unit_players"])
        label = str(row["unit_label"])
        toi_str = str(row["toi_formatted"])
        cf, ca, cf_pct = int(row["cf"]), int(row["ca"]), float(row["cf_pct"])
        sf, sa = int(row["sf"]), int(row["sa"])
        gf, ga = int(row["gf"]), int(row["ga"])
        p3_shift = float(row["p3_deployment_shift_pct"])
        consec = int(row["max_consec_positive_stints"])

        score_clause = f"outscoring the opposition {gf}-{ga}" if gf > ga else (
            f"surrendering a {gf}-{ga} goal margin" if ga > gf else f"breaking even on the scoreboard ({gf}-{ga})"
        )
        poss_clause = f"{cf}-{ca} shot attempts ({cf_pct:.1f}% CF%, {sf}-{sa} SOG)"
        cf_cmp = (
            f"{delta_cf:+.1f}% CF% improvement over their {ref_cf_pct:.1f}% L5 baseline"
            if delta_cf >= 0
            else f"{delta_cf:+.1f}% CF% vs. their {ref_cf_pct:.1f}% L5 baseline, offsetting territorial pressure with clinical finishing"
        )

        if l5_avail == 0:
            base = f"{label} ({players}) logged {toi_str} together, posting {poss_clause} and {score_clause}."
        elif change_status in ("NEW_LINE_COMBO", "IN_GAME_SHUFFLE"):
            tag = "Mid-game bench adjustment" if change_status == "IN_GAME_SHUFFLE" else "Newly assembled combination"
            if verdict == "LINE_CHANGE_IMPROVED":
                base = (
                    f"{tag} {label} ({players}, {toi_str} TOI) paid immediate dividends, posting {poss_clause} "
                    f"and {score_clause} ({cf_cmp})."
                )
            elif verdict == "LINE_CHANGE_REGRESSED":
                base = (
                    f"{tag} {label} ({players}, {toi_str} TOI) struggled to find chemistry, posting {poss_clause} "
                    f"while {score_clause} ({delta_cf:+.1f}% CF% vs. their {ref_cf_pct:.1f}% L5 baseline)."
                )
            else:
                base = (
                    f"{tag} {label} ({players}, {toi_str} TOI) held steady in its debut look with {poss_clause} "
                    f"and {score_clause} (vs. {ref_cf_pct:.1f}% L5 baseline)."
                )
        else:
            if verdict == "ESTABLISHED_SURGE":
                if delta_cf >= 0:
                    base = (
                        f"Established {label} ({players}, {toi_str} TOI, {l5_unit_games} of L5 games together) surged above its "
                        f"{ref_cf_pct:.1f}% L5 average, driving play at {poss_clause} ({delta_cf:+.1f}% CF%) and {score_clause}."
                    )
                else:
                    base = (
                        f"Established {label} ({players}, {toi_str} TOI, {l5_unit_games} of L5 games together) delivered opportunistic "
                        f"production ({score_clause}) despite {poss_clause} ({delta_cf:+.1f}% CF% vs. {ref_cf_pct:.1f}% L5 average)."
                    )
            elif verdict == "ESTABLISHED_REGRESSION":
                base = (
                    f"Established {label} ({players}, {toi_str} TOI) regressed from its {ref_cf_pct:.1f}% L5 form, "
                    f"finishing at {poss_clause} ({delta_cf:+.1f}% CF%) while {score_clause}."
                )
            else:
                base = (
                    f"Established {label} ({players}, {toi_str} TOI) maintained steady continuity ({poss_clause}, "
                    f"{delta_cf:+.1f}% vs. {ref_cf_pct:.1f}% L5 baseline), {score_clause}."
                )

        extras = []
        if p3_shift >= 8.0:
            extras.append(f"heavily leaned on in Period 3 (+{p3_shift:.1f}% TOI share jump)")
        elif p3_shift <= -8.0:
            extras.append(f"sheltered/shortened in Period 3 ({p3_shift:.1f}% TOI share drop)")
        if consec >= 3:
            extras.append(f"strung together {consec} consecutive positive territorial shifts")

        if extras:
            base += " Notable progression: " + "; ".join(extras) + "."
        return base

    def _compute_team_game_trends_l5(
        self,
        season: int,
        df_gs: pd.DataFrame,
        df_box: pd.DataFrame,
        df_line_trends_l5: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute team-level L5 rolling form, schedule adversity, and line shift summary per game."""
        if df_gs.empty:
            return pd.DataFrame()

        df_gs = df_gs.copy()
        df_gs["gameid_for"] = df_gs["gameid_for"].astype(int)
        df_gs = df_gs.sort_values(["gameDate_for", "gameid_for"]).reset_index(drop=True)

        # Index line trends by (team, gameid)
        lines_by_tg: Dict[Tuple[str, int], pd.DataFrame] = {}
        if not df_line_trends_l5.empty:
            for (tm, gid), grp in df_line_trends_l5.groupby(["team", "gameid"]):
                lines_by_tg[(str(tm), int(gid))] = grp

        rows: List[Dict[str, Any]] = []

        for team, t_df in df_gs.groupby("team_tri_for"):
            t_df = t_df.sort_values(["gameDate_for", "gameid_for"]).reset_index(drop=True)
            consec_road = 0
            consec_home = 0

            for idx, r in t_df.iterrows():
                gid = int(r["gameid_for"])
                g_date = str(r.get("gameDate_for", ""))
                opp = str(r.get("team_tri_against", ""))
                loc = str(r.get("teamloc_for", "")).upper()
                home_away = "HOME" if loc in ("H", "HOME") else "AWAY"

                if home_away == "AWAY":
                    consec_road += 1
                    consec_home = 0
                else:
                    consec_home += 1
                    consec_road = 0

                def _parse_days(v: Any) -> int:
                    if pd.isna(v):
                        return 3
                    s = str(v).strip().split()[0]
                    try:
                        return max(1, int(float(s)))
                    except Exception:
                        return 3

                rest_days = _parse_days(r.get("gameDate_del_for"))
                opp_rest = _parse_days(r.get("gameDate_del_against"))
                is_b2b = (rest_days == 1)

                # Rolling L5 prior games (strictly preceding this game: [max(0, idx-5) : idx])
                prior_df = t_df.iloc[max(0, idx - 5) : idx]
                l5_n = len(prior_df)
                if l5_n > 0:
                    l5_w = int(prior_df["win_for"].sum())
                    l5_l = l5_n - l5_w
                    l5_rec = f"{l5_w}-{l5_l}"
                    l5_gf_pg = round(float(prior_df["goals_for"].mean()), 2)
                    l5_ga_pg = round(float(prior_df["goals_against"].mean()), 2)
                    l5_gd = int((prior_df["goals_for"] - prior_df["goals_against"]).sum())
                    cf_sum = float(prior_df["corsi_for"].sum())
                    ca_sum = float(prior_df["corsi_against"].sum())
                    ff_sum = float(prior_df["fenwick_for"].sum())
                    fa_sum = float(prior_df["fenwick_against"].sum())
                    sf_sum = float(prior_df["shots_for"].sum())
                    sa_sum = float(prior_df["shots_against"].sum())
                    l5_cf_pct = round(100.0 * cf_sum / (cf_sum + ca_sum), 1) if (cf_sum + ca_sum) > 0 else 50.0
                    l5_ff_pct = round(100.0 * ff_sum / (ff_sum + fa_sum), 1) if (ff_sum + fa_sum) > 0 else 50.0
                    l5_sf_pct = round(100.0 * sf_sum / (sf_sum + sa_sum), 1) if (sf_sum + sa_sum) > 0 else 50.0
                    l5_hits_pg = round(float(prior_df["hits_for"].mean()), 1)
                    l5_blocks_pg = round(float(prior_df["shots_blocked_for"].mean()), 1)
                    l5_to_diff = int((prior_df["poss_takeaway_for"] - prior_df["poss_giveaway_for"]).sum())
                else:
                    l5_w = l5_l = l5_gd = l5_to_diff = 0
                    l5_rec = "0-0"
                    l5_gf_pg = l5_ga_pg = l5_hits_pg = l5_blocks_pg = 0.0
                    l5_cf_pct = l5_ff_pct = l5_sf_pct = 50.0

                cur_cf = float(r.get("corsi_for", 0))
                cur_ca = float(r.get("corsi_against", 0))
                cur_cf_pct = round(100.0 * cur_cf / (cur_cf + cur_ca), 1) if (cur_cf + cur_ca) > 0 else 50.0
                cf_delta_vs_l5 = round(cur_cf_pct - l5_cf_pct, 1) if l5_n > 0 else 0.0

                if l5_n >= 2 and (l5_w / l5_n >= 0.75 or (l5_w / l5_n >= 0.6 and l5_cf_pct >= 52.0)):
                    l5_form = "HOT_SURGE"
                elif l5_n >= 2 and (l5_w / l5_n <= 0.25 or (l5_w / l5_n <= 0.4 and l5_cf_pct <= 47.0)):
                    l5_form = "COLD_SLUMP"
                elif l5_cf_pct >= 51.5:
                    l5_form = "STEADY_PROCESS"
                else:
                    l5_form = "BALANCED"

                # Line Shift Summary for (team, gid)
                tg_lines = lines_by_tg.get((str(team), gid), pd.DataFrame())
                has_shifts = not tg_lines.empty
                line_cont_pct = 100.0
                new_lines_cnt = 0
                in_game_blender = False

                top_line_p = top_line_toi = top_line_verdict = ""
                top_line_cf = top_line_imp = 50.0
                top_line_gf = top_line_ga = 0

                best_line_p = best_line_lbl = best_line_toi = best_line_verdict = best_line_story = ""
                best_line_cf = best_line_imp = 50.0
                best_line_gf = best_line_ga = 0

                worst_line_p = worst_line_lbl = worst_line_toi = worst_line_verdict = worst_line_story = ""
                worst_line_cf = worst_line_imp = 50.0
                worst_line_gf = worst_line_ga = 0

                top_pair_p = top_pair_toi = top_pair_story = ""
                top_pair_cf = top_pair_imp = 50.0
                top_pair_gf = top_pair_ga = 0

                if has_shifts:
                    fwds_df = tg_lines[tg_lines["unit_type"] == "3F"].sort_values("unit_rank")
                    defs_df = tg_lines[tg_lines["unit_type"] == "2D"].sort_values("unit_rank")

                    if not fwds_df.empty:
                        tot_f_toi = max(1, int(fwds_df["toi_sec"].sum()))
                        est_toi = int(
                            fwds_df[
                                fwds_df["line_change_status"].isin(["ESTABLISHED_UNIT", "SEASON_DEBUT_BASELINE"])
                            ]["toi_sec"].sum()
                        )
                        line_cont_pct = round(100.0 * est_toi / tot_f_toi, 1)
                        new_lines_cnt = int(
                            fwds_df["line_change_status"].isin(["NEW_LINE_COMBO", "IN_GAME_SHUFFLE"]).sum()
                        )
                        in_game_blender = bool(
                            (fwds_df["line_change_status"] == "IN_GAME_SHUFFLE").any()
                            or (fwds_df["p3_deployment_shift_pct"].abs() >= 10.0).any()
                        )

                        # Top line by TOI (Rank 1)
                        tl = fwds_df.iloc[0]
                        top_line_p = str(tl["unit_players"])
                        top_line_toi = str(tl["toi_formatted"])
                        top_line_cf = float(tl["cf_pct"])
                        top_line_gf = int(tl["gf"])
                        top_line_ga = int(tl["ga"])
                        top_line_imp = float(tl["line_impact_score"])
                        top_line_verdict = str(tl["l5_trend_verdict"])

                        # Best & Worst lines among top 4 regular lines (min 180s TOI if available)
                        reg_fwds = fwds_df[fwds_df["toi_sec"] >= 180]
                        if reg_fwds.empty:
                            reg_fwds = fwds_df

                        best_row = reg_fwds.sort_values(
                            by=["line_impact_score", "goal_diff", "cf_pct"], ascending=False
                        ).iloc[0]
                        best_line_p = str(best_row["unit_players"])
                        best_line_lbl = str(best_row["unit_label"])
                        best_line_toi = str(best_row["toi_formatted"])
                        best_line_cf = float(best_row["cf_pct"])
                        best_line_gf = int(best_row["gf"])
                        best_line_ga = int(best_row["ga"])
                        best_line_imp = float(best_row["line_impact_score"])
                        best_line_verdict = str(best_row["l5_trend_verdict"])
                        best_line_story = str(best_row["line_story_summary"])

                        worst_row = reg_fwds.sort_values(
                            by=["line_impact_score", "goal_diff", "cf_pct"], ascending=True
                        ).iloc[0]
                        worst_line_p = str(worst_row["unit_players"])
                        worst_line_lbl = str(worst_row["unit_label"])
                        worst_line_toi = str(worst_row["toi_formatted"])
                        worst_line_cf = float(worst_row["cf_pct"])
                        worst_line_gf = int(worst_row["gf"])
                        worst_line_ga = int(worst_row["ga"])
                        worst_line_imp = float(worst_row["line_impact_score"])
                        worst_line_verdict = str(worst_row["l5_trend_verdict"])
                        worst_line_story = str(worst_row["line_story_summary"])

                    if not defs_df.empty:
                        tp = defs_df.iloc[0]
                        top_pair_p = str(tp["unit_players"])
                        top_pair_toi = str(tp["toi_formatted"])
                        top_pair_cf = float(tp["cf_pct"])
                        top_pair_gf = int(tp["gf"])
                        top_pair_ga = int(tp["ga"])
                        top_pair_imp = float(tp["line_impact_score"])
                        top_pair_story = str(tp["line_story_summary"])

                rows.append(
                    {
                        "season": season,
                        "gameid": gid,
                        "game_date": g_date,
                        "team": str(team),
                        "opponent": opp,
                        "home_away": home_away,
                        "win": int(r.get("win_for", 0)),
                        "goals_for": int(r.get("goals_for", 0)),
                        "goals_against": int(r.get("goals_against", 0)),
                        "rest_days": rest_days,
                        "is_back_to_back": is_b2b,
                        "opponent_rest_days": opp_rest,
                        "rest_advantage": rest_days - opp_rest,
                        "consec_road_games": consec_road,
                        "consec_home_games": consec_home,
                        "season_gp": int(r.get("rgame_for", idx + 1)),
                        "season_wins": int(r.get("rwin_for", 0)),
                        "season_wp": round(100.0 * float(r.get("wp", 0.5)), 1),
                        "season_cf_pct": round(100.0 * float(r.get("rcorsi_pct", 0.5)), 1),
                        "season_ff_pct": round(
                            100.0
                            * float(r.get("rfenwick_for", 1))
                            / max(1.0, float(r.get("rfenwick_for", 1)) + float(r.get("rfenwick_against", 1))),
                            1,
                        ),
                        "season_pythagorean_pe": round(100.0 * float(r.get("rpe_for", 0.5)), 1),
                        "season_rpi": round(100.0 * float(r.get("rpi", 0.5)), 1),
                        "l5_games": l5_n,
                        "l5_wins": l5_w,
                        "l5_losses": l5_l,
                        "l5_record": l5_rec,
                        "l5_gf_per_game": l5_gf_pg,
                        "l5_ga_per_game": l5_ga_pg,
                        "l5_goal_diff": l5_gd,
                        "l5_cf_pct": l5_cf_pct,
                        "l5_ff_pct": l5_ff_pct,
                        "l5_sf_pct": l5_sf_pct,
                        "l5_hits_per_game": l5_hits_pg,
                        "l5_blocks_per_game": l5_blocks_pg,
                        "l5_turnover_diff": l5_to_diff,
                        "game_cf_pct": cur_cf_pct,
                        "game_vs_l5_cf_delta": cf_delta_vs_l5,
                        "l5_form_status": l5_form,
                        "has_shift_data": has_shifts,
                        "line_continuity_pct": line_cont_pct,
                        "new_lines_deployed_count": new_lines_cnt,
                        "in_game_line_blender": in_game_blender,
                        "top_line_players": top_line_p,
                        "top_line_toi": top_line_toi,
                        "top_line_cf_pct": top_line_cf,
                        "top_line_gf": top_line_gf,
                        "top_line_ga": top_line_ga,
                        "top_line_impact": top_line_imp,
                        "top_line_l5_verdict": top_line_verdict,
                        "best_line_players": best_line_p,
                        "best_line_label": best_line_lbl,
                        "best_line_toi": best_line_toi,
                        "best_line_cf_pct": best_line_cf,
                        "best_line_gf": best_line_gf,
                        "best_line_ga": best_line_ga,
                        "best_line_impact": best_line_imp,
                        "best_line_l5_verdict": best_line_verdict,
                        "best_line_story": best_line_story,
                        "worst_line_players": worst_line_p,
                        "worst_line_label": worst_line_lbl,
                        "worst_line_toi": worst_line_toi,
                        "worst_line_cf_pct": worst_line_cf,
                        "worst_line_gf": worst_line_gf,
                        "worst_line_ga": worst_line_ga,
                        "worst_line_impact": worst_line_imp,
                        "worst_line_l5_verdict": worst_line_verdict,
                        "worst_line_story": worst_line_story,
                        "top_pair_players": top_pair_p,
                        "top_pair_toi": top_pair_toi,
                        "top_pair_cf_pct": top_pair_cf,
                        "top_pair_gf": top_pair_gf,
                        "top_pair_ga": top_pair_ga,
                        "top_pair_impact": top_pair_imp,
                        "top_pair_story": top_pair_story,
                    }
                )

        df_out = pd.DataFrame(rows)
        if not df_out.empty:
            df_out = df_out.sort_values(["game_date", "gameid", "team"]).reset_index(drop=True)
        return df_out

    def run(self, seasons: Optional[List[str]] = None) -> None:
        """Execute shift and L5 trend processing across specified seasons."""
        target_seasons = seasons or self.get_available_seasons()
        all_lines = []
        all_line_l5 = []
        all_team_l5 = []

        for s in target_seasons:
            df_l, df_ll5, df_tl5 = self.process_season(str(s))
            if not df_l.empty:
                all_lines.append(df_l)
            if not df_ll5.empty:
                all_line_l5.append(df_ll5)
            if not df_tl5.empty:
                all_team_l5.append(df_tl5)

        if all_lines:
            pd.concat(all_lines, ignore_index=True).to_csv(
                os.path.join(self.output_dir, "game_line_shifts.csv"), index=False
            )
        if all_line_l5:
            pd.concat(all_line_l5, ignore_index=True).to_csv(
                os.path.join(self.output_dir, "game_line_trends_l5.csv"), index=False
            )
        if all_team_l5:
            pd.concat(all_team_l5, ignore_index=True).to_csv(
                os.path.join(self.output_dir, "team_game_trends_l5.csv"), index=False
            )
