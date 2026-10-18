"""
data_proc_xg.py — CLI & Pipeline Runner for Expected Goals (xG), Individual xG (ixG),
and Goals Saved Above Expected (GSAx) across Team and Player Game & Season Statistics.

Usage:
    # Process active seasons from config.py
    uv run python src/data/apinhle/data_proc_xg.py

    # Process all historical seasons with play-by-play data (2013-2026)
    uv run python src/data/apinhle/data_proc_xg.py --all

    # Process specific seasons
    uv run python src/data/apinhle/data_proc_xg.py --seasons 2024 2025
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from function.procs_xg import enrich_season_xg


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute xG, ixG, and GSAx across NHL team and player datasets.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all available play-by-play seasons in latest/play/.",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=None,
        help="Explicit list of season start years to process (e.g. 2024 2025).",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="./latest",
        help="Base data directory (default: ./latest).",
    )
    args = parser.parse_args()

    if args.seasons:
        seasons = sorted(args.seasons)
    elif args.all:
        pbp_files = sorted(glob.glob(os.path.join(args.base_dir, "play", "*_playbyplay.csv")))
        seasons = []
        for f in pbp_files:
            yr_str = os.path.basename(f)[:4]
            if yr_str.isdigit():
                seasons.append(int(yr_str))
    else:
        seasons = list(config.SEASON_YEARS)

    for yr in seasons:
        res = enrich_season_xg(yr, base_dir=args.base_dir)
        if res.get("status", 1) == 0:
            print(f"[{yr}] Skipped (no play-by-play data found).")
        else:
            print(
                f"[{yr}] xG & GSAx estimation complete -> "
                f"box={res.get('box_rows', 0)}, "
                f"gameStats={res.get('game_stats_rows', 0)}, "
                f"teamSeason={res.get('team_season_rows', 0)}, "
                f"playerGame={res.get('box_player_rows', 0)}, "
                f"playerSeason={res.get('player_season_rows', 0)}"
            )


if __name__ == "__main__":
    main()
