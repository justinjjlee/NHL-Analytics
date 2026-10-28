"""
NHL Line Shift & Rolling L5 Trend Analytics Processing Pipeline.

Reconstructs 3F forward lines and 2D defensive pairings from play-by-play shift charts,
intersects on-ice stints with play-by-play events (Corsi, Fenwick, Shots, Goals, Hits,
Blocks, Turnovers, Zone Faceoffs), evaluates sequential shift momentum and in-game
period-by-period bench adjustments, and computes rolling 5-game (L5) line continuity
and improvement/regression metrics.

Saves CSVs under latest/action/shifts/:
  - {season}_game_line_shifts.csv (and master game_line_shifts.csv)
  - {season}_game_line_trends_l5.csv (and master game_line_trends_l5.csv)
  - {season}_team_game_trends_l5.csv (and master team_game_trends_l5.csv)

Usage:
  uv run python src/data/apinhle/data_proc_shifts.py
  uv run python src/data/apinhle/data_proc_shifts.py --seasons 2026
"""

import argparse
import os
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from config import (
        IS_DATABRICKS,
        get_action_shifts_dir,
        get_box_dir,
        get_play_dir,
        get_team_dir,
    )
    from function.procs_shifts import ShiftLineAnalyticsProcessor
except ImportError:
    from src.data.apinhle.config import (
        IS_DATABRICKS,
        get_action_shifts_dir,
        get_box_dir,
        get_play_dir,
        get_team_dir,
    )
    from src.data.apinhle.function.procs_shifts import ShiftLineAnalyticsProcessor


def main():
    parser = argparse.ArgumentParser(description="NHL Line Shift & Rolling L5 Trend Processing Pipeline")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all available historical shift seasons.",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=None,
        help="Season starting years to process (e.g. '2026' or 'all'). Defaults to active season.",
    )
    parser.add_argument("--play-dir", type=str, default=None, help="Directory containing play-by-play & shift CSVs.")
    parser.add_argument("--box-dir", type=str, default=None, help="Directory containing box score CSVs.")
    parser.add_argument("--team-dir", type=str, default=None, help="Directory containing team season CSVs.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (defaults to latest/action/shifts).")

    args = parser.parse_args()

    play_dir = args.play_dir or get_play_dir()
    box_dir = args.box_dir or get_box_dir()
    team_dir = args.team_dir or get_team_dir()
    output_dir = args.output_dir or get_action_shifts_dir()

    print("=" * 70)
    print("Starting NHL Line Shift & L5 Trend Analytics Pipeline")
    print(f"Environment:       {'Databricks' if IS_DATABRICKS else 'Local / CLI'}")
    print(f"Play Directory:    {play_dir}")
    print(f"Box Directory:     {box_dir}")
    print(f"Output Directory:  {output_dir}")
    print("=" * 70)

    processor = ShiftLineAnalyticsProcessor(
        play_dir=play_dir,
        box_dir=box_dir,
        team_dir=team_dir,
        output_dir=output_dir,
    )

    if getattr(args, "all", False) or (args.seasons and "all" in args.seasons):
        target_seasons = processor.get_available_seasons()
    elif args.seasons:
        target_seasons = [str(s) for s in args.seasons]
    else:
        try:
            from config import get_current_season_year
        except ImportError:
            from src.data.apinhle.config import get_current_season_year
        target_seasons = [str(get_current_season_year())]

    print(f"Target seasons ({len(target_seasons)}): {', '.join(target_seasons)}")
    t0 = time.time()

    processor.run(seasons=target_seasons)

    elapsed = time.time() - t0
    print("=" * 70)
    print(f"✅ Line Shift & L5 Trend processing completed in {elapsed:.2f}s!")
    print(f"Artifacts saved under: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
