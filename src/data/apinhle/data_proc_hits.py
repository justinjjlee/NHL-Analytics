"""
NHL Hit Analytics Processing Pipeline.

Processes physical hit telemetry, event sequences, reaction latencies,
spatial distributions, and downstream scoreboard outcomes across NHL seasons.

Saves season-partitioned CSVs under latest/action/hits/ and compiles
unified master datasets across the decade.

Usage:
  # Process full decade (2015-2016 to 2025-2026+):
  uv run python src/data/apinhle/data_proc_hits.py

  # Process single current season (daily incremental):
  uv run python src/data/apinhle/data_proc_hits.py --seasons 2026

  # Process incrementally (skip already cached historical seasons):
  uv run python src/data/apinhle/data_proc_hits.py --incremental

  # Process all available play-by-play files:
  uv run python src/data/apinhle/data_proc_hits.py --seasons all
"""

import os
import sys
import argparse
import time

# Ensure imports resolve whether executed from repo root or src/data/apinhle/
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from config import get_play_dir, get_action_hits_dir, IS_DATABRICKS
    from function.procs_hits import HitAnalyticsProcessor
except ImportError:
    from src.data.apinhle.config import get_play_dir, get_action_hits_dir, IS_DATABRICKS
    from src.data.apinhle.function.procs_hits import HitAnalyticsProcessor


def main():
    parser = argparse.ArgumentParser(description="NHL Hit Analytics Processing Pipeline")
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=None,
        help="Season starting years to process (e.g. '2015 2016 2024', 'decade', or 'all'). Default: decade (2015-2026)."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Force full re-computation of all seasons (disables default incremental mode)."
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        default=True,
        help="Skip re-processing historical seasons that already have output CSVs generated (default: True)."
    )
    parser.add_argument(
        "--play-dir",
        type=str,
        default=None,
        help="Path to directory containing play-by-play CSVs (defaults to config)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to output directory for action hits CSVs (defaults to latest/action/hits)."
    )
    parser.add_argument(
        "--no-legacy-sync",
        action="store_true",
        help="Disable mirroring to legacy data directories."
    )

    args = parser.parse_args()

    # Incremental mode is enabled by default unless --full is explicitly specified
    incremental_mode = not args.full

    play_dir = args.play_dir or get_play_dir()
    output_dir = args.output_dir or get_action_hits_dir()

    print("=" * 70)
    print("Starting NHL Hit Analytics Processing Pipeline")
    print(f"Environment:       {'Databricks' if IS_DATABRICKS else 'Local / CLI'}")
    print(f"Play Directory:    {play_dir}")
    print(f"Output Directory:  {output_dir}")
    print(f"Incremental Mode:  {incremental_mode}")
    print("=" * 70)

    processor = HitAnalyticsProcessor(play_dir=play_dir, output_dir=output_dir)

    # Determine seasons to process
    if args.seasons is None or "decade" in args.seasons:
        target_seasons = processor.get_decade_seasons()
    elif "all" in args.seasons:
        target_seasons = processor.get_available_seasons()
    else:
        target_seasons = [str(s) for s in args.seasons]

    print(f"Target seasons ({len(target_seasons)}): {', '.join(target_seasons)}")
    t0 = time.time()

    processor.run(
        seasons=target_seasons,
        incremental=incremental_mode,
        sync_legacy=not args.no_legacy_sync
    )

    elapsed = time.time() - t0
    print("=" * 70)
    print(f"✅ Hit Analytics processing completed in {elapsed:.2f}s!")
    print(f"Artifacts saved under: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
