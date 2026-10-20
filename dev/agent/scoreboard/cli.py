"""
cli.py — Command-Line Interface for the Scoreboard Agent Framework.

Usage:
    python dev/agent/scoreboard/cli.py --latest
    python dev/agent/scoreboard/cli.py --date 2026-10-13
    python dev/agent/scoreboard/cli.py --game-id 2026020090
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# Add current directory and repo root to Python path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

repo_root = current_dir.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from dev.agent.scoreboard.agent import ScoreboardAgent
except (ImportError, ValueError):
    from agent import ScoreboardAgent


def main():
    parser = argparse.ArgumentParser(
        description="NHL Scoreboard Agent — Automated game recaps, player standouts, and slate analytics."
    )
    parser.add_argument("--date", type=str, default=None, help="Slate date to analyze (YYYY-MM-DD).")
    parser.add_argument("--latest", action="store_true", help="Automatically analyze the most recent completed slate.")
    parser.add_argument("--game-id", type=int, default=None, help="Analyze a specific single game ID (e.g., 2026020090).")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save generated markdown recaps.")
    parser.add_argument("--no-save", action="store_true", help="Do not write markdown reports to disk.")
    parser.add_argument("--print-recap", action="store_true", help="Print the generated narrative directly to stdout.")
    parser.add_argument("--llm-prompt", action="store_true", help="Generate and print an LLM synthesis prompt.")
    parser.add_argument("--record-hf", action="store_true", help="Record and sync recaps directly to Hugging Face dataset (jjerlee/nhl-analytics).")
    parser.add_argument("--mode", type=str, choices=["auto", "lm-studio", "local-llm", "llm", "router-llm", "algorithmic"], default="auto", help="Narrative generator mode: 'lm-studio' (Local LM Studio at http://127.0.0.1:1234), 'router-llm' (Meta-Llama-3.2-3B via router), 'algorithmic' (deterministic guardrails), or 'auto' (LM Studio with fallback).")
    parser.add_argument("--lm-studio-url", type=str, default="http://127.0.0.1:1234/v1", help="Base URL for local LM Studio API (default: http://127.0.0.1:1234/v1).")
    parser.add_argument("--lm-model", type=str, default=None, help="Explicit model ID in LM Studio (default: auto-detect from /v1/models).")
    parser.add_argument("--all-games-llm", action="store_true", help="Generate full LLM narratives for every individual game on the slate (default: generates executive slate review with LLM and individual game recaps with Algorithmic Engine for fast batch performance).")
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Hugging Face model ID for router LLM generation.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose debug logging.")

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    agent = ScoreboardAgent(
        output_dir=args.output_dir,
        model_mode=args.mode,
        lm_studio_url=args.lm_studio_url,
        lm_model_id=args.lm_model,
        llm_model_id=args.model_id,
    )

    # 1. Single Game Analysis
    if args.game_id:
        print(f"\n🏒 Running Scoreboard Agent for Game ID: {args.game_id}...")
        game = agent.analyze_game(args.game_id, save_report=not args.no_save)
        print(f"\n✅ Analysis Complete: {game.headline}")
        print(f"Final Score: {game.away_team.team_abbrev} {game.away_team.score} @ {game.home_team.team_abbrev} {game.home_team.score} ({game.period_type})")
        print("\nThree Stars:")
        for s in game.three_stars:
            print(f"  ⭐ {s.star}. {s.name} ({s.team_abbrev} - {s.position}): {s.stat_summary}")

        if game.notables:
            print("\nKey Notables:")
            for n in game.notables:
                print(f"  🔥 [{n.category.upper()}] {n.title} -> {n.description}")

        if args.print_recap:
            print("\n" + "=" * 70)
            print(agent.narrator.generate_single_game_markdown(game))
            print("=" * 70)
        return

    # 2. Slate-wide Analysis
    target_date = args.date
    if args.latest or not target_date:
        target_date = agent.get_latest_date()

    print(f"\n🏒 Running Scoreboard Agent for Slate Date: {target_date}...")
    game_mode = "algorithmic" if args.mode == "algorithmic" else None
    slate = agent.analyze_slate(target_date, save_report=not args.no_save, game_narrative_mode=game_mode)

    print(f"\n✅ Slate Analysis Complete for {slate.date}:")
    print(f"   • Total Games: {slate.total_games} | Total Goals: {slate.total_goals} (Avg {slate.avg_goals_per_game:.1f})")
    print(f"   • Home Wins: {slate.home_wins} | Away Wins: {slate.away_wins} | OT/SO Games: {slate.ot_so_games}")

    if slate.slate_three_stars:
        print("\n🏆 League-Wide Three Stars of the Night:")
        for s in slate.slate_three_stars:
            print(f"   ⭐ {s.star}. {s.name} ({s.team_abbrev} - {s.position}): {s.stat_summary}")

    if slate.slate_notables:
        print(f"\n🔥 Top Milestones & Highlights ({len(slate.slate_notables)} detected):")
        for n in slate.slate_notables[:5]:
            print(f"   • {n.title} ({n.entity}): {n.description}")

    print(f"\n📁 Reports directory: {agent.output_dir}")

    if args.record_hf:
        print("\n☁️ Recording and syncing recaps directly to Hugging Face dataset (jjerlee/nhl-analytics)...")
        try:
            from dev.agent.scoreboard.dataset_store import HFScoreboardStore
        except ImportError:
            from dataset_store import HFScoreboardStore
        store = HFScoreboardStore()
        store._load_tables(refresh_remote=True)
        slate_model = getattr(slate, "generator_model", "algorithmic-guardrails-v1")
        game_model = (
            getattr(slate.games[0], "generator_model", "algorithmic-guardrails-v1")
            if slate.games
            else slate_model
        )
        store.record_slate(slate, generator_model=slate_model)
        store.record_games(slate.games, generator_model=game_model)
        print("✅ Finished recording to Hugging Face dataset store!")

        # If running in a headless CI environment that only produced Tier-1 algorithmic recaps,
        # wake/restart the Hugging Face Space so its ZeroGPU worker upgrades the date to Tier-2 LLM recaps.
        if not store.is_slate_llm_generated(target_date):
            try:
                import requests
                from huggingface_hub import HfApi
                if store.token:
                    HfApi(token=store.token).restart_space("jjerlee/rinklytics")
                    print("⚡ Triggered Hugging Face Space (jjerlee/rinklytics) ZeroGPU worker to upgrade recaps to LLM.")
                else:
                    requests.get("https://jjerlee-rinklytics.hf.space", timeout=10)
            except Exception as e:
                print(f"ℹ️ Note waking Hugging Face Space: {e}")


    if args.llm_prompt:
        print("\n" + "-" * 70)
        print("🤖 Generated LLM Synthesis Prompt:")
        print(agent.generate_llm_synthesis_prompt(slate))
        print("-" * 70)

    if args.print_recap:
        print("\n" + "=" * 70)
        print(agent.narrator.generate_slate_markdown(slate))
        print("=" * 70)


if __name__ == "__main__":
    main()
