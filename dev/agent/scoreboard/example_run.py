"""
example_run.py — Demonstration script for the Scoreboard Agent Framework.

Runs end-to-end extraction, statistical milestone detection, and journalistic narration
for a multi-game slate and deep-dives into a thrilling single matchup.
"""
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


def run_demonstration():
    print("=" * 75)
    print("🏒  NHL SCOREBOARD AGENT FRAMEWORK — DEMONSTRATION")
    print("=" * 75)

    agent = ScoreboardAgent()

    # 1. Detect latest slate date (or test with 2026-10-13)
    target_date = "2026-10-13"
    print(f"\n[Step 1] Ingesting and collectively analyzing slate for {target_date}...")
    slate = agent.analyze_slate(target_date, save_report=True)

    print(f"\n📊 Slate Overview ({slate.date}):")
    print(f"   • Total Matchups: {slate.total_games}")
    print(f"   • Total Goals Scored: {slate.total_goals} (Avg {slate.avg_goals_per_game:.1f} per game)")
    print(f"   • Outcome Distribution: {slate.home_wins} Home Wins, {slate.away_wins} Road Wins")
    print(f"   • Overtime/Shootout Drama: {slate.ot_so_games} games went beyond regulation")

    print("\n🏆 League-Wide Three Stars of the Night:")
    for star in slate.slate_three_stars:
        print(f"   ⭐ Star #{star.star}: {star.name} ({star.team_abbrev} - {star.position}) | {star.stat_summary}")

    print("\n🔥 Key Slate Milestones Detected:")
    for n in slate.slate_notables[:4]:
        print(f"   • [{n.category.upper()}] {n.title}: {n.description}")

    # 2. Deep Dive into Single Thriller Game
    sample_game_id = 2026020090  # NJD @ DET (5-4 OT)
    print(f"\n[Step 2] Running deep-dive narrative analysis for Game {sample_game_id} (NJD @ DET)...")
    game = agent.analyze_game(sample_game_id, save_report=True)

    print(f"\n📰 Generated Headline:")
    print(f"   \"{game.headline}\"")

    print(f"\n📝 Narrative Recap:")
    for p in game.recap_narrative.split("\n\n"):
        print(f"   {p}")

    print(f"\n⭐ Game Three Stars:")
    for s in game.three_stars:
        print(f"   • Star {s.star}: {s.name} ({s.team_abbrev}) - {s.stat_summary}")

    print(f"\n📊 Head-to-Head Team Sheet:")
    print(f"   • {game.away_team.team_abbrev}: {game.away_team.score} Goals | {game.away_team.sog} SOG | PP: {game.away_team.power_play_conv} | {game.away_team.hits} Hits | {game.away_team.faceoff_pct:.1f}% FO")
    print(f"   • {game.home_team.team_abbrev}: {game.home_team.score} Goals | {game.home_team.sog} SOG | PP: {game.home_team.power_play_conv} | {game.home_team.hits} Hits | {game.home_team.faceoff_pct:.1f}% FO")

    print("\n" + "=" * 75)
    print(f"✅ Success! Reports generated and saved in:")
    print(f"   📁 {agent.output_dir}")
    print("=" * 75)


if __name__ == "__main__":
    run_demonstration()
