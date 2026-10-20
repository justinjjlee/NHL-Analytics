"""
agent.py — Primary Orchestrator for the Scoreboard Agent Framework.

Coordinates data ingestion, multi-game analysis, narrative synthesis, and publication reports.
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

try:
    from .analyzer import GameAnalyzer, SlateAnalyzer
    from .client import NHLScoreboardClient
    from .env_utils import is_local_streamlit
    from .models import AnalyzedGame, AnalyzedSlate
    from .narrator import NarratorEngine
except (ImportError, ValueError):
    from analyzer import GameAnalyzer, SlateAnalyzer
    from client import NHLScoreboardClient
    from env_utils import is_local_streamlit
    from models import AnalyzedGame, AnalyzedSlate
    from narrator import NarratorEngine

logger = logging.getLogger("ScoreboardAgent")


class ScoreboardAgent:
    """Agent orchestrator for reviewing recent games, generating recaps, and detecting notable news."""

    def __init__(
        self,
        client: Optional[NHLScoreboardClient] = None,
        output_dir: Optional[str] = None,
        model_mode: str = "auto",
        lm_studio_url: str = "http://127.0.0.1:1234/v1",
        lm_model_id: Optional[str] = None,
        llm_model_id: str = "meta-llama/Llama-3.2-3B-Instruct",
        token: Optional[str] = None,
    ):
        self.client = client or NHLScoreboardClient()
        self.narrator = NarratorEngine(
            model_mode=model_mode,
            lm_studio_url=lm_studio_url,
            lm_model_id=lm_model_id,
            llm_model_id=llm_model_id,
            token=token,
        )

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            cache_base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
            self.output_dir = cache_base / "rinklytics" / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_latest_date(self) -> str:
        """Find the latest date that has completed games."""
        return self.client.find_latest_completed_slate_date()

    def analyze_game(
        self,
        game_id: int,
        save_report: bool = True,
        force_mode: Optional[str] = None,
    ) -> AnalyzedGame:
        """
        Pull boxscore, landing, and right-rail for a single game, run deep statistical
        analysis, and build the journalistic narrative.
        """
        logger.info("Ingesting game feeds for game ID: %d", game_id)
        bundle = self.client.get_game_bundle(game_id)

        analyzer = GameAnalyzer(bundle)
        analyzed_game = analyzer.analyze()

        # Build narration
        analyzed_game.headline = self.narrator.build_game_headline(analyzed_game)
        narrative, model_stamp = self.narrator.build_game_narrative(analyzed_game, force_mode=force_mode)
        analyzed_game.recap_narrative = narrative
        analyzed_game.generator_model = model_stamp
        analyzed_game.full_report_markdown = self.narrator.generate_single_game_markdown(analyzed_game)

        if save_report:
            md_content = analyzed_game.full_report_markdown
            filename = f"game_{game_id}_{analyzed_game.away_team.team_abbrev}_at_{analyzed_game.home_team.team_abbrev}.md"
            out_path = self.output_dir / filename
            out_path.write_text(md_content, encoding="utf-8")
            logger.info("Saved single game recap to %s", out_path)

        return analyzed_game

    def analyze_slate(
        self,
        date_str: Optional[str] = None,
        save_report: bool = True,
        game_narrative_mode: Optional[str] = None,
    ) -> AnalyzedSlate:
        """
        Analyze all completed games on a specific date (or the latest completed slate).
        """
        if not date_str:
            date_str = self.get_latest_date()

        logger.info("Fetching slate games for date: %s", date_str)
        score_data = self.client.get_score_slate(date_str)
        games_list = score_data.get("games", [])

        completed_games = [
            g for g in games_list
            if g.get("gameState") in ("OFF", "FINAL")
        ]

        if not completed_games:
            logger.warning("No completed games found on %s (%d listed games). Skipping unplayed games.", date_str, len(games_list))
            target_games = []
        else:
            target_games = completed_games

        logger.info("Analyzing %d games on %s...", len(target_games), date_str)
        analyzed_games: List[AnalyzedGame] = []

        effective_mode = game_narrative_mode or self.narrator.model_mode
        use_batch_zerogpu = (
            not is_local_streamlit()
            and self.narrator._can_use_zerogpu()
            and effective_mode != "algorithmic"
        )

        if use_batch_zerogpu and target_games:
            for g in target_games:
                game_id = g.get("id")
                if not game_id:
                    continue
                try:
                    logger.info("Ingesting game feeds for game ID: %d", game_id)
                    bundle = self.client.get_game_bundle(game_id)
                    analyzed = GameAnalyzer(bundle).analyze()
                    analyzed.headline = self.narrator.build_game_headline(analyzed)
                    analyzed_games.append(analyzed)
                except Exception as e:
                    logger.error("Error analyzing game %s: %s", game_id, e)

            if analyzed_games:
                self.narrator.populate_games_narratives_zerogpu_batch(analyzed_games, batch_size=4)
        else:
            for g in target_games:
                game_id = g.get("id")
                if not game_id:
                    continue
                try:
                    analyzed = self.analyze_game(game_id, save_report=False, force_mode=game_narrative_mode)
                    analyzed_games.append(analyzed)
                except Exception as e:
                    logger.error("Error analyzing game %s: %s", game_id, e)

        # Collective Slate Analysis
        slate_analyzer = SlateAnalyzer(date_str, analyzed_games)
        slate = slate_analyzer.analyze()

        slate_narrative, slate_model = self.narrator.build_slate_narrative(slate, force_mode=game_narrative_mode)
        slate.executive_summary = slate_narrative
        slate.generator_model = slate_model

        if save_report:
            md_content = self.narrator.generate_slate_markdown(slate)
            filename = f"slate_{date_str}_night_in_review.md"
            out_path = self.output_dir / filename
            out_path.write_text(md_content, encoding="utf-8")
            logger.info("Saved slate recap report to %s", out_path)

        return slate


    def generate_llm_synthesis_prompt(self, slate: AnalyzedSlate) -> str:
        """
        Produce a high-context prompt for downstream LLMs to create tailored commentary.
        """
        prompt = [
            f"You are a veteran NHL sports editor analyzing the games played on {slate.date}.",
            f"Key metrics: {slate.total_games} games, {slate.total_goals} goals (avg {slate.avg_goals_per_game:.1f}), {slate.ot_so_games} OT/SO finishes.",
            "\nTop Three Stars of the Night:",
        ]
        for s in slate.slate_three_stars:
            prompt.append(f"- {s.star}. {s.name} ({s.team_abbrev} - {s.position}): {s.stat_summary}")

        prompt.append("\nNotable Highlights Across the League:")
        for n in slate.slate_notables:
            prompt.append(f"- {n.title}: {n.description}")

        prompt.append("\nGames on the Slate:")
        for g in slate.games:
            prompt.append(f"- {g.away_team.team_abbrev} {g.away_team.score} @ {g.home_team.team_abbrev} {g.home_team.score} ({g.period_type}): {g.headline}")

        prompt.append("\nPlease write an engaging, insightful 3-paragraph executive sports column capturing the momentum, surprises, and storylines.")
        return "\n".join(prompt)
