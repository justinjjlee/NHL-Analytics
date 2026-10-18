"""
client.py — Modern NHL API Client for the Scoreboard Agent Framework.

Interfaces with the official NHL API (api-web.nhle.com/v1) to fetch schedule slates,
game landings, box scores, team statistics, and player performance feeds.
"""
import json
import logging
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ScoreboardAgent.Client")


class NHLScoreboardClient:
    """HTTP Client for querying NHL game-center and schedule endpoints."""

    BASE_URL = "https://api-web.nhle.com/v1"
    DEFAULT_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }

    def __init__(self, timeout: int = 15, cache_ttl_seconds: int = 300):
        self.timeout = timeout
        self.cache_ttl = cache_ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _get(self, endpoint: str) -> Dict[str, Any]:
        """Perform cached HTTP GET request with retries and exponential backoff."""
        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        now = time.time()

        # Check in-memory cache
        if url in self._cache:
            entry = self._cache[url]
            if now - entry["timestamp"] < self.cache_ttl:
                return entry["data"]

        last_error = None
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers=self.DEFAULT_HEADERS)
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    content = response.read().decode("utf-8")
                    data = json.loads(content)
                    self._cache[url] = {"timestamp": now, "data": data}
                    return data
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
                last_error = e
                wait_time = 0.5 * (2 ** attempt)
                logger.warning("HTTP error fetching %s (attempt %d/3): %s. Retrying in %.1fs", url, attempt + 1, e, wait_time)
                time.sleep(wait_time)

        raise RuntimeError(f"Failed to fetch {url} after 3 attempts: {last_error}")

    def get_score_slate(self, date_str: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch the scoreboard schedule for a given date (YYYY-MM-DD) or current date.
        """
        endpoint = f"score/{date_str}" if date_str else "score/now"
        return self._get(endpoint)

    def get_boxscore(self, game_id: int) -> Dict[str, Any]:
        """Fetch the official game box score including player-level game stats."""
        return self._get(f"gamecenter/{game_id}/boxscore")

    def get_landing(self, game_id: int) -> Dict[str, Any]:
        """Fetch game landing details, three stars, scoring plays, and period breakdowns."""
        return self._get(f"gamecenter/{game_id}/landing")

    def get_right_rail(self, game_id: int) -> Dict[str, Any]:
        """Fetch supplementary team metrics (faceoffs, power play, hits, blocks, takeaways)."""
        try:
            return self._get(f"gamecenter/{game_id}/right-rail")
        except Exception as e:
            logger.debug("Right-rail unavailable for game %d: %s", game_id, e)
            return {}

    def get_play_by_play(self, game_id: int) -> Dict[str, Any]:
        """Fetch play-by-play shot telemetry, coordinates, and event timeline."""
        try:
            return self._get(f"gamecenter/{game_id}/play-by-play")
        except Exception as e:
            logger.debug("Play-by-play unavailable for game %d: %s", game_id, e)
            return {}

    def get_game_bundle(self, game_id: int) -> Dict[str, Any]:
        """
        Aggregate boxscore, landing, right-rail, and play-by-play feeds into a consolidated game bundle.
        """
        landing = self.get_landing(game_id)
        boxscore = self.get_boxscore(game_id)
        right_rail = self.get_right_rail(game_id)
        play_by_play = self.get_play_by_play(game_id)

        return {
            "game_id": game_id,
            "landing": landing,
            "boxscore": boxscore,
            "right_rail": right_rail,
            "play_by_play": play_by_play,
        }

    def find_latest_completed_slate_date(self, max_days_back: int = 10) -> str:
        """
        Scan backwards from today to find the most recent date with completed games.
        """
        today = datetime.now()
        for i in range(max_days_back):
            check_date = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            try:
                data = self.get_score_slate(check_date)
                games = data.get("games", [])
                completed_games = [
                    g for g in games
                    if g.get("gameState") in ("OFF", "FINAL")
                ]
                if completed_games:
                    logger.info("Found %d completed games on %s", len(completed_games), check_date)
                    return check_date
            except Exception as e:
                logger.debug("Error checking date %s: %s", check_date, e)

        # Fallback to current date
        return today.strftime("%Y-%m-%d")
