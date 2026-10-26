"""
Launcher for Rinklytics Streamlit Dashboard on Hugging Face Spaces and local environments.
"""
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path before importing scoreboard modules
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Required for Hugging Face ZeroGPU static scanner & runtime registration
try:
    import spaces  # noqa: F401
    from dev.agent.scoreboard.narrator import ensure_zerogpu_startup_report

    ensure_zerogpu_startup_report()
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SpaceLauncher")


def _discover_recent_completed_dates(limit: int = 2) -> list[str]:
    """Finds the most recent completed game dates from remote Hugging Face dataset, local box CSVs, or the NHL API."""
    dates: list[str] = []

    # 1. Primary: Discover completed dates from Hugging Face dataset
    try:
        url = "https://huggingface.co/datasets/jjerlee/nhl-analytics/resolve/main/data/games.parquet"
        df = pd.read_parquet(url, columns=["date"])
        if not df.empty and "date" in df.columns:
            found = sorted(pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d").dropna().unique().tolist(), reverse=True)
            for d in found[:limit]:
                if d not in dates:
                    dates.append(d)
    except Exception:
        pass

    # 2. Local fallback if available
    if len(dates) < limit:
        try:
            box_dir = Path(ROOT_DIR) / "latest" / "box"
            if box_dir.exists():
                for csv_file in sorted(box_dir.glob("*_box.csv"), reverse=True):
                    df = pd.read_csv(csv_file, usecols=lambda c: c in ("date", "gameDate"))
                    d_col = "date" if "date" in df.columns else ("gameDate" if "gameDate" in df.columns else None)
                    if d_col and not df.empty:
                        found = sorted(pd.to_datetime(df[d_col]).dt.strftime("%Y-%m-%d").dropna().unique().tolist(), reverse=True)
                        for d in found:
                            if d not in dates:
                                dates.append(d)
                    if len(dates) >= limit:
                        break
        except Exception as e:
            logger.warning("Box CSV date discovery warning: %s", e)

    # 3. Fallback to live NHL API
    try:
        from dev.agent.scoreboard.agent import ScoreboardAgent
        api_latest = ScoreboardAgent().get_latest_date()
        if api_latest and api_latest not in dates:
            dates.insert(0, api_latest)
    except Exception:
        pass

    dates.sort(reverse=True)
    return dates[:limit]


def _space_auto_synthesis_worker():
    """Background daemon on Hugging Face Spaces that automatically upgrades recent slates to ZeroGPU LLM recaps."""
    # Allow Streamlit server to bind port 7860 first so container health check passes immediately
    time.sleep(8)

    try:
        from dev.agent.scoreboard.env_utils import is_local_streamlit
        if is_local_streamlit():
            return
    except Exception:
        return

    logger.info("Starting Hugging Face Space ZeroGPU auto-synthesis background monitor...")

    while True:
        try:
            from dev.agent.scoreboard.agent import ScoreboardAgent
            from dev.agent.scoreboard.dataset_store import HFScoreboardStore, get_model_priority

            store = HFScoreboardStore()
            store._load_tables(refresh_remote=True)
            recent_dates = _discover_recent_completed_dates(limit=2)
            lock_file = store.local_dir / ".synthesis.lock"

            for target_date in recent_dates:
                store._load_tables(refresh_remote=True)
                slate_is_llm = store.is_slate_llm_generated(target_date)
                existing_games = store.get_games_for_date(target_date)
                games_are_llm = bool(
                    existing_games
                    and all(
                        get_model_priority(g.get("generator_model")) >= 2
                        and str(g.get("headline", "")).strip().lower() not in ("", "headline", "headline:", "title", "title:")
                        and len(str(g.get("narrative_recap", "")).strip()) >= 1100
                        for g in existing_games
                    )
                )

                if slate_is_llm and games_are_llm:
                    logger.info("Date %s already has verified Tier-2 LLM recaps (%s). Skipping.", target_date, store.get_slate_generator_model(target_date))
                    continue

                logger.info(
                    "Auto-synthesizing ZeroGPU LLM recaps for %s (current model: %s)...",
                    target_date,
                    store.get_slate_generator_model(target_date) or "unrecorded",
                )
                try:
                    lock_file.write_text(
                        json.dumps({"date": target_date, "started_at": time.time()}),
                        encoding="utf-8",
                    )
                except Exception:
                    pass

                try:
                    agent = ScoreboardAgent(model_mode="auto")
                    slate = agent.analyze_slate(target_date, save_report=False, game_narrative_mode="auto")
                    if slate and slate.games:
                        slate_model = getattr(slate, "generator_model", "algorithmic-guardrails-v1")
                        # Determine highest priority game model achieved across the slate
                        game_models = [getattr(g, "generator_model", "algorithmic-guardrails-v1") for g in slate.games]
                        best_game_model = max(game_models, key=get_model_priority) if game_models else slate_model
                        is_tier2 = get_model_priority(best_game_model) >= 2

                        store._load_tables(refresh_remote=True)
                        store.record_slate(slate, generator_model=slate_model, force=is_tier2)
                        store.record_games(slate.games, generator_model=best_game_model, force=is_tier2)


                        if store.token:
                            sync_res = store.sync_to_huggingface(
                                commit_message=f"Auto-synthesize ZeroGPU LLM recaps for {target_date} ({best_game_model})"
                            )
                            logger.info("Synced %s recaps to Hugging Face dataset: %s", target_date, sync_res)
                            # Purge local temporary cache tables after successful sync
                            store.clear_local_cache()
                finally:
                    try:
                        if lock_file.exists():
                            lock_file.unlink()
                    except Exception:
                        pass
                    # Immediately release GPU allocation and temporary memory
                    try:
                        import gc
                        import torch
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

        except Exception as e:
            logger.error("Error in Space auto-synthesis worker: %s", e)

        # Re-check every 20 minutes for newly completed slates
        time.sleep(1200)


if __name__ == "__main__":
    if not os.environ.get("DISABLE_SPACE_AUTO_SYNTHESIS"):
        worker = threading.Thread(target=_space_auto_synthesis_worker, name="ZeroGPUAutoSynthesis", daemon=True)
        worker.start()

    app_file = os.path.join(ROOT_DIR, "app", "app.py")
    port = os.environ.get("PORT", "7860")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_file,
        f"--server.port={port}",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
    ]
    subprocess.run(cmd)

