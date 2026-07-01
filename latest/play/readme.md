# Play-by-play data - NHL API
Latest data processed. While this is primarily handled by the Github Action daily schedule, you can manually trigger an ad-hoc pull to get the latest live game data outside of the container schedule.

### Ad-hoc Data Pull Instructions
To pull the most recent play-by-play data manually into this folder, run the following commands from the repository root using the `uv` environment:

1. **Pull the latest Game Box Scores (required for game IDs):**
   ```bash
   uv run python src/data/apinhle/data_pull_box.py
   ```

2. **Pull the new Play-by-Play records:**
   ```bash
   uv run python src/data/apinhle/data_pull_plays.py
   ```

3. **Pull the shift chart (lines) data:**
   ```bash
   uv run python src/data/apinhle/data_pull_lines.py
   ```

The scripts will automatically detect existing game records in `latest/play/` and only append the new games since the last update.