# Box Score and Game Summary Data
This directory stores game-level box scores, betting odds, and processed game summaries. While updates are primarily handled by the GitHub Action daily schedule, you can manually trigger an ad-hoc pull.

### Ad-hoc Data Pull Instructions
To manually update the data in this directory, run the following scripts from the repository root using the `uv` environment:

1. **Pull Game Box Scores:**
   ```bash
   uv run python src/data/apinhle/data_pull_box.py
   ```

2. **Pull Game Betting Odds:**
   ```bash
   uv run python src/data/apinhle/data_pull_box_odds.py
   ```

3. **Process Game Summaries:**
   ```bash
   uv run python src/data/apinhle/data_proc_gameSummary.py
   ```
