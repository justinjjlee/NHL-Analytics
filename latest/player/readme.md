# Player Data - NHL API
Latest data processed. While this is primarily handled by the Github Action daily schedule, you can manually trigger an ad-hoc pull to update roster and player stats.

### Ad-hoc Data Pull Instructions
To manually pull the latest player statistics and roster data, run the following script from the repository root:

```bash
uv run python src/data/apinhle/data_pull_player.py
```
*(Note: In some local configurations, the `data_pull_player.py` script may output directly to the `latest/team/` directory depending on your `config.py` settings.)*