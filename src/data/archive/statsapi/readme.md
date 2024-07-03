# About statsapi
As of 2024, NHL has shut down `statsapi` and its tabular format. Codes in this repository are archived and will not be maintained.

## Data Processing NHL API data for analysis

This workflow is written to use the latest season of NHL data, which is acquired through NHL API stream and workflow I wrote [here](https://github.com/justinjoliver/NHL-Analytics/blob/main/src/NHL_gameData_pull.py). Note that prior to 2011 season, NHL only recorded goals and penalities as game-level events. For historical data points, I use my web-scraping workflow I coded [here](https://github.com/justinjoliver/NHL-Analytics/blob/main/src/NHL_summary_data_pull.py), which does not provide as granular information as in recent time. Given the recency and relevancy, historical data before the decade is only needed as a reference.

The following workflow can be adjusted to process historical data back to 2011.

## Data Processing
* Data pull from the official NHL API
* Data baseline process to create tables of - play-by-play data, team data, and player data
* Data evaluation of team performance metrics
* Data creation of game, play-by-play sequence data

The following workflow can be adjusted to be used to explore docker proc