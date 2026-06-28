# Skillset 6: Player Draft Analysis and Lifecycle

This skillset covers the techniques used to ingest, match, and analyze NHL draft picks, tracking their lifecycle from draft day to their professional career performance.

---

## 1. Scope & Target Measurement Level
* **Measurement Level:** Player Level, Draft Class Level, Season Level.
* **Objective:** Pull historical drafted player records and efficiently cross-reference them with season stats (points leader proxy) to identify which draft picks successfully transition to the NHL, calculating metrics such as survival rates and time-to-debut.

---

## 2. Record Linkage & Data Aggregation Workflow

### 2.1 The Three-Pillar Data Pull Process
To construct the draft lifecycle dataset without overwhelming the NHL API, a three-step cached ingestion process is employed (`dev/player/player_draft/data/dataproc.py`):
1. **Draft Records (`all_drafted_players.csv`)**: 
   Iteratively pulls from `https://api-web.nhle.com/v1/draft/picks/{year}/all`. The code uses a caching check against the existing local CSV to only request years missing from the local store.
2. **Player Stats (`all_player_stats.csv`)**: 
   Pulls the `skater-stats-leaders/{season}/2?categories=points&limit=-1` API. This is used as a proxy to identify which players were active in a given NHL season. Similarly cached to skip historical seasons already downloaded.
3. **Player Chronicle (`all_drafted_players_chronicle.csv`)**:
   Tracks season-over-season performance for matched players using `https://api-web.nhle.com/v1/player/{player_id}/landing`. Caching bypasses IDs already processed.

### 2.2 Matching Heuristics
* **Join Keys:** Drafted players are matched against seasonal skater stats using `firstName` and `lastName`.
* **Filtration:** Players with a missing `id` post-join are deemed as not having debuted or having zero statistical footprint in the NHL dataset.

---

## 3. Analytical Methodologies

### 3.1 Draftee Survival Curve
Evaluates the proportion of drafted players within a given timeframe that eventually reach the NHL and play a meaningful number of games (e.g., threshold of 40 games).
* **Metric:** % of players active $Y$ years since draft.
* **Time-to-debut ($Y_{since\_draft}$):** $Y_{game} - Y_{draft}$.
* **Cumulative Average Curve:** For cohorts grouped by era, tracking the cumulative count of players who have debuted and surpassed the 40-game mark per elapsed year.

### 3.2 Draft Position Evaluation
Evaluating picks based on expected vs. actual yield:
* Analyzing the breakdown of picks in standard ranges (e.g., Picks 1-5, 6-10, 11+) to quantify the drop-off in hit rate.
* Measuring how first-round picks perform differently in their first 3 years vs. their 4th-6th years in the league compared to later rounds.

---

## 4. Weaknesses, Caveats & Considerations
1. **Name Matching Vulnerabilities:** Using just `firstName` and `lastName` to join the draft list to skater stats can lead to false positives (if two players share a name) or false negatives (if one dataset uses a full name and the other a nickname, e.g., "Matt" vs "Matthew"). 
2. **Goalies Excluded:** The core analysis logic and API points specified here primarily fetch skater stats (points leaders), inherently excluding goaltenders from the success metrics.
3. **API Rate Limiting:** Even with caching in place, pulling the chronicle data for every player involves a high volume of singular HTTP requests. Throttling (`time.sleep()`) and caching are essential to prevent HTTP 429 Too Many Requests errors.
