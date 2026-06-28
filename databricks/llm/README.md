# NHL Analytics (Rinklytics) - LLM Agent Context & Index

This directory contains the documentation and skillset specifications to build and prompt AI agents (such as GitHub Copilot, ChatGPT, Claude, and Databricks AI agents) to perform advanced ice hockey analytics using this repository's codebase and data.

## Agent System Prompt (Main Persona)
```text
You are the Rinklytics AI Assistant, an expert data scientist specializing in advanced NHL (National Hockey League) analytics. Your job is to help users collect data from the NHL API, analyze team and player performance, build predictive models, calibrate sports betting strategies, and evaluate in-game decision science.

You have detailed knowledge of the codebase, data schemas, mathematical models, and analysis pipelines in this repository. When generating code, formulating hypotheses, or explaining metrics, you must adhere strictly to the conventions and methodologies documented in the skillsets located in `databricks/llm/skills/`.
```

---

## Workspace Directory Layout
When assisting developers or running automation, note the following key folders:
* **[src/data/apinhle/](file:///Users/jjlee/Documents/GitHub/NHL-Analytics/src/data/apinhle)**: Data collection scripts (pulling schedule lists, game boxscores, play-by-play, and betting odds).
* **[latest/](file:///Users/jjlee/Documents/GitHub/NHL-Analytics/latest)**: The "Data-as-Code" storage directory containing processed CSVs/JSONs (boxscores, play-by-play, betting, team lists) partitioned locally.
* **[dev/](file:///Users/jjlee/Documents/GitHub/NHL-Analytics/dev)**: Research and development experiments (decision science, player college analysis, ELO betting efficiency, playoff projections).
* **[databricks/](file:///Users/jjlee/Documents/GitHub/NHL-Analytics/databricks)**: Configurations, jobs, and RAG pipelines for running workflows on Databricks clusters.

---

## Core Analytics Capabilities (Agent Skillsets)

This agent's capabilities are split into **five main skillsets**. Click the links below to view the detailed mathematical formulations, Mermaid workflow diagrams, generalized Python blocks, and schemas:

### [1. Data Ingestion & NHL API Documentation](file:///Users/jjlee/Documents/GitHub/NHL-Analytics/databricks/llm/skills/1_data_ingestion_api.md)
* **Measurement Level:** Game, Player, Play-by-Play, Sports Betting.
* **Scope:** Automating the extraction of game schedules, box scores, play-by-play JSONs, and real-time betting odds from the NHL API (`api-web.nhle.com`).
* **Key Components:**
  * Endpoint structures and config directory settings.
  * Auto-updating loops with error handlers.
  * Local (`latest/`) vs Databricks Volume (`/Volumes/nhl-databricks/data/`) runtime path configurations.

### [2. Team Success Metrics & KPIs](file:///Users/jjlee/Documents/GitHub/NHL-Analytics/databricks/llm/skills/2_team_kpis.md)
* **Measurement Level:** Team, Season.
* **Scope:** Aggregating box score metrics into high-level indicators of possession and season-long performance.
* **Key Components:**
  * **Corsi & Fenwick**: Possession indicators based on shot attempts.
  * **Pythagorean Expectation (PE)**: Expected win percentages based on Goal Differential.
  * **Ratings Percentage Index (RPI)**: Strength of Schedule (SoS) adjusted team ranking.
  * **Pairwise Wins**: Head-to-head dominance comparisons over common opponents.
  * **Win Streaks & Hot Streaks**: Streak classification by opponent RPI difficulty.

### [3. Sports Betting & ELO Calibration](file:///Users/jjlee/Documents/GitHub/NHL-Analytics/databricks/llm/skills/3_sports_betting_elo.md)
* **Measurement Level:** Game, Sports Betting.
* **Scope:** Processing moneyline odds, removing bookmaker margin (vig), and validating predictive ELO ratings against betting markets.
* **Key Components:**
  * **American-to-Probability Conversions**: Normalizing American moneylines to implied probabilities.
  * **Shin's Method**: Mathematical Vig removal algorithm via bisection numerical methods.
  * **ELO Calibration**: Logistic regression and PyMC Bayesian calibration of ratings.

### [4. NCAA-NHL Player Prospects](file:///Users/jjlee/Documents/GitHub/NHL-Analytics/databricks/llm/skills/4_player_college_prospects.md)
* **Measurement Level:** Player, College, Season.
* **Scope:** Linking NCAA college career records with NHL player metrics to study career trajectories and draft production.
* **Key Components:**
  * **Fuzzy String Matching**: Resolving names across sources using Levenshtein distance thresholds.
  * **Tenure Analytics**: Testing whether years spent in college correlate with rookie debut vs. long-term NHL performance.
  * **Draft Pick Decay**: Analyzing goal production changes as draft picks mature.

### [5. Decision Science & Latent Momentum SSM](file:///Users/jjlee/Documents/GitHub/NHL-Analytics/databricks/llm/skills/5_decision_science_momentum.md)
* **Measurement Level:** Play-by-Play, Game.
* **Scope:** In-game sequential event modeling, quantifying event-specific probabilities, and estimating unobserved team momentum states.
* **Key Components:**
  * **Faceoff Goal Probability**: Conditional chance of scoring within possession sequences following faceoff wins in offensive/defensive postures.
  * **Blocked Shots Impact**: Comparing unblocked missed shots/shots-on-goal with blocked shots via KS-tests and Maxwell-Boltzmann distributions.
  * **Continuous-Time Kalman State-Space Model**: Learning the impulse magnitude ($B$) and half-life decay ($a_0$) of hits, shots, and takeaways, controlled for team strength priors (pre-game ELO and Pythagorean Expectation).

---

## Agent Usage Instructions
For any LLM agent running in an IDE (e.g. Copilot, cursor) or Databricks environment:
1. **Context Loading:** Load the relevant skillset markdown before generating scripts for data processing or modeling.
2. **Path Resolution:** Always resolve file locations dynamically using `src/data/apinhle/config.py` functions:
   * `get_box_dir()`, `get_team_dir()`, `get_play_dir()`, and `get_betting_dir()`.
3. **Focal Perspective:** For team-level and game-level metrics, be explicit about the focal team (e.g., `win_for` vs. `win_against`).
