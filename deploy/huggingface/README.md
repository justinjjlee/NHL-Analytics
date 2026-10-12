# Hugging Face Ecosystem: Spaces & Public Datasets

This directory orchestrates the automated synchronization, deployment, and data modeling for both the **Hugging Face Spaces web application** and the **Hugging Face Public Dataset Lakehouse**.

---

## 🌐 Public Endpoints & Use Cases

| Resource | URL | Description |
| :--- | :--- | :--- |
| **🏒 Rinklytics Web Dashboard** | [https://huggingface.co/spaces/jjerlee/rinklytics](https://huggingface.co/spaces/jjerlee/rinklytics) | Live interactive Streamlit dashboard featuring daily scoreboards, player PPI rankings, physical hit impact heatmaps, and team ratings. |
| **📦 NHL Analytics Public Dataset** | [https://huggingface.co/datasets/jjerlee/nhl-analytics](https://huggingface.co/datasets/jjerlee/nhl-analytics) | Cleaned, longitudinal multi-season Parquet panel datasets (games, box scores, odds, play-by-play, hits, player metrics) spanning 2013 to present. |
| **🐙 GitHub Source Code** | [https://github.com/justinjjlee/NHL-Analytics](https://github.com/justinjjlee/NHL-Analytics) | Primary source repository containing data extraction pipelines, decision science notebooks, and automation workflows. |
| **✍️ Medium Research Blog** | [https://medium.com/@thinkingjustin](https://medium.com/@thinkingjustin) | Deep-dive analytical essays and decision science studies (e.g. valor of blocking shots, physical momentum modeling). |

---

## 📁 Directory Architecture

```
deploy/huggingface/
├── README.md                  # Comprehensive guide for HF Spaces and Datasets
├── spaces_rinklytics.py       # Deployment script for Streamlit app to HF Spaces
├── sync_dataset.py            # Daily incremental sync engine with deduplication & diff checks
├── bootstrap_dataset.py       # Multi-season historical consolidator & dataset card builder
├── bootstrap_dataset.sh       # Convenience shell script for dataset bootstrap
└── huggingface_dataset.md     # Detailed dataset schema design & technical roadmap
```

---

## 1. 🏒 Hugging Face Spaces Deployment (`jjerlee/rinklytics`)

The interactive Streamlit dashboard is hosted and mirrored on Hugging Face Spaces as a Gradio/Streamlit container:
👉 **`https://huggingface.co/spaces/jjerlee/rinklytics`**

### Deployment Mechanism
* **Deployment Script:** [`spaces_rinklytics.py`](spaces_rinklytics.py)
* **Space Configuration:**
  - `sdk: gradio` (supports standard Streamlit execution via `app.py`)
  - `python_version: "3.12"`
  - `app_file: app.py`
* **Asset Upload Optimization:**
  To maintain a lightweight container footprint, non-dashboard assets and large raw data directories are excluded (`latest/play/**`, `src/**`, `databricks/**`, `dev/**/*.png`, `.github/**`). The dashboard streams ongoing data directly from the public Hugging Face dataset via HTTP/S.

### Automated CI/CD
Deployments to the Space are automated via GitHub Actions in [`.github/workflows/sync_hf_space.yml`](../../.github/workflows/sync_hf_space.yml):
* **Triggers:**
  - On push to `main` branch.
  - Automatically triggered upon successful completion of the daily game stats pull (`workflow_run`).
  - Scheduled daily at 10:00 UTC as a safety fallback.
* **Pipeline Step:**
  ```yaml
  - name: Deploy to Hugging Face Space
    if: ${{ secrets.HF_TOKEN != '' }}
    env:
      HF_TOKEN: ${{ secrets.HF_TOKEN }}
    run: |
      python deploy/huggingface/spaces_rinklytics.py --repo "jjerlee/rinklytics" --no-wait
  ```

### Manual Local Deployment
```bash
# Uses HF_TOKEN from environment or local .env file
uv run --with huggingface_hub python deploy/huggingface/spaces_rinklytics.py

# Or explicitly pass token and target repo:
HF_TOKEN="<YOUR_TOKEN>" uv run --with huggingface_hub python deploy/huggingface/spaces_rinklytics.py --repo "jjerlee/rinklytics"
```

---

## 2. 📦 Hugging Face Public Dataset (`jjerlee/nhl-analytics`)

The repository synchronizes cleaned longitudinal panel datasets to Hugging Face Datasets:
👉 **`https://huggingface.co/datasets/jjerlee/nhl-analytics`**

### Dataset Structure
The dataset organizes NHL data by domain entity and analytical type in high-performance **Apache Parquet** format:

#### Tier 1: Raw Longitudinal Panels (`data/`)
* **`data/games.parquet`**: Master schedule and game outcome panel (17,000+ games). Primary key: `game_id`.
* **`data/team_box.parquet`**: Team box scores, special teams, and shot differentials (34,000+ rows). Primary key: `(game_id, team)`.
* **`data/player_box.parquet`**: Player-level game logs with enriched own team, opponent, and game indicators (630,000+ rows). Primary key: `(game_id, player_id)`.
* **`data/odds.parquet`**: Closing betting odds tracking both US and Canadian markets (16,000+ rows). Primary key: `(game_id, odds_description, country)`.
* **`data/playbyplay.parquet`**: Granular play-by-play microdata events (5.4M+ events). Primary key: `(game_id, event_id)`.
* **`data/playbyplay_player.parquet`**: Official game sheet roster participants (680,000+ rows). Primary key: `(game_id, player_id)`.
* **`data/shifts.parquet`**: Player shift-level tracking. Primary key: `(game_id, player_id, shift_number, start_time)`.

#### Tier 2: Measured & Computed Statistics (`measurement/`)
All measurement datasets include standardized **`yr_season`** (e.g. 2013 to 2026) for longitudinal joining:
* **`measurement/box/team_season.parquet`**: Multi-season team KPI aggregates, Fenwick/Corsi differentials, Pythagorean expectation, and RPI.
* **`measurement/box/game_stats.parquet`**: Rolling momentum and advanced game stats.
* **`measurement/player/player_season.parquet`**: Skater and goaltender season aggregates (18,900+ rows).
* **`measurement/action/hits/*.parquet`**: 9 tables covering sequence outcomes, 30s net shot impact, spatial density grids, and next-event transition probabilities.

### Incremental Synchronization Engine (`sync_dataset.py`)
Incoming data updates run through [`deploy/huggingface/sync_dataset.py`](sync_dataset.py), ensuring safe, idempotent synchronization:
1. **Active Season Partition Replacement:** Isolates and replaces only the current active season (`game_id[:4] == active_year` or `yr_season == active_year`) while strictly protecting historical vintages (2013–2024).
2. **Deterministic Primary Key Deduplication:** Guarantees idempotency and prevents row explosion across all tables.
3. **Change Detection (Zero Noise Commits):** Compares candidate updates with the current remote snapshot. If incoming data introduces no changes, commits are skipped to keep git history clean.
4. **Country-Aware Odds Harmonization:** Tracks US and CAN betting feeds distinctly with the `country` column (`"US"` or `"CAN"`).
5. **Atomic Upload:** Stages all modified tables locally and executes a single atomic commit to Hugging Face via `HfApi.upload_folder`.

### Automated Daily GitHub Actions
Dataset synchronization is integrated into repository ingestion workflows:
* **Game Stats & Plays (Daily 09:00 UTC):** [`.github/workflows/proc_pull_box.yml`](../../.github/workflows/proc_pull_box.yml)
  ```bash
  python deploy/huggingface/sync_dataset.py --repo "jjerlee/nhl-analytics" --target box
  ```
* **Betting Odds (Daily 20:00 UTC):** [`.github/workflows/proc_pull_odds.yml`](../../.github/workflows/proc_pull_odds.yml)
  ```bash
  python deploy/huggingface/sync_dataset.py --repo "jjerlee/nhl-analytics" --target odds
  ```
* **Player Season Metrics (Mondays 08:00 UTC):** [`.github/workflows/proc_pull_player.yml`](../../.github/workflows/proc_pull_player.yml)
  ```bash
  python deploy/huggingface/sync_dataset.py --repo "jjerlee/nhl-analytics" --target player
  ```
* **On-Demand Manual Synchronization:** [`.github/workflows/sync_hf_dataset.yml`](../../.github/workflows/sync_hf_dataset.yml) allows manual dispatch with options for target domain (`all`, `box`, `odds`, `player`), forced sync, or custom season year.

### Manual Synchronization Commands
```bash
# Sync box scores, game stats, and play-by-play events
uv run python deploy/huggingface/sync_dataset.py --target box

# Sync closing betting odds (US and CAN markets)
uv run python deploy/huggingface/sync_dataset.py --target odds

# Sync player season metrics
uv run python deploy/huggingface/sync_dataset.py --target player

# Full synchronization across all domains
uv run python deploy/huggingface/sync_dataset.py --target all
```

---

## 3. 🔌 App Data Integration Architecture

The Rinklytics dashboard accesses data via [`app/exe/data_loader.py`](../../app/exe/data_loader.py):

* **Ongoing Data**: Streams Parquet tables directly from Hugging Face URLs (`https://huggingface.co/datasets/jjerlee/nhl-analytics/resolve/main/...`) with automatic caching (`@st.cache_data`) and local file fallback.
* **One-Off Data**: Static research datasets (Arena Geography in `map_utils.py`, Amateur Draft Lifecycle in `draft_analysis.py`, and Shot Blocking Valor in `blocking_shots.py`) are loaded from local files or GitHub repository raw URLs (`https://raw.githubusercontent.com/justinjjlee/NHL-Analytics/main/...`).

### Direct Querying Examples

#### A. Zero-Install DuckDB Streaming SQL
```sql
SELECT 
    season, 
    home_team, 
    COUNT(*) AS home_games,
    ROUND(AVG(home_score), 2) AS avg_home_goals
FROM 'hf://datasets/jjerlee/nhl-analytics/data/games.parquet'
WHERE season >= 20232024
GROUP BY season, home_team
ORDER BY season DESC, avg_home_goals DESC;
```

#### B. Python Streaming (`polars` or `pandas`)
```python
import pandas as pd

# Stream games directly from Hugging Face
df_games = pd.read_parquet("https://huggingface.co/datasets/jjerlee/nhl-analytics/resolve/main/data/games.parquet")
print(f"Total Games: {len(df_games):,}")

# Stream betting lines
df_odds = pd.read_parquet("https://huggingface.co/datasets/jjerlee/nhl-analytics/resolve/main/data/odds.parquet")
print(df_odds[df_odds["country"] == "US"].tail(3))
```
