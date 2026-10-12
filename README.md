<h1>
<a href="https://medium.com/@thinkingjustin">
<img src="app/assets/img_main.jpeg" width="80px" align="left" style="margin-right: 15px;" alt="nhla-logo"> 
</a> Rinklytics: Sports Data Analytics & Public Lakehouse
</h1>

[![justinjjlee - NHL-Analytics](https://img.shields.io/static/v1?label=justinjjlee&message=NHL-Analytics&color=blue&logo=github)](https://github.com/justinjjlee/NHL-Analytics "Go to GitHub repo")
[![stars - NHL-Analytics](https://img.shields.io/github/stars/justinjjlee/NHL-Analytics?style=social)](https://github.com/justinjjlee/NHL-Analytics)
[![forks - NHL-Analytics](https://img.shields.io/github/forks/justinjjlee/NHL-Analytics?style=social)](https://github.com/justinjjlee/NHL-Analytics)
[![Medium - NHL-Analytics](https://img.shields.io/badge/Medium-Read-green?logo=Medium)](https://medium.com/@thinkingjustin)
[![Hugging Face - Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/jjerlee/rinklytics)
[![Hugging Face - Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue)](https://huggingface.co/datasets/jjerlee/nhl-analytics)
[![Databricks](https://img.shields.io/badge/Databricks-Cloud-FF3621?logo=databricks&logoColor=white)](https://databricks.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](deploy/local/README.md)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org)

**NHL Analytics (Rinklytics)** is an end-to-end sports analytics platform and public open-data lakehouse. It collects, cleans, models, and visualizes National Hockey League (NHL) data, uncovering tactical behaviors, physical momentum, market odds efficiencies, and player career trajectories.

All deep-dive articles and analytical methodologies are published on **[Medium (@thinkingjustin)](https://medium.com/@thinkingjustin)**.

---

## 🏛️ System Scope & Architecture

The platform operates across four coordinated environments:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        NHL API & Data Sources                          │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ Daily Automated Pulls (09:00 & 20:00 UTC)
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   GitHub Actions Automation & CI/CD                    │
│   • Scrapes raw box scores, odds, player logs, and play-by-play events │
│   • Commits incremental updates to latest/ repository files            │
│   • Deduplicates & synchronizes typed Parquet panels to Hugging Face   │
└──────────────────┬──────────────────────────────────┬──────────────────┘
                   │                                  │
                   ▼                                  ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────┐
│  🤗 Hugging Face Public Datasets     │  │  ☁️ Databricks & Cloud Lake  │
│  (jjerlee/nhl-analytics)             │  │  • Large-scale out-of-core   │
│  • 14 seasons (2013–present)         │  │    data engineering (DuckDB) │
│  • Snappy-compressed Parquet tables  │  │  • Continuous-time SSM       │
│  • Games, Odds, Plays, Hits, Players │  │    Kalman Filter estimation  │
└──────────────────┬───────────────────┘  └──────────────────────────────┘
                   │
                   ▼ Fast Parquet Streaming
┌────────────────────────────────────────────────────────────────────────┐
│                   🤗 Rinklytics Web Application                        │
│   • Live Production Space: https://huggingface.co/spaces/jjerlee/rinklytics │
│   • Local Testing Container: Docker on port 1901 (http://localhost:1901)│
│   • Visualizes Game Scores, Betting Odds, Hit Momentum, & Shot Maps    │
└────────────────────────────────────────────────────────────────────────┘
```

### Core Components

| Domain | Platform / Endpoint | Description |
| :--- | :--- | :--- |
| **Interactive Dashboard** | [**Hugging Face Spaces (`jjerlee/rinklytics`)**](https://huggingface.co/spaces/jjerlee/rinklytics) | Live Streamlit application streaming longitudinal data directly from Hugging Face Parquet tables with zero cold-start database overhead. |
| **Public Data Lakehouse** | [**Hugging Face Datasets (`jjerlee/nhl-analytics`)**](https://huggingface.co/datasets/jjerlee/nhl-analytics) | Clean, typed multi-season panel datasets (2013–2026+) covering game box scores, US/CAN closing betting odds, player game appearances, play-by-play events, and physical hit sequences. |
| **Big Data Engineering** | **Databricks Workspaces** | Distributed compute and ML pipelines for processing multi-million play-by-play sequences, training PyTorch state-space models, and out-of-core analytics via DuckDB. |
| **Pipeline Automation** | **GitHub Actions Workflows** | Daily automated cron jobs that harvest NHL API endpoints, commit data to GitHub, and trigger atomic Parquet syncs to Hugging Face. |

---

## 🛠️ Technology Stack

| Layer | Tools & Technologies |
| :--- | :--- |
| **Languages** | Python 3.13, Julia |
| **Cloud & Deployment** | Databricks, Hugging Face Spaces & Datasets, GitHub Actions, Docker |
| **Data Engine & Storage** | DuckDB, Pandas, PyArrow, Parquet (Snappy), Unity Catalog |
| **Machine Learning & Modeling** | PyTorch (State-Space Models & Kalman Filtering), SciPy, Scikit-learn |
| **Visualization & App** | Streamlit, Plotly Express & Graph Objects, Folium, Matplotlib, Seaborn |

---

## 🚀 Running Locally

### 1. Dashboard Application

You can run the Rinklytics interactive dashboard either directly with `uv` or inside an isolated container:

#### Option A: Native Execution
```bash
# Install dependencies
uv pip install -r src/requirements/requirements.txt

# Run the Streamlit app
uv run streamlit run app/app.py
```
Access the application at 👉 **`http://localhost:8501`**

#### Option B: Isolated Docker Container (Port 1901)
To run in an isolated local container mirroring production:
```bash
# Build and run container
docker build -t rinklytics:local -f deploy/local/Dockerfile .
docker run -d --name rinklytics-local -p 1901:1901 rinklytics:local
```
Access the container at 👉 **`http://localhost:1901`**  
*(See [**`deploy/local/README.md`**](deploy/local/README.md) for full Docker and Docker Compose instructions).*

---

### 2. Data Pipelines & Hugging Face Dataset Sync

Data extraction and panel dataset generation can be triggered locally:

```bash
# Pull today's game box scores
uv run python src/data/apinhle/data_pull_box.py

# Pull today's closing betting odds
uv run python src/data/apinhle/data_pull_box_odds.py

# Sync updated box scores to the Hugging Face Dataset
uv run python deploy/huggingface/sync_dataset.py --target box

# Synchronize all domains with automatic deduplication
uv run python deploy/huggingface/sync_dataset.py --target all
```
*(See [**`deploy/huggingface/README.md`**](deploy/huggingface/README.md) for detailed sync engine architecture and parameters).*

---

## 📚 Documentation & Deep-Dive Guides

For in-depth specifications, architectural details, and research notebooks, refer to the dedicated guides:

| Guide / Resource | Description |
| :--- | :--- |
| [**`deploy/README.md`**](deploy/README.md) | Overview of deployment environments and container architecture. |
| [**`deploy/huggingface/README.md`**](deploy/huggingface/README.md) | Complete guide to Hugging Face Spaces & Datasets synchronization engines and CI/CD triggers. |
| [**`deploy/huggingface/huggingface_dataset.md`**](deploy/huggingface/huggingface_dataset.md) | Official Dataset Card, column dictionaries, schemas, and PyArrow/Polars loading examples. |
| [**`deploy/local/README.md`**](deploy/local/README.md) | Docker & Docker Compose setup for local container testing on port 1901. |
| [**`dev/decision_science/impact_of_hits/`**](dev/decision_science/impact_of_hits/) | Continuous-time State-Space Model and Kalman filter modeling the momentum effect of physical hits. |
| [**`dev/player/player_draft/`**](dev/player/player_draft/) | Amateur draft lifecycle analysis, survival modeling, and NCAA prospect progression. |
| [**`dev/team_success/`**](dev/team_success/) | Pythagorean expectation, Corsi/Fenwick metrics, and Elo rating betting efficiency. |

---

```
   ,
    -   \O                                     ,  .-.___
  -     /\                                   O/  /xx\XXX\
 -   __/\ `\                                 /\  |xx|XXX|
    `    \, \_ =                          _/` << |xx|XXX|
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
```
