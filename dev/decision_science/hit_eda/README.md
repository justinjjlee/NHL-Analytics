# Physical Hits Exploratory Data Analysis (Hit EDA)

This subproject provides a grounded, one-dimensional Exploratory Data Analysis (EDA) framework and data pipeline for dissecting NHL body check ("hit") micro-telemetry from play-by-play data.

---

## Directory Architecture

```
dev/decision_science/hit_eda/
├── README.md               # Quickstart guide, environment triggers & CLI commands
├── EXPLORATORY.md          # Full conceptual, mathematical & visualization blueprint
├── config.py               # Dual-runtime environment configuration (Local vs. Databricks)
├── dataproc.py             # Production sequence processor (Pandas & PySpark engines)
├── databricks_job.py       # Databricks workflow & interactive notebook entrypoint
└── data/                   # Generated exploratory summary tables
    ├── hit_zone_summary.csv        # Macro KPIs by zone (D, N, O)
    ├── hit_next_events.csv         # Step +1 event category, ownership & latency matrix
    └── hit_sequences_sample.csv    # Sample micro-event sequence records
```

---

## 1. Running Locally

The local pipeline processes raw play-by-play files (`latest/play/{YYYY}_playbyplay.csv`) using a vectorized Pandas engine.

### Quick Start (Latest Season):
```bash
uv run python dev/decision_science/hit_eda/dataproc.py --env local --seasons 2026
```

### Multi-Season Ingestion:
```bash
uv run python dev/decision_science/hit_eda/dataproc.py --env local --seasons 2024,2025,2026
```

### Full Historical Archive:
```bash
uv run python dev/decision_science/hit_eda/dataproc.py --env local --seasons all
```

---

## 2. Running in Databricks (Unity Catalog)

The pipeline is pre-configured to detect Databricks cluster runtimes and integrate directly with **Unity Catalog**:

- **Input Volume:** `/Volumes/nhl-databricks/data/play/*_playbyplay.csv`
- **Output Volume:** `/Volumes/nhl-databricks/data/hit_eda/`
- **Managed Delta Tables:**
  - `` `nhl-databricks`.data.hit_eda_sequences ``
  - `` `nhl-databricks`.data.hit_eda_zone_kpi ``

### Option A: Trigger via Databricks Jobs / CLI
```bash
python dev/decision_science/hit_eda/dataproc.py --env databricks --engine spark --seasons all --save-delta
```

### Option B: Interactive Notebook Execution
Run [`databricks_job.py`](databricks_job.py) directly inside your Databricks workspace or attach it to a Databricks Workflow job. The notebook provides interactive widgets for setting target seasons and enabling/disabling Delta table creation.

---

## 3. Environment Triggers & CLI Options

| Argument | Options | Default | Description |
| :--- | :--- | :---: | :--- |
| `--env` | `auto`, `local`, `databricks` | `auto` | Execution environment trigger (auto-detects via `DATABRICKS_RUNTIME_VERSION` or `NHL_ENV`). |
| `--seasons` | `2026`, `2024,2025,2026`, `all` | `2026` | Comma-separated list of seasons to process. |
| `--engine` | `auto`, `pandas`, `spark` | `auto` | Engine selection (defaults to `pandas` locally and `spark` on Databricks). |
| `--save-delta` | Flag | `False` | When enabled on Databricks/Spark, writes managed Delta tables to Unity Catalog. |
| `--limit` | Integer | `None` | Restricts rows per file for rapid testing. |

---

## 4. Key Analytical Metrics Generated

1. **First-Event Ownership ($\Pi_{\text{recovery}}$):**
   Percentage of post-hit events where the hitting team immediately recovers the puck vs. the opponent retaining possession.
2. **Action Latency Horizon ($\Delta t$):**
   Elapsed seconds between physical contact and the next registered event.
3. **Shot-to-Goal Pipeline ($Hit \rightarrow SOG \rightarrow Goal$):**
   Shots on Goal and Goals generated per 100 hits for both teams.
4. **Defensive Zone Exit Rate ($ZER$):**
   Frequency with which defensive zone hits result in successful puck clearance into neutral or offensive ice ($D \rightarrow N/O$).
