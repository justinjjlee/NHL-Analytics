# Deployment & Container Environments

This directory organizes the deployment workflows and container runtime assets for the **NHL-Analytics (Rinklytics)** platform.

---

## Directory Architecture

```
deploy/
├── README.md                  # Deployment overview & environments guide
├── huggingface/               # Hugging Face cloud ecosystems (Spaces & Datasets)
│   ├── spaces_rinklytics.py   # Automated sync & deployment to Hugging Face Spaces
│   ├── sync_dataset.py        # Incremental dataset sync engine with deduplication
│   ├── bootstrap_dataset.py   # Multi-season panel bootstrapping & card generator
│   └── huggingface_dataset.md # Dataset architecture, schema, and API specification
└── local/                     # Local testing container environment (Port 1901)
    ├── Dockerfile             # Multi-layer build running Streamlit on port 1901
    ├── .dockerignore          # Excludes large raw data and dev caches
    ├── requirements.txt       # Pinned container runtime dependencies
    ├── docker-compose.yml     # Single-command container orchestration
    └── README.md              # Step-by-step local build and execution guide
```

---

## 1. Cloud Production Deployment: Hugging Face Spaces

The live interactive dashboard is hosted and mirrored on Hugging Face Spaces at:
👉 **[https://huggingface.co/spaces/jjerlee/rinklytics](https://huggingface.co/spaces/jjerlee/rinklytics)**

* **Deployment Script:** [`deploy/huggingface/spaces_rinklytics.py`](huggingface/spaces_rinklytics.py)
* **Automated CI/CD:** Triggered daily via [`.github/workflows/sync_hf_space.yml`](../.github/workflows/sync_hf_space.yml) upon completion of the game stats data pull.
* **Manual Local Execution:**
  ```bash
  # Reads HF_TOKEN from environment or .env
  uv run --with huggingface_hub python deploy/huggingface/spaces_rinklytics.py

  # Or specify custom repository:
  HF_TOKEN="<YOUR_TOKEN>" uv run --with huggingface_hub python deploy/huggingface/spaces_rinklytics.py --repo "jjerlee/rinklytics"
  ```

---

## 2. Public Data Lakehouse: Hugging Face Datasets

The repository synchronizes cleaned longitudinal panel datasets to Hugging Face Datasets at:
👉 **[https://huggingface.co/datasets/jjerlee/nhl-analytics](https://huggingface.co/datasets/jjerlee/nhl-analytics)**

* **Sync Engine:** [`deploy/huggingface/sync_dataset.py`](huggingface/sync_dataset.py)
* **Bootstrap Consolidator:** [`deploy/huggingface/bootstrap_dataset.py`](huggingface/bootstrap_dataset.py)
* **Detailed Architecture:** [`deploy/huggingface/huggingface_dataset.md`](huggingface/huggingface_dataset.md)
* **Automated CI/CD:**
  * **Daily Box & Plays (09:00 UTC):** [`.github/workflows/proc_pull_box.yml`](../.github/workflows/proc_pull_box.yml) (`--target box`)
  * **Daily Closing Betting Odds (20:00 UTC):** [`.github/workflows/proc_pull_odds.yml`](../.github/workflows/proc_pull_odds.yml) (`--target odds`)
  * **Weekly Player Aggregates (Monday 08:00 UTC):** [`.github/workflows/proc_pull_player.yml`](../.github/workflows/proc_pull_player.yml) (`--target player`)
  * **Manual Workflow Dispatch:** [`.github/workflows/sync_hf_dataset.yml`](../.github/workflows/sync_hf_dataset.yml)
* **Manual Local Execution:**
  ```bash
  # Sync game records, box scores, and play-by-play
  uv run python deploy/huggingface/sync_dataset.py --target box

  # Sync closing betting odds (US and CAN markets)
  uv run python deploy/huggingface/sync_dataset.py --target odds

  # Sync player season metrics
  uv run python deploy/huggingface/sync_dataset.py --target player

  # Full synchronization with deduplication and change detection
  uv run python deploy/huggingface/sync_dataset.py --target all
  ```

---

## 2. Local Container Testing: Docker Environment (Port 1901)

To test the application locally inside an isolated container identical to production:

* **Directory:** [`deploy/local/`](local/)
* **Port:** **`1901`** (`http://localhost:1901`)
* **Quick Run:**
  ```bash
  docker build -t rinklytics:local -f deploy/local/Dockerfile .
  docker run -d --name rinklytics-local -p 1901:1901 rinklytics:local
  ```
  Or via Docker Compose:
  ```bash
  docker compose -f deploy/local/docker-compose.yml up -d --build
  ```
