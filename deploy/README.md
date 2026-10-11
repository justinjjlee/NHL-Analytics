# Deployment & Container Environments

This directory organizes the deployment workflows and container runtime assets for the **NHL-Analytics (Rinklytics)** platform.

---

## Directory Architecture

```
deploy/
├── README.md           # Deployment overview & environments guide
├── huggingface.py      # Automated sync & deployment to Hugging Face Spaces
└── local/              # Local testing container environment (Port 1901)
    ├── Dockerfile      # Multi-layer build running Streamlit on port 1901
    ├── .dockerignore   # Excludes large raw data and dev caches
    ├── requirements.txt# Pinned container runtime dependencies
    ├── docker-compose.yml # Single-command container orchestration
    └── README.md       # Step-by-step local build and execution guide
```

---

## 1. Cloud Production Deployment: Hugging Face Spaces

The live interactive dashboard is hosted and mirrored on Hugging Face Spaces at:
👉 **[https://huggingface.co/spaces/jjerlee/rinklytics](https://huggingface.co/spaces/jjerlee/rinklytics)**

* **Script:** [`deploy/huggingface.py`](huggingface.py)
* **Automated CI/CD:** Triggered daily via [`.github/workflows/sync_hf_space.yml`](../.github/workflows/sync_hf_space.yml) upon completion of the game stats data pull.
* **Manual Local Execution:**
  ```bash
  # Reads HF_TOKEN from environment or .env
  uv run --with huggingface_hub python deploy/huggingface.py

  # Or specify custom repository:
  HF_TOKEN="<YOUR_TOKEN>" uv run --with huggingface_hub python deploy/huggingface.py --repo "jjerlee/rinklytics"
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
