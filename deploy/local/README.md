# Local Docker Deployment Guide (Port 1901)

This directory contains the Docker configuration to build, run, and test the **Rinklytics (NHL Analytics)** dashboard locally in an isolated container on port **1901**.

---

## 🚀 Quick Start

### Option A: Using Docker CLI

Run these commands from the **repository root directory**:

1. **Build the image:**
   ```bash
   docker build -t rinklytics:local -f deploy/local/Dockerfile .
   ```

2. **Run the container (binding port 1901):**
   ```bash
   docker run -d --name rinklytics-local -p 1901:1901 rinklytics:local
   ```

3. **Verify in browser:**
   Open your browser and navigate to:
   👉 **[http://localhost:1901](http://localhost:1901)**

---

### Option B: Using Docker Compose

From the repository root directory:
```bash
docker compose -f deploy/local/docker-compose.yml up -d --build
```

To stop:
```bash
docker compose -f deploy/local/docker-compose.yml down
```

---

## 🛠 Container Management & Debugging

* **View live logs:**
  ```bash
  docker logs -f rinklytics-local
  ```

* **Check container health:**
  ```bash
  docker ps -f name=rinklytics-local
  ```

* **Stop the container:**
  ```bash
  docker stop rinklytics-local
  ```

* **Remove the container:**
  ```bash
  docker rm rinklytics-local
  ```

* **Rebuild after making code changes:**
  ```bash
  docker build -t rinklytics:local -f deploy/local/Dockerfile .
  docker run -d --name rinklytics-local -p 1901:1901 rinklytics:local
  ```

---

## ⚙️ Architecture & Configuration Notes

* **Port Mapping (`1901:1901`):** The container configures Streamlit to listen natively on port `1901` (`--server.port=1901`) with external binding (`--server.address=0.0.0.0`).
* **Optimized Layer Caching:** System dependencies and `requirements.txt` are cached in early Docker layers. Code modifications will rebuild in seconds.
* **Excluded Data Context:** Large raw play-by-play files (`latest/play/` ~700MB) and development caches are excluded via `.dockerignore`, ensuring the build context transfers quickly.
