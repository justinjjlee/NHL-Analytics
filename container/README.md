# 🐳 Rinklytics Docker Container Deployment Guide

This directory contains the Docker assets and documentation required to build, test, and deploy the **Rinklytics** (NHL Analytics) dashboard as a containerized application.

---

## 📋 Directory Contents

- **`Dockerfile`**: Defines the multi-layer build instructions for the container using a lightweight Python 3.11 base.
- **`.dockerignore`**: Excludes unnecessary development files (`.venv`, notebooks, Git history, local caches) to optimize container build context size and security.
- **`requirements.txt`**: A curated list of required Python packages optimized for the container runtime.

---

## 🚀 Step 1: Local Development & Verification

Before deploying to production, verify that the container builds and runs correctly on your local machine.

### 1. Build the Docker Image
Run the following command from the **repository root directory** (not from inside the `container` folder):
```bash
docker build -t rinklytics:latest -f container/Dockerfile .
```

### 2. Run the Container Locally
Run the container and map the default Streamlit port (`8501`) to your host machine:
```bash
docker run -p 8501:8501 rinklytics:latest
```

### 3. Verify in Browser
Open your browser and navigate to:
- **Local URL**: [http://localhost:8501](http://localhost:8501)

---

## 📤 Step 2: Push to a Container Registry

To deploy the container to the cloud, you must first push it to a container registry. Below are guides for **Docker Hub**, **Amazon Elastic Container Registry (ECR)**, and **Google Artifact Registry (GAR)**.

### Option A: Docker Hub (Easiest)
1. **Log in to Docker Hub**:
   ```bash
   docker login
   ```
2. **Tag your image**:
   ```bash
   docker tag rinklytics:latest your_dockerhub_username/rinklytics:latest
   ```
3. **Push the image**:
   ```bash
   docker push your_dockerhub_username/rinklytics:latest
   ```

### Option B: AWS Elastic Container Registry (ECR)
1. **Create an ECR Repository**:
   ```bash
   aws ecr create-repository --repository-name rinklytics --region us-east-1
   ```
2. **Authenticate Docker to ECR**:
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your_aws_account_id.dkr.ecr.us-east-1.amazonaws.com
   ```
3. **Tag and Push**:
   ```bash
   docker tag rinklytics:latest your_aws_account_id.dkr.ecr.us-east-1.amazonaws.com/rinklytics:latest
   docker push your_aws_account_id.dkr.ecr.us-east-1.amazonaws.com/rinklytics:latest
   ```

---

## 🌐 Step 3: Cloud Deployment Options

Here are step-by-step guides for hosting your containerized Streamlit app.

### Option 1: Google Cloud Run (Recommended - Serverless & Fast)
Google Cloud Run is ideal for Streamlit because it scales to zero when not in use, saving costs.

1. **Deploy directly from the source** (using GCP's Cloud Build under the hood):
   ```bash
   gcloud run deploy rinklytics \
     --source . \
     --port 8501 \
     --platform managed \
     --allow-unauthenticated \
     --region us-central1
   ```
2. **Retrieve URL**:
   Once completed, Google Cloud Run will output a public URL, for example:
   `https://rinklytics-xxxxxx-uc.a.run.app`

---

### Option 2: AWS App Runner (Recommended for AWS Users)
AWS App Runner provides a fully managed service that takes a container image directly from ECR and hosts it with automatic scaling and SSL.

1. **Create an App Runner Service via AWS CLI**:
   Create a file `app-runner-config.json`:
   ```json
   {
     "ServiceName": "rinklytics",
     "SourceConfiguration": {
       "ImageRepository": {
         "ImageIdentifier": "your_aws_account_id.dkr.ecr.us-east-1.amazonaws.com/rinklytics:latest",
         "ImageConfiguration": {
           "Port": "8501"
         },
         "ImageRepositoryType": "ECR"
       },
       "AutoDeploymentsEnabled": false
     }
   }
   ```
2. **Create the service**:
   ```bash
   aws apprunner create-service --cli-input-json file://app-runner-config.json
   ```
3. **Retrieve URL**:
   App Runner will allocate a secure URL:
   `https://xxxxxx.us-east-1.awsapprunner.com`

---

### Option 3: Render (Easiest UI-based hosting)
Render is an excellent alternative PaaS that supports building directly from your GitHub repo using a Dockerfile.

1. Create a account at [render.com](https://render.com).
2. Click **New +** and select **Web Service**.
3. Connect your GitHub repository.
4. Set the following configurations:
   - **Runtime**: `Docker`
   - **Dockerfile Path**: `container/Dockerfile`
5. Click **Deploy**. Render will build and deploy the container automatically, assigning you a sub-domain:
   `https://rinklytics.onrender.com`

---

## ⚙️ Custom Configurations (Environment Variables)

Streamlit behavior can be configured inside the Docker container by using environment variables or passing flags to the command line:

- **Configure Theme**:
  You can mount or write a `.streamlit/config.toml` inside the workspace or override colors via Streamlit configuration flags.
- **Adjust Memory/CPU**:
  Since DuckDB and Pandas read local datasets (e.g. box scores, play-by-play, cohort files), ensure the server has **at least 1GB or 2GB of RAM** configured (GCP Cloud Run defaults to 512MB or 1GB, which can be increased via the console or CLI flag `--memory 2Gi`).
