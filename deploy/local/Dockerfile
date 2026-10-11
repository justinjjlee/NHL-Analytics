# Use official Python 3.13 slim image as the base
FROM python:3.13-slim

# Set environment variables to optimize Python & Streamlit execution
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=1901 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Set the working directory inside the container
WORKDIR /workspace

# Install system dependencies needed for native C extensions & healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to maximize Docker layer caching
COPY src/requirements/requirements.txt /workspace/requirements.txt

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and necessary analytical datasets
COPY app/ /workspace/app/
COPY latest/box/ /workspace/latest/box/
COPY latest/team/ /workspace/latest/team/
COPY latest/player/ /workspace/latest/player/
COPY dev/player/player_draft/ /workspace/dev/player/player_draft/
COPY dev/decision_science/impact_of_hits/data/ /workspace/dev/decision_science/impact_of_hits/data/
COPY dev/decision_science/plays_blockingShots/ /workspace/dev/decision_science/plays_blockingShots/
COPY dev/decision_science/hit_eda/ /workspace/dev/decision_science/hit_eda/

# Expose the local testing port
EXPOSE 1901

# Healthcheck to verify Streamlit server readiness
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:1901/_stcore/health || exit 1

# Launch the Streamlit application on port 1901
CMD ["streamlit", "run", "app/app.py", "--server.port=1901", "--server.address=0.0.0.0"]
