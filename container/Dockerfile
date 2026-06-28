# Use official Python 3.11 slim image as the base
FROM python:3.11-slim

# Set environment variables to optimize Python execution in container
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501

# Set the working directory inside the container
WORKDIR /workspace

# Install system dependencies needed for installing packages and running the app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to take advantage of Docker layer caching
COPY requirements.txt .

# Install the Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the necessary files and directories to keep the image size small
COPY app/ app/
COPY latest/ latest/
COPY dev/decision_science/plays_blockingShots/result/ dev/decision_science/plays_blockingShots/result/
COPY dev/player/player_draft/data/ dev/player/player_draft/data/

# Expose Streamlit's default port
EXPOSE 8501

# Run the Streamlit application on start
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
