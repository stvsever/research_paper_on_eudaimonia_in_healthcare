FROM python:3.11-slim

LABEL maintainer="Stijn Van Severen"
LABEL description="Reproducible environment for eudaimonia-in-healthcare research pipeline"

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Default: run v2 pipeline
CMD ["python", "src/v2/run_pipeline.py"]
