# =============================================================================
# Unified Cloud Dockerfile — StockTimePredictor
# Single image containing ALL model services + API endpoints.
#
# Build context: Backend/  (i.e. docker build -f cloud/Dockerfile .)
# =============================================================================
FROM python:3.11-slim

# ---- System deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Python deps (cached layer) ----
COPY cloud/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- Application code ----
COPY . /app

# ---- Ensure model directories exist ----
RUN mkdir -p /app/models/rf /app/models/lstm /app/models/xgb /app/models/prophet

# ---- Environment ----
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PORT=8080

EXPOSE 8080

# ---- Health-check (Cloud Run uses the HTTP probe on /) ----
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# ---- Start ----
# 2 Gunicorn workers + long timeout for training requests.
# Cloud Run injects $PORT automatically.
CMD exec gunicorn cloud.main:app \
    --bind 0.0.0.0:${PORT} \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers 2 \
    --timeout 900 \
    --graceful-timeout 120 \
    --keep-alive 5
