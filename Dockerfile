FROM python:3.12-slim

WORKDIR /app

# Install system libraries needed by ML dependencies (e.g. LightGBM / XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install the reviewed lock first for deterministic, cacheable builds.
COPY requirements.txt requirements.lock.txt ./
RUN python -m pip install --no-cache-dir pip==24.2 \
    && python -m pip install --no-cache-dir --no-deps -r requirements.lock.txt \
    && python -m pip check

# Copy application source and trained models
COPY . .

# Create non-root user for security
RUN useradd -u 1000 -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/docs')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
