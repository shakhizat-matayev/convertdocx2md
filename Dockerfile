FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Minimal OS packages (curl often handy for debugging/health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r /app/requirements.txt

COPY . /app

# Non-root user
RUN useradd -m appuser \
 && mkdir -p /data \
 && chown -R appuser:appuser /app /data

USER appuser

EXPOSE 8000

# Tunables for ACA without rebuild:
# - WEB_CONCURRENCY: number of workers
# - GUNICORN_TIMEOUT: request timeout (DRIFT needs higher)
ENV WEB_CONCURRENCY=2 \
    GUNICORN_TIMEOUT=900

CMD ["bash", "-lc", "gunicorn -k uvicorn.workers.UvicornWorker app.main:app \
  --bind 0.0.0.0:${PORT:-8000} \
  --workers ${WEB_CONCURRENCY} \
  --timeout ${GUNICORN_TIMEOUT} \
  --graceful-timeout 30 \
  --keep-alive 120 \
  --access-logfile - \
  --error-logfile -"]