# syntax=docker/dockerfile:1
# Stage 1: install dependencies and train the model artifacts (UCI data, ~5 MB download).
FROM python:3.14-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ src/
COPY scripts/ scripts/
ARG USE_SYNTHETIC_DATA=false
ENV PYTHONPATH=/app
RUN USE_SYNTHETIC_DATA=${USE_SYNTHETIC_DATA} python -m src.training.train \
 && USE_SYNTHETIC_DATA=${USE_SYNTHETIC_DATA} python -m scripts.make_demo_batches --out /app/demo

# Stage 2: slim runtime, non-root (uid 1000 as required by Hugging Face Spaces).
FROM python:3.14-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app PORT=7860 PATH=/opt/venv/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd -m -u 1000 appuser
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=appuser /app/models models/
COPY --from=builder --chown=appuser /app/reports reports/
COPY --from=builder --chown=appuser /app/demo demo/
COPY --chown=appuser src/ src/
COPY --chown=appuser examples/ examples/
RUN mkdir -p logs && chown appuser logs
USER appuser
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('PORT','7860'), timeout=3)"
CMD ["sh", "-c", "exec uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"]
