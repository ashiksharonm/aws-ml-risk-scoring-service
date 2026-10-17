from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from mangum import Mangum

from src.api.schemas import (
    DriftReport,
    DriftRequest,
    HealthCheck,
    PredictionRequest,
    PredictionResponse,
)
from src.decisioning.reason_codes import adverse_action_reasons
from src.decisioning.shadow import ShadowScorer
from src.explainability.contributions import compute_contributions, top_contributions
from src.monitoring.drift import drift_report, load_reference
from src.monitoring.live_buffer import LiveBuffer
from src.utils.config import settings
from src.utils.logging import configure_logging, request_id_var

configure_logging(settings.LOG_LEVEL)
logger = logging.getLogger("api")

# Loaded artifacts: 'pipeline' (champion), 'shadow', 'reference', 'model_card'.
models: Dict[str, Any] = {}
live_buffer = LiveBuffer(settings.LIVE_BUFFER_SIZE)


def _load_optional(name: str, path: str, loader: Any) -> None:
    if Path(path).exists():
        try:
            models[name] = loader(path)
        except Exception:
            logger.exception("failed to load %s from %s", name, path)
    else:
        logger.info("optional artifact %s not found at %s", name, path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        models["pipeline"] = joblib.load(settings.MODEL_PATH)
        logger.info("champion model loaded")
    except Exception:
        # Stay up so /healthz passes; /readyz and /predict report the problem.
        logger.exception("failed to load model from %s", settings.MODEL_PATH)
    _load_optional("reference", settings.REFERENCE_PATH, load_reference)
    _load_optional("model_card", settings.MODEL_CARD_PATH, lambda p: json.loads(Path(p).read_text()))
    _load_optional(
        "shadow",
        settings.CHALLENGER_PATH,
        lambda p: ShadowScorer(joblib.load(p), settings.SHADOW_LOG_PATH, settings.DECISION_THRESHOLD),
    )
    yield
    models.clear()


app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION, lifespan=lifespan)


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex
    token = request_id_var.set(rid)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        latency = time.perf_counter() - start
        logger.info(
            "request",
            extra={
                "fields": {
                    "method": request.method,
                    "path": request.url.path,
                    "latency_ms": round(latency * 1000, 2),
                }
            },
        )
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = rid
    return response


@app.get("/healthz", response_model=HealthCheck)
@app.get("/health", response_model=HealthCheck, include_in_schema=False)
def healthz() -> HealthCheck:
    """Liveness: the process is up. Reports 'degraded' when the model is not loaded."""
    status = "healthy" if "pipeline" in models else "degraded"
    return HealthCheck(status=status, version=settings.VERSION)


@app.get("/readyz")
def readyz() -> dict:
    if "pipeline" not in models:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready", "challenger": "shadow" in models, "reference": "reference" in models}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, background: BackgroundTasks) -> PredictionResponse:
    if "pipeline" not in models:
        raise HTTPException(status_code=503, detail="Model not loaded")
    pipeline = models["pipeline"]
    rid = request_id_var.get()
    row = request.model_dump()
    frame = pd.DataFrame([row])

    try:
        prob = float(pipeline.predict_proba(frame)[:, 1][0])
    except Exception as exc:
        logger.exception("prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    threshold = settings.DECISION_THRESHOLD
    declined = prob >= threshold
    shap_dict: Dict[str, float] = {}
    reasons: list = []
    try:
        names, contribs = compute_contributions(pipeline, frame)
        shap_dict = top_contributions(names, contribs[0], 5)
        if declined:
            reasons = adverse_action_reasons(
                names, contribs[0], top_n=settings.REASON_CODE_COUNT, applicant=row
            )
    except Exception:
        logger.exception("feature explanation failed")

    live_buffer.add(row, prob)
    shadow = models.get("shadow")
    if shadow is not None:
        # Shadow scoring runs after the response is built; it cannot alter the decision.
        background.add_task(shadow.score, frame, prob, rid, threshold)

    return PredictionResponse(
        default_probability=prob,
        is_default=int(declined),
        decision="decline" if declined else "approve",
        threshold=threshold,
        adverse_action_reasons=reasons,
        shap_values=shap_dict,
        top_features=list(shap_dict),
        request_id=rid,
    )


def _report(batch: pd.DataFrame, scores: Any) -> dict:
    if "reference" not in models:
        raise HTTPException(status_code=503, detail="Reference distribution not loaded")
    return drift_report(batch, models["reference"], scores=scores, min_rows=settings.MIN_DRIFT_ROWS)


@app.post("/monitoring/drift", response_model=DriftReport)
def monitoring_drift(body: DriftRequest) -> dict:
    """PSI feature drift and score drift of a posted batch against the training reference."""
    if "pipeline" not in models:
        raise HTTPException(status_code=503, detail="Model not loaded")
    batch = pd.DataFrame([r.model_dump() for r in body.rows])
    scores = models["pipeline"].predict_proba(batch)[:, 1]
    return _report(batch, scores)


@app.get("/monitoring/live", response_model=DriftReport)
def monitoring_live() -> dict:
    """Drift of the most recent /predict requests (in-memory rolling window)."""
    if len(live_buffer) == 0:
        raise HTTPException(status_code=404, detail="No requests scored yet")
    batch, scores = live_buffer.snapshot()
    return _report(batch, scores)


@app.get("/monitoring/shadow")
def monitoring_shadow() -> dict:
    shadow = models.get("shadow")
    if shadow is None:
        raise HTTPException(status_code=404, detail="No challenger model loaded")
    return shadow.summary()


@app.get("/model-card")
def model_card() -> dict:
    if "model_card" not in models:
        raise HTTPException(status_code=404, detail="Model card not generated; run `make train`")
    return models["model_card"]


# Handler for AWS Lambda (optional deploy path)
handler = Mangum(app, lifespan="on")
