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
from fastapi.responses import HTMLResponse
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


_DEMO_PAGE = """<!doctype html><meta charset=utf-8><title>Risk Scoring Demo</title>
<style>body{font:15px system-ui;max-width:760px;margin:2rem auto;padding:0 1rem}
pre{background:#f4f4f4;padding:1rem;overflow:auto}button{margin-right:.5rem;padding:.5rem 1rem}</style>
<h1>Credit-default scoring</h1>
<p>Scores a sample applicant. Declined applicants get SHAP-based adverse-action reason codes.
Full API: <a href=/docs>/docs</a>. Model card: <a href=/model-card>/model-card</a>.</p>
<button onclick="run(HIGH)">Score high-risk applicant</button>
<button onclick="run(LOW)">Score low-risk applicant</button><pre id=out>(click a button)</pre>
<script>
const HIGH={limit_bal:20000,sex:2,education:2,marriage:1,age:35,pay_1:2,pay_2:2,pay_3:2,pay_4:0,pay_5:0,pay_6:0,
bill_amt1:18000,bill_amt2:17500,bill_amt3:17000,bill_amt4:16000,bill_amt5:15000,bill_amt6:14000,
pay_amt1:0,pay_amt2:1000,pay_amt3:0,pay_amt4:500,pay_amt5:500,pay_amt6:500};
const LOW={limit_bal:300000,sex:1,education:1,marriage:2,age:42,pay_1:-1,pay_2:-1,pay_3:-1,pay_4:-1,pay_5:-1,pay_6:-1,
bill_amt1:5000,bill_amt2:4000,bill_amt3:6000,bill_amt4:3000,bill_amt5:5000,bill_amt6:4500,
pay_amt1:5000,pay_amt2:4000,pay_amt3:6000,pay_amt4:3000,pay_amt5:5000,pay_amt6:4500};
async function run(b){const r=await fetch('/predict',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(b)});
document.getElementById('out').textContent=JSON.stringify(await r.json(),null,2)}
</script>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def demo_page() -> str:
    return _DEMO_PAGE


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
