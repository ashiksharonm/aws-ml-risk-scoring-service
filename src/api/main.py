from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
from mangum import Mangum
import joblib
import pandas as pd
import numpy as np
import logging
import time
import uuid

from src.api.schemas import PredictionRequest, PredictionResponse, HealthCheck
from src.utils.config import settings
import xgboost as xgb

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# Global variables for model
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    try:
        logger.info("Loading model artifacts...")
        pipeline = joblib.load(settings.MODEL_PATH)
        models['pipeline'] = pipeline
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # We don't crash here to allow health check to pass, but predict will fail
    yield
    # Clean up resources if needed
    models.clear()

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION, lifespan=lifespan)

# Middleware for request ID and logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"RID={request_id} METHOD={request.method} PATH={request.url.path} STATUS={response.status_code} LATENCY={process_time:.4f}s")
    
    return response

@app.get("/health", response_model=HealthCheck)
def health_check():
    status = "healthy" if 'pipeline' in models else "degraded"
    return HealthCheck(status=status, version=settings.VERSION)

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if 'pipeline' not in models:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    pipeline = models['pipeline']
    
    try:
        # Convert request to DataFrame
        data_dict = request.model_dump()
        # Ensure column order matches training (though pandas handles by name usually, better safe)
        input_df = pd.DataFrame([data_dict])
        
        # Predict
        prob = pipeline.predict_proba(input_df)[:, 1][0]
        label = int(pipeline.predict(input_df)[0])
        
        # Per-feature contributions via XGBoost's native TreeSHAP (pred_contribs=True).
        # Values are in log-odds space; the last column is the bias term and is dropped.
        try:
            model = pipeline.named_steps['classifier']
            preprocessor = pipeline.named_steps['preprocessor']
            transformed_data = preprocessor.transform(input_df)
            feature_names = list(preprocessor.get_feature_names_out())
            contribs = model.get_booster().predict(
                xgb.DMatrix(transformed_data, feature_names=feature_names),
                pred_contribs=True,
            )[0][:-1]
            order = np.argsort(-np.abs(contribs))[:5]
            shap_dict = {feature_names[k]: float(contribs[k]) for k in order}
            top_feats = list(shap_dict.keys())
        except Exception as e:
            logger.warning(f"Feature explanation failed: {e}")
            shap_dict = {}
            top_feats = []

        return PredictionResponse(
            default_probability=float(prob),
            is_default=label,
            shap_values=shap_dict,
            top_features=top_feats
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Handler for AWS Lambda
handler = Mangum(app, lifespan="on")
