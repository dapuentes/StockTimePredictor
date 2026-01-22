"""
API Gateway - Cloud Run Version
=============================================================================
Simplified API Gateway for Google Cloud Run deployment.
Main differences from local version:
- Training is synchronous (no Celery job tracking)
- Optimized for serverless environment
- Extended timeouts for ML operations
=============================================================================
"""
import os
import httpx
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query, Path, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="StockTime Predictor API Gateway",
    version="2.0.0-cloud",
    description="Cloud Run optimized API Gateway for stock prediction services"
)

# CORS Configuration - Update with your Vercel/Netlify domain in production
origins = [
    "http://localhost:3000",
    "https://*.vercel.app",
    "https://*.netlify.app",
    # Add your production frontend URL here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, use specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Microservices URLs - These will be Cloud Run service URLs in production
microservices = {
    "rf": os.getenv("RF_SERVICE_URL", "http://localhost:8001"),
    "lstm": os.getenv("LSTM_SERVICE_URL", "http://localhost:8002"),
    "xgboost": os.getenv("XGB_SERVICE_URL", "http://localhost:8003"),
    "prophet": os.getenv("PROPHET_SERVICE_URL", "http://localhost:8004"),
    "shap": os.getenv("SHAP_SERVICE_URL", "http://localhost:8005"),
    "ensemble": os.getenv("ENSEMBLE_SERVICE_URL", "http://localhost:8006"),
}

# Extended timeout for training operations (Cloud Run supports up to 60 min)
TRAINING_TIMEOUT = float(os.getenv("TRAINING_TIMEOUT", "600"))  # 10 minutes default
PREDICTION_TIMEOUT = float(os.getenv("PREDICTION_TIMEOUT", "120"))  # 2 minutes default


class TrainRequest(BaseModel):
    """Training request model."""
    ticket: str = "NU"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    training_period: Optional[str] = None
    n_lags: Optional[int] = 10
    target_col: Optional[str] = "Close"
    train_size: Optional[float] = 0.8


class ExplainRequest(BaseModel):
    """SHAP explanation request."""
    ticker: str = "NU"
    model_type: str = "xgboost"
    top_features: int = 10


class EnsemblePredictRequest(BaseModel):
    """Ensemble prediction request."""
    ticker: str = "NU"
    forecast_horizon: int = 10
    target_col: str = "Close"
    models: Optional[List[str]] = None
    ensemble_method: str = "weighted_average"


# =============================================================================
# Root & Health Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "service": "StockTime Predictor API Gateway",
        "version": "2.0.0-cloud",
        "environment": "cloud",
        "available_models": list(microservices.keys()),
        "endpoints": {
            "training": "/train/{model_type}",
            "prediction": "/predict/{model_type}",
            "models": "/models/{model_type}",
            "shap": "/explain",
            "ensemble": "/ensemble/predict"
        }
    }


@app.get("/health")
async def health_check():
    """Health check for Cloud Run."""
    return {
        "status": "healthy",
        "service": "api-gateway",
        "version": "2.0.0-cloud"
    }


# =============================================================================
# Training Endpoints (Synchronous in Cloud)
# =============================================================================

@app.post("/train/{model_type}")
async def train_model(
    model_type: str = Path(..., description="Model type: rf, lstm, xgboost, prophet"),
    train_data: TrainRequest = Body(...)
):
    """
    Train a model synchronously.
    
    In Cloud Run, training is synchronous with extended timeout.
    For very long training jobs, consider using Cloud Tasks.
    """
    if model_type.lower() not in microservices:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model type. Supported: {list(microservices.keys())}"
        )

    service_url = f"{microservices[model_type.lower()]}/train"
    payload = train_data.model_dump(exclude_none=True)

    print(f"[Gateway] Training {model_type} with payload: {payload}")

    async with httpx.AsyncClient() as client:
        try:
            # Extended timeout for training
            response = await client.post(
                service_url, 
                json=payload, 
                timeout=TRAINING_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.ReadTimeout:
            raise HTTPException(
                status_code=504, 
                detail=f"Training timeout. Consider reducing data range or using Cloud Tasks."
            )
        except httpx.HTTPStatusError as exc:
            error_detail = exc.response.text
            try:
                error_detail = exc.response.json().get("detail", error_detail)
            except:
                pass
            raise HTTPException(status_code=exc.response.status_code, detail=error_detail)
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=503, 
                detail=f"Cannot connect to {model_type} service: {str(exc)}"
            )


# =============================================================================
# Prediction Endpoints
# =============================================================================

@app.get("/predict/{model_type}")
async def predict(
    model_type: str = Path(..., description="Model type: rf, lstm, xgboost, prophet"),
    ticket: str = Query("NU", description="Stock ticker symbol"),
    forecast_horizon: int = Query(10, description="Days to forecast"),
    target_col: str = Query("Close", description="Target column"),
    historical_days: int = Query(365, description="Historical days to return")
):
    """Get predictions from a trained model."""
    if model_type.lower() not in microservices:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model type. Supported: {list(microservices.keys())}"
        )

    service_url = f"{microservices[model_type.lower()]}/predict"
    params = {
        "ticket": ticket,
        "forecast_horizon": forecast_horizon,
        "target_col": target_col,
        "history_days": historical_days
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                service_url, 
                params=params, 
                timeout=PREDICTION_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.ReadTimeout:
            raise HTTPException(status_code=504, detail="Prediction timeout")
        except httpx.HTTPStatusError as exc:
            error_detail = exc.response.text
            try:
                error_detail = exc.response.json().get("detail", error_detail)
            except:
                pass
            raise HTTPException(status_code=exc.response.status_code, detail=error_detail)
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail=f"Cannot connect to {model_type} service")


@app.get("/models/{model_type}")
async def list_models(
    model_type: str = Path(..., description="Model type")
):
    """List available trained models."""
    if model_type.lower() not in microservices:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model type. Supported: {list(microservices.keys())}"
        )

    service_url = f"{microservices[model_type.lower()]}/models"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(service_url, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail=f"Cannot connect to {model_type} service")


# =============================================================================
# SHAP Explainer Endpoints
# =============================================================================

@app.post("/explain")
async def explain_prediction(request: ExplainRequest):
    """Get SHAP explanations for model predictions."""
    service_url = f"{microservices['shap']}/explain"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                service_url, 
                json=request.model_dump(),
                timeout=90.0
            )
            response.raise_for_status()
            return response.json()
        except httpx.ReadTimeout:
            raise HTTPException(status_code=504, detail="SHAP explanation timeout")
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail="Cannot connect to SHAP service")


@app.get("/explain/importance/{model_type}")
async def get_feature_importance(
    model_type: str = Path(...),
    ticker: str = Query("NU"),
    max_samples: int = Query(500)
):
    """Get global feature importance using SHAP."""
    service_url = f"{microservices['shap']}/global-importance"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                service_url,
                json={"ticker": ticker, "model_type": model_type, "max_samples": max_samples},
                timeout=90.0
            )
            response.raise_for_status()
            return response.json()
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail="Cannot connect to SHAP service")


@app.get("/explain/plot/{model_type}")
async def get_shap_plot(
    model_type: str = Path(...),
    ticker: str = Query("NU"),
    plot_type: str = Query("bar"),
    max_features: int = Query(15)
):
    """Get SHAP summary plot."""
    service_url = f"{microservices['shap']}/summary-plot"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                service_url,
                params={"ticker": ticker, "model_type": model_type, "plot_type": plot_type, "max_features": max_features},
                timeout=90.0
            )
            response.raise_for_status()
            return response.json()
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail="Cannot connect to SHAP service")


# =============================================================================
# Ensemble Endpoints
# =============================================================================

@app.post("/ensemble/predict")
async def ensemble_predict(request: EnsemblePredictRequest):
    """Get ensemble prediction combining multiple models."""
    service_url = f"{microservices['ensemble']}/predict"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                service_url,
                json=request.model_dump(),
                timeout=180.0  # Longer timeout - calls multiple services
            )
            response.raise_for_status()
            return response.json()
        except httpx.ReadTimeout:
            raise HTTPException(status_code=504, detail="Ensemble prediction timeout")
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail="Cannot connect to ensemble service")


@app.get("/ensemble/compare")
async def compare_predictions(
    ticker: str = Query("NU"),
    forecast_horizon: int = Query(10),
    target_col: str = Query("Close")
):
    """Compare predictions from all models."""
    service_url = f"{microservices['ensemble']}/predict/compare"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                service_url,
                params={"ticker": ticker, "forecast_horizon": forecast_horizon, "target_col": target_col},
                timeout=180.0
            )
            response.raise_for_status()
            return response.json()
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail="Cannot connect to ensemble service")


@app.get("/ensemble/models")
async def list_ensemble_models():
    """List models available for ensemble."""
    service_url = f"{microservices['ensemble']}/models"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(service_url, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail="Cannot connect to ensemble service")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
