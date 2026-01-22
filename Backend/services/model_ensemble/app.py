"""
FastAPI Service for Ensemble Model Predictions
Combines predictions from multiple models for improved accuracy
"""
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import httpx
import asyncio
from datetime import datetime, timedelta
import os

app = FastAPI(
    title="Ensemble Model API",
    description="Combines predictions from RF, LSTM, XGBoost, and Prophet models",
    version="1.0.0",
)

# Service URLs (from environment or defaults)
SERVICE_URLS = {
    "rf": os.getenv("RF_SERVICE_URL", "http://model-rf-api:8001"),
    "lstm": os.getenv("LSTM_SERVICE_URL", "http://model-lstm-api:8002"),
    "xgboost": os.getenv("XGB_SERVICE_URL", "http://model-xgb-api:8003"),
    "prophet": os.getenv("PROPHET_SERVICE_URL", "http://model-prophet-api:8004"),
}


class EnsemblePredictRequest(BaseModel):
    """Request model for ensemble prediction."""
    ticker: str = "NU"
    forecast_horizon: int = 10
    target_col: str = "Close"
    models: Optional[List[str]] = None  # None = use all models
    ensemble_method: str = "weighted_average"  # simple_average, weighted_average, median, best_model


class ModelWeight(BaseModel):
    """Custom model weight configuration."""
    model: str
    weight: float


class CustomWeightsRequest(BaseModel):
    """Request with custom model weights."""
    ticker: str = "NU"
    forecast_horizon: int = 10
    target_col: str = "Close"
    weights: List[ModelWeight]


async def fetch_prediction(
    client: httpx.AsyncClient,
    model_name: str,
    ticker: str,
    horizon: int,
    target_col: str
) -> Dict[str, Any]:
    """Fetch prediction from a single model service."""
    url = f"{SERVICE_URLS[model_name]}/predict"
    
    try:
        # Build params based on model
        params = {
            "ticket": ticker,
            "forecast_horizon": horizon,
            "target_col": target_col
        }
        
        # Prophet uses 'horizon' instead of 'forecast_horizon'
        if model_name == "prophet":
            params = {
                "ticket": ticker,
                "horizon": horizon,
                "target_col": target_col
            }
        
        response = await client.get(url, params=params, timeout=60.0)
        response.raise_for_status()
        data = response.json()
        
        return {
            "model": model_name,
            "status": "success",
            "data": data
        }
        
    except httpx.HTTPStatusError as e:
        return {
            "model": model_name,
            "status": "error",
            "error": f"HTTP {e.response.status_code}: {e.response.text}"
        }
    except httpx.RequestError as e:
        return {
            "model": model_name,
            "status": "error",
            "error": f"Connection error: {str(e)}"
        }
    except Exception as e:
        return {
            "model": model_name,
            "status": "error",
            "error": str(e)
        }


@app.get("/")
def read_root():
    """Root endpoint."""
    return {
        "service": "Ensemble Model API",
        "version": "1.0.0",
        "description": "Combines predictions from RF, LSTM, XGBoost, and Prophet",
        "endpoints": [
            "/predict - Get ensemble prediction",
            "/predict/compare - Compare all model predictions",
            "/models - List available models"
        ]
    }


@app.post("/predict")
async def ensemble_predict(request: EnsemblePredictRequest):
    """
    Get ensemble prediction combining multiple models.
    
    The ensemble combines predictions using the specified method:
    - simple_average: Equal weight to all models
    - weighted_average: Weights based on model performance (MAE)
    - median: Median of predictions (robust to outliers)
    - best_model: Use only the best performing model
    """
    from model_ensemble.ensemble_model import EnsembleModel, EnsembleMethod, create_ensemble_from_services
    
    # Determine which models to use
    models_to_use = request.models or list(SERVICE_URLS.keys())
    
    # Validate models
    invalid_models = [m for m in models_to_use if m not in SERVICE_URLS]
    if invalid_models:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid models: {invalid_models}. Available: {list(SERVICE_URLS.keys())}"
        )
    
    # Fetch predictions from all models in parallel
    async with httpx.AsyncClient() as client:
        tasks = [
            fetch_prediction(client, model, request.ticker, request.forecast_horizon, request.target_col)
            for model in models_to_use
        ]
        results = await asyncio.gather(*tasks)
    
    # Process results
    successful_predictions = {}
    failed_models = []
    
    for result in results:
        if result["status"] == "success":
            model_data = result["data"]
            predictions = model_data.get("predictions", [])
            
            # Extract prediction values
            if isinstance(predictions, list) and len(predictions) > 0:
                if isinstance(predictions[0], dict):
                    pred_values = [p.get("prediction", p.get("value", 0)) for p in predictions]
                else:
                    pred_values = predictions
                
                successful_predictions[result["model"]] = {
                    "predictions": pred_values,
                    "metrics": model_data.get("metrics", {}),
                    "raw_response": model_data
                }
        else:
            failed_models.append({
                "model": result["model"],
                "error": result["error"]
            })
    
    if not successful_predictions:
        raise HTTPException(
            status_code=503,
            detail=f"No models returned predictions. Errors: {failed_models}"
        )
    
    # Create ensemble prediction
    try:
        ensemble_result = create_ensemble_from_services(
            successful_predictions,
            method=request.ensemble_method
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating ensemble: {str(e)}")
    
    # Format response
    base_date = datetime.now()
    prediction_dates = [
        (base_date + timedelta(days=i+1)).strftime("%Y-%m-%d")
        for i in range(len(ensemble_result["predictions"]))
    ]
    
    return {
        "ticker": request.ticker,
        "target_col": request.target_col,
        "forecast_horizon": request.forecast_horizon,
        "ensemble_method": request.ensemble_method,
        "predictions": [
            {
                "date": date,
                "prediction": float(pred),
                "lower_bound": float(lb),
                "upper_bound": float(ub)
            }
            for date, pred, lb, ub in zip(
                prediction_dates,
                ensemble_result["predictions"],
                ensemble_result["lower_bound"],
                ensemble_result["upper_bound"]
            )
        ],
        "confidence_level": ensemble_result["confidence_level"],
        "model_agreement": ensemble_result["model_agreement"],
        "models_used": list(successful_predictions.keys()),
        "failed_models": failed_models,
        "model_contributions": ensemble_result["model_contributions"],
        "metadata": ensemble_result["metadata"],
        "generated_at": datetime.now().isoformat()
    }


@app.get("/predict/compare")
async def compare_predictions(
    ticker: str = Query("NU", description="Stock ticker"),
    forecast_horizon: int = Query(10, description="Forecast horizon in days"),
    target_col: str = Query("Close", description="Target column")
):
    """
    Get and compare predictions from all available models.
    
    Useful for understanding model agreement and disagreement.
    """
    models_to_use = list(SERVICE_URLS.keys())
    
    # Fetch predictions from all models
    async with httpx.AsyncClient() as client:
        tasks = [
            fetch_prediction(client, model, ticker, forecast_horizon, target_col)
            for model in models_to_use
        ]
        results = await asyncio.gather(*tasks)
    
    # Process and format results
    comparison = {}
    all_predictions = []
    
    for result in results:
        model_name = result["model"]
        
        if result["status"] == "success":
            model_data = result["data"]
            predictions = model_data.get("predictions", [])
            
            # Extract values
            if isinstance(predictions, list) and len(predictions) > 0:
                if isinstance(predictions[0], dict):
                    pred_values = [p.get("prediction", p.get("value", 0)) for p in predictions]
                else:
                    pred_values = predictions
                
                comparison[model_name] = {
                    "status": "success",
                    "predictions": pred_values,
                    "mean": float(sum(pred_values) / len(pred_values)) if pred_values else None,
                    "metrics": model_data.get("metrics", {}),
                }
                all_predictions.append(pred_values)
        else:
            comparison[model_name] = {
                "status": "error",
                "error": result["error"]
            }
    
    # Calculate agreement metrics
    import numpy as np
    if len(all_predictions) > 1:
        all_preds_array = np.array(all_predictions)
        agreement = {
            "mean_std": float(np.mean(np.std(all_preds_array, axis=0))),
            "correlation_matrix": np.corrcoef(all_preds_array).tolist(),
            "range_overlap": float(1 - np.mean(np.ptp(all_preds_array, axis=0)) / (np.mean(all_preds_array) + 1e-6))
        }
    else:
        agreement = {"note": "Not enough models for agreement calculation"}
    
    return {
        "ticker": ticker,
        "forecast_horizon": forecast_horizon,
        "target_col": target_col,
        "model_predictions": comparison,
        "agreement_metrics": agreement,
        "generated_at": datetime.now().isoformat()
    }


@app.get("/models")
async def list_models():
    """List available models and their status."""
    model_status = {}
    
    async with httpx.AsyncClient() as client:
        for model_name, url in SERVICE_URLS.items():
            try:
                response = await client.get(f"{url}/health", timeout=5.0)
                model_status[model_name] = {
                    "url": url,
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "status_code": response.status_code
                }
            except Exception as e:
                model_status[model_name] = {
                    "url": url,
                    "status": "unavailable",
                    "error": str(e)
                }
    
    return {
        "available_models": model_status,
        "ensemble_methods": [
            "simple_average",
            "weighted_average", 
            "median",
            "best_model"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ensemble-model"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
