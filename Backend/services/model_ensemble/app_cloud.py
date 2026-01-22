"""
Ensemble Model Service - Cloud Run Version
=============================================================================
Combines predictions from multiple models for improved accuracy.
=============================================================================
"""
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import httpx
import asyncio
from datetime import datetime
import os
import numpy as np

app = FastAPI(
    title="Ensemble Model API",
    version="2.0.0-cloud",
    description="Cloud Run optimized ensemble prediction service"
)

# Service URLs from environment
SERVICE_URLS = {
    "rf": os.getenv("RF_SERVICE_URL", "http://model-rf:8001"),
    "lstm": os.getenv("LSTM_SERVICE_URL", "http://model-lstm:8002"),
    "xgboost": os.getenv("XGB_SERVICE_URL", "http://model-xgb:8003"),
    "prophet": os.getenv("PROPHET_SERVICE_URL", "http://model-prophet:8004"),
}


class EnsemblePredictRequest(BaseModel):
    """Ensemble prediction request."""
    ticker: str = "NU"
    forecast_horizon: int = 10
    target_col: str = "Close"
    models: Optional[List[str]] = None
    ensemble_method: str = "weighted_average"


class ModelWeight(BaseModel):
    """Custom model weight."""
    model: str
    weight: float


async def fetch_prediction(
    client: httpx.AsyncClient,
    model_name: str,
    ticker: str,
    horizon: int,
    target_col: str
) -> Dict[str, Any]:
    """Fetch prediction from a model service."""
    url = f"{SERVICE_URLS[model_name]}/predict"
    
    try:
        params = {
            "ticket": ticker,
            "forecast_horizon": horizon,
            "target_col": target_col
        }
        
        # Prophet uses 'horizon' parameter
        if model_name == "prophet":
            params = {"ticket": ticker, "horizon": horizon, "target_col": target_col}
        
        response = await client.get(url, params=params, timeout=60.0)
        response.raise_for_status()
        
        return {
            "model": model_name,
            "status": "success",
            "data": response.json()
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
def root():
    return {
        "service": "Ensemble Model API",
        "version": "2.0.0-cloud",
        "available_models": list(SERVICE_URLS.keys()),
        "ensemble_methods": ["simple_average", "weighted_average", "median"]
    }


@app.post("/predict")
async def ensemble_predict(request: EnsemblePredictRequest):
    """Get ensemble prediction combining multiple models."""
    models_to_use = request.models or list(SERVICE_URLS.keys())
    
    # Validate models
    invalid_models = [m for m in models_to_use if m not in SERVICE_URLS]
    if invalid_models:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid models: {invalid_models}. Available: {list(SERVICE_URLS.keys())}"
        )
    
    # Fetch predictions from all models
    async with httpx.AsyncClient() as client:
        tasks = [
            fetch_prediction(client, model, request.ticker, request.forecast_horizon, request.target_col)
            for model in models_to_use
        ]
        results = await asyncio.gather(*tasks)
    
    # Process results
    successful_predictions = []
    failed_models = []
    
    for result in results:
        if result["status"] == "success":
            successful_predictions.append(result)
        else:
            failed_models.append({"model": result["model"], "error": result["error"]})
    
    if not successful_predictions:
        raise HTTPException(
            status_code=503,
            detail=f"All model predictions failed: {failed_models}"
        )
    
    # Extract predictions and combine
    all_predictions = []
    model_names = []
    
    for pred in successful_predictions:
        model_names.append(pred["model"])
        preds = pred["data"].get("predictions", [])
        all_predictions.append([p["prediction"] for p in preds])
    
    # Convert to numpy for calculation
    pred_array = np.array(all_predictions)
    
    # Calculate ensemble based on method
    if request.ensemble_method == "simple_average":
        ensemble_preds = np.mean(pred_array, axis=0)
    elif request.ensemble_method == "median":
        ensemble_preds = np.median(pred_array, axis=0)
    else:  # weighted_average (equal weights for now)
        ensemble_preds = np.mean(pred_array, axis=0)
    
    # Calculate uncertainty
    std_dev = np.std(pred_array, axis=0)
    
    # Get dates from first successful prediction
    first_pred = successful_predictions[0]["data"]
    dates = [p["date"] for p in first_pred.get("predictions", [])]
    
    ensemble_predictions = [
        {
            "date": dates[i] if i < len(dates) else None,
            "prediction": float(ensemble_preds[i]),
            "lower_bound": float(ensemble_preds[i] - 1.96 * std_dev[i]),
            "upper_bound": float(ensemble_preds[i] + 1.96 * std_dev[i]),
            "std_dev": float(std_dev[i])
        }
        for i in range(len(ensemble_preds))
    ]
    
    return {
        "status": "success",
        "ticker": request.ticker,
        "forecast_horizon": request.forecast_horizon,
        "ensemble_method": request.ensemble_method,
        "models_used": model_names,
        "failed_models": failed_models,
        "predictions": ensemble_predictions,
        "individual_predictions": {
            pred["model"]: pred["data"].get("predictions", [])
            for pred in successful_predictions
        },
        "historical_dates": first_pred.get("historical_dates", []),
        "historical_values": first_pred.get("historical_values", []),
        "generated_at": datetime.now().isoformat()
    }


@app.get("/predict/compare")
async def compare_predictions(
    ticker: str = Query("NU"),
    forecast_horizon: int = Query(10),
    target_col: str = Query("Close")
):
    """Compare predictions from all models."""
    async with httpx.AsyncClient() as client:
        tasks = [
            fetch_prediction(client, model, ticker, forecast_horizon, target_col)
            for model in SERVICE_URLS.keys()
        ]
        results = await asyncio.gather(*tasks)
    
    comparison = {}
    for result in results:
        model = result["model"]
        if result["status"] == "success":
            comparison[model] = {
                "status": "success",
                "predictions": result["data"].get("predictions", []),
                "last_actual_value": result["data"].get("last_actual_value")
            }
        else:
            comparison[model] = {
                "status": "error",
                "error": result["error"]
            }
    
    return {
        "ticker": ticker,
        "forecast_horizon": forecast_horizon,
        "comparison": comparison,
        "generated_at": datetime.now().isoformat()
    }


@app.get("/models")
async def list_models():
    """List available models and their status."""
    model_status = {}
    
    async with httpx.AsyncClient() as client:
        for model, url in SERVICE_URLS.items():
            try:
                response = await client.get(f"{url}/health", timeout=5.0)
                model_status[model] = {
                    "available": response.status_code == 200,
                    "url": url
                }
            except Exception:
                model_status[model] = {
                    "available": False,
                    "url": url
                }
    
    return {
        "models": model_status,
        "total_available": sum(1 for m in model_status.values() if m["available"])
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ensemble-model", "version": "2.0.0-cloud"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
