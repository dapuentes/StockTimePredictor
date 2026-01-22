"""
XGBoost Time Series Model Service - Cloud Run Version
=============================================================================
Optimized for Google Cloud Run:
- Synchronous training
- GCS integration for model storage
- Extended timeout support
=============================================================================
"""
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
import os
from typing import Optional, Dict
from datetime import datetime, timedelta
import traceback

# Local imports - adjusted for cloud structure
from .xgb_model import XGBoostModel
from .forecast import forecast_future_prices
from utils.import_data import load_data
from utils.gcs_storage import (
    save_model_to_gcs,
    load_model_from_gcs,
    list_models_in_gcs,
    load_model_metadata,
    is_cloud_environment
)


app = FastAPI(
    title="XGBoost Time Series Model Service",
    version="2.0.0-cloud",
    description="Cloud Run optimized service for XGBoost time series forecasting"
)


class TrainRequest(BaseModel):
    """Training request model."""
    ticket: str = "NU"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    training_period: Optional[str] = None
    n_lags: int = 10
    target_col: str = "Close"
    train_size: float = 0.8


# Model storage paths
LOCAL_MODEL_DIR = os.getenv("LOCAL_MODEL_DIR", "/app/models/xgb")
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# In-memory model cache
loaded_models: Dict[str, XGBoostModel] = {}


def get_date_range(start_date, end_date, training_period, default_start="2020-12-10"):
    """Determine date range for training."""
    if start_date and end_date:
        return start_date, end_date
    
    end_dt = datetime.now()
    if training_period:
        periods = {"1_year": 365, "3_years": 365*3, "5_years": 365*5, "10_years": 365*10}
        days = periods.get(training_period, 365*3)
        start_dt = end_dt - timedelta(days=days)
    else:
        start_dt = datetime.strptime(default_start, "%Y-%m-%d")
    
    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


def load_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load stock data."""
    try:
        data = load_data(ticker=ticker, start_date=start_date, end_date=end_date)
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")


def find_model_for_ticker(ticker: str) -> Optional[XGBoostModel]:
    """Find and load model for ticker."""
    cache_key = f"xgb_{ticker}"
    if cache_key in loaded_models:
        return loaded_models[cache_key]
    
    model = load_model_from_gcs("xgb", ticker)
    if model is not None:
        loaded_models[cache_key] = model
        return model
    
    # Try local files
    local_paths = [
        os.path.join(LOCAL_MODEL_DIR, f"xgb_model_{ticker}_latest.joblib"),
        os.path.join(LOCAL_MODEL_DIR, f"xgb_model_{ticker}.joblib"),
        os.path.join(LOCAL_MODEL_DIR, "xgb_model.joblib")
    ]
    
    for path in local_paths:
        if os.path.exists(path):
            try:
                model = XGBoostModel.load(path)
                loaded_models[cache_key] = model
                return model
            except Exception as e:
                print(f"Error loading from {path}: {e}")
    
    return None


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "XGBoost Time Series Model Service",
        "version": "2.0.0-cloud",
        "environment": "cloud" if is_cloud_environment() else "local"
    }


@app.post("/train")
async def train_model(request: TrainRequest):
    """Train XGBoost model synchronously."""
    print(f"[CLOUD] Training XGBoost for ticker: {request.ticket}")
    
    try:
        start_date, end_date = get_date_range(
            request.start_date, request.end_date, request.training_period
        )
        
        data = load_stock_data(request.ticket, start_date, end_date)
        print(f"[CLOUD] Loaded {len(data)} rows")
        
        if len(data) < 260:
            raise HTTPException(status_code=400, detail=f"Need at least 260 data points, got {len(data)}")
        
        # Train model
        model = XGBoostModel(n_lags=request.n_lags)
        
        # Prepare data
        processed_data = model.prepare_data(data, target_col=request.target_col)
        
        # Split data
        train_size = int(len(processed_data) * request.train_size)
        train_data = processed_data.iloc[:train_size]
        test_data = processed_data.iloc[train_size:]
        
        X_train = train_data.drop(columns=[request.target_col])
        y_train = train_data[request.target_col]
        X_test = test_data.drop(columns=[request.target_col])
        y_test = test_data[request.target_col]
        
        # Train
        model.fit(X_train, y_train, X_test, y_test)
        
        # Prepare metadata
        metadata = {
            "ticker": request.ticket,
            "model_type": "XGBoost",
            "training_start_date": start_date,
            "training_end_date": end_date,
            "n_lags": request.n_lags,
            "target_col": request.target_col,
            "data_points": len(data),
            "metrics": model.metrics if hasattr(model, 'metrics') else {}
        }
        
        # Save model
        model_path = save_model_to_gcs(
            model=model,
            model_type="xgb",
            ticker=request.ticket,
            metadata=metadata,
            start_date=start_date,
            end_date=end_date
        )
        
        loaded_models[f"xgb_{request.ticket}"] = model
        
        return {
            "status": "success",
            "message": f"XGBoost model trained for {request.ticket}",
            "ticker": request.ticket,
            "model_type": "XGBoost",
            "metrics": model.metrics if hasattr(model, 'metrics') else None,
            "model_path": model_path
        }
        
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.get("/predict")
async def predict(
    ticket: str = Query("NU"),
    forecast_horizon: int = Query(10),
    target_col: str = Query("Close"),
    history_days: int = Query(365)
):
    """Get predictions."""
    try:
        model = find_model_for_ticker(ticket)
        if model is None:
            raise HTTPException(status_code=404, detail=f"No model found for {ticket}")
        
        # Get metadata for end date
        metadata = load_model_metadata("xgb", ticket)
        end_date = datetime.now()
        if metadata and "training_end_date" in metadata:
            try:
                end_date = datetime.strptime(metadata["training_end_date"], "%Y-%m-%d")
            except:
                pass
        
        start_date = end_date - timedelta(days=365*3)
        data = load_stock_data(ticket, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        
        # Generate forecast
        forecast, lower_bounds, upper_bounds = forecast_future_prices(
            model=model,
            data=data.copy(),
            forecast_horizon=forecast_horizon,
            target_col=target_col
        )
        
        last_date = data.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=forecast_horizon,
            freq='B'
        ).strftime('%Y-%m-%d').tolist()
        
        predictions = [
            {
                "date": forecast_dates[i],
                "prediction": float(forecast[i]),
                "lower_bound": float(lower_bounds[i]) if lower_bounds is not None else None,
                "upper_bound": float(upper_bounds[i]) if upper_bounds is not None else None
            }
            for i in range(len(forecast_dates))
        ]
        
        historical_data = data.iloc[-history_days:]
        
        return {
            "status": "success",
            "ticker": ticket,
            "target_column": target_col,
            "forecast_horizon": forecast_horizon,
            "historical_dates": historical_data.index.strftime('%Y-%m-%d').tolist(),
            "historical_values": historical_data[target_col].tolist(),
            "predictions": predictions,
            "last_actual_date": last_date.strftime("%Y-%m-%d"),
            "last_actual_value": float(data[target_col].iloc[-1])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/models")
async def list_models():
    """List available models."""
    models = list_models_in_gcs("xgb")
    return {"total_models": len(models), "models": models}


@app.get("/health")
async def health_check():
    """Health check."""
    return {"status": "healthy", "service": "xgb-model", "version": "2.0.0-cloud"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
