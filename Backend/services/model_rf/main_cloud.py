"""
Random Forest Time Series Model Service - Cloud Run Version
=============================================================================
This version is optimized for Google Cloud Run:
- Synchronous training (no Celery/Redis)
- GCS integration for model storage
- Extended timeout support (up to 10 minutes for training)
- Stateless design
=============================================================================
"""
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel
import pandas as pd
import os
from typing import Optional, Tuple, Dict, Any
import glob
from datetime import datetime, timedelta
import traceback

# Local imports
from .rf_model import TimeSeriesRandomForestModel
from .train import train_ts_model
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
    title="Random Forest Time Series Model Service",
    version="2.0.0-cloud",
    description="Cloud Run optimized service for RF time series forecasting"
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


class TrainResponse(BaseModel):
    """Training response model."""
    status: str
    message: str
    ticker: str
    model_type: str = "RandomForest"
    metrics: Optional[Dict] = None
    features_names: Optional[list] = None
    best_params: Optional[Dict] = None
    residuals: Optional[list] = None
    residual_dates: Optional[list] = None
    acf: Optional[Dict] = None
    pacf: Optional[Dict] = None
    model_path: Optional[str] = None


# Model storage paths
LOCAL_MODEL_DIR = os.getenv("LOCAL_MODEL_DIR", "/app/models/rf")
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# In-memory model cache
loaded_models: Dict[str, TimeSeriesRandomForestModel] = {}


def actual_date_range(
    start_date: Optional[str],
    end_date: Optional[str],
    training_period: Optional[str],
    default_start: str = "2020-12-10"
) -> Tuple[str, str]:
    """Determine the actual date range for training."""
    
    if start_date and end_date:
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Dates must be in YYYY-MM-DD format.")
        
        if start_dt > end_dt:
            raise HTTPException(status_code=400, detail="Start date cannot be after end date.")
        
        return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")
        
    elif training_period:
        end_dt = datetime.now()
        period_days = {
            "1_year": 365,
            "3_years": 365 * 3,
            "5_years": 365 * 5,
            "10_years": 365 * 10
        }
        days = period_days.get(training_period, 365 * 3)
        start_dt = end_dt - timedelta(days=days)
    else:
        start_dt = datetime.strptime(default_start, "%Y-%m-%d")
        end_dt = datetime.now()

    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


def load_stock_data(ticket: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load stock data from yfinance."""
    try:
        data = load_data(ticker=ticket, start_date=start_date, end_date=end_date)
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for ticker {ticket}")
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading data: {str(e)}")


def find_model_for_ticker(ticker: str) -> Optional[TimeSeriesRandomForestModel]:
    """Find and load a model for the given ticker."""
    # Check cache first
    cache_key = f"rf_{ticker}"
    if cache_key in loaded_models:
        return loaded_models[cache_key]
    
    # Try to load from GCS or local storage
    model = load_model_from_gcs("rf", ticker)
    
    if model is not None:
        loaded_models[cache_key] = model
        return model
    
    # Try local file system as fallback
    local_paths = [
        os.path.join(LOCAL_MODEL_DIR, f"rf_model_{ticker}_latest.joblib"),
        os.path.join(LOCAL_MODEL_DIR, f"rf_model_{ticker}.joblib"),
        os.path.join(LOCAL_MODEL_DIR, "rf_model.joblib")
    ]
    
    for path in local_paths:
        if os.path.exists(path):
            try:
                model = TimeSeriesRandomForestModel.load_model(path)
                loaded_models[cache_key] = model
                return model
            except Exception as e:
                print(f"Error loading model from {path}: {e}")
    
    return None


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def read_root():
    """Root endpoint with service info."""
    return {
        "service": "Random Forest Time Series Model Service",
        "version": "2.0.0-cloud",
        "environment": "cloud" if is_cloud_environment() else "local",
        "endpoints": ["/train", "/predict", "/models", "/health"]
    }


@app.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest):
    """
    Train a Random Forest model synchronously.
    
    In Cloud Run, training runs synchronously with extended timeout.
    For long-running training, consider using Cloud Tasks.
    """
    print(f"[CLOUD] Training request received for ticker: {request.ticket}")
    
    try:
        # Determine date range
        start_date, end_date = actual_date_range(
            request.start_date,
            request.end_date,
            request.training_period
        )
        print(f"[CLOUD] Training date range: {start_date} to {end_date}")
        
        # Load stock data
        data = load_stock_data(request.ticket, start_date, end_date)
        print(f"[CLOUD] Loaded {len(data)} rows of data")
        
        # Validate data size
        min_days = 260
        if len(data) < min_days:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough data for training. Need at least {min_days} rows, got {len(data)}."
            )
        
        # Train the model (synchronous)
        print(f"[CLOUD] Starting model training...")
        model, feature_names, residuals, residual_dates, acf_values, pacf_values, confint_acf, confint_pacf = train_ts_model(
            data=data,
            n_lags=request.n_lags,
            target_col=request.target_col,
            train_size=request.train_size,
            save_model_path=None,  # We'll save using GCS
            bucket_name=None
        )
        print(f"[CLOUD] Model training completed")
        
        # Prepare metadata
        metadata = {
            "ticker": request.ticket,
            "model_type": "RandomForest",
            "training_start_date": start_date,
            "training_end_date": end_date,
            "n_lags": request.n_lags,
            "target_col": request.target_col,
            "train_size": request.train_size,
            "data_points": len(data),
            "metrics": model.metrics if hasattr(model, 'metrics') else {},
            "best_params": model.best_params_ if hasattr(model, 'best_params_') else {},
            "feature_names": feature_names
        }
        
        # Save model to GCS or local storage
        model_path = save_model_to_gcs(
            model=model,
            model_type="rf",
            ticker=request.ticket,
            metadata=metadata,
            start_date=start_date,
            end_date=end_date
        )
        print(f"[CLOUD] Model saved to: {model_path}")
        
        # Update cache
        loaded_models[f"rf_{request.ticket}"] = model
        
        # Prepare response
        return TrainResponse(
            status="success",
            message=f"Model trained successfully for {request.ticket}",
            ticker=request.ticket,
            model_type="RandomForest",
            metrics=model.metrics if hasattr(model, 'metrics') else None,
            features_names=feature_names,
            best_params=model.best_params_ if hasattr(model, 'best_params_') else None,
            residuals=residuals.tolist() if residuals is not None else None,
            residual_dates=[d.strftime("%Y-%m-%d") for d in residual_dates] if residual_dates is not None else None,
            acf={
                "values": acf_values.tolist() if acf_values is not None else [],
                "confint_lower": confint_acf[:, 0].tolist() if confint_acf is not None else [],
                "confint_upper": confint_acf[:, 1].tolist() if confint_acf is not None else []
            } if acf_values is not None else None,
            pacf={
                "values": pacf_values.tolist() if pacf_values is not None else [],
                "confint_lower": confint_pacf[:, 0].tolist() if confint_pacf is not None else [],
                "confint_upper": confint_pacf[:, 1].tolist() if confint_pacf is not None else []
            } if pacf_values is not None else None,
            model_path=model_path
        )
        
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.get("/predict")
async def predict(
    ticket: str = Query("NU", description="Stock ticker symbol"),
    forecast_horizon: int = Query(10, description="Number of days to forecast"),
    target_col: str = Query("Close", description="Target column for prediction"),
    history_days: int = Query(365, description="Historical days to return")
):
    """Make predictions using a trained model."""
    try:
        print(f"[CLOUD] Prediction request for ticker: {ticket}")
        
        # Find model
        model = find_model_for_ticker(ticket)
        if model is None:
            raise HTTPException(
                status_code=404,
                detail=f"No trained model found for {ticket}. Train a model first."
            )
        
        # Load metadata to get training end date
        metadata = load_model_metadata("rf", ticket)
        if metadata and "training_end_date" in metadata:
            end_date = datetime.strptime(metadata["training_end_date"], "%Y-%m-%d")
        else:
            end_date = datetime.now()
        
        # Load historical data
        start_date = end_date - timedelta(days=365 * 3)
        data = load_stock_data(ticket, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        
        if target_col not in data.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{target_col}' not found. Available: {data.columns.tolist()}"
            )
        
        # Generate forecast
        forecast, lower_bounds, upper_bounds = forecast_future_prices(
            model=model,
            data=data.copy(),
            forecast_horizon=forecast_horizon,
            target_col=target_col
        )
        
        # Prepare response
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
                "lower_bound": float(lower_bounds[i]),
                "upper_bound": float(upper_bounds[i])
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
    """List all available models."""
    try:
        models = list_models_in_gcs("rf")
        return {
            "total_models": len(models),
            "models": models,
            "storage": "GCS" if is_cloud_environment() else "local"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run."""
    return {
        "status": "healthy",
        "service": "rf-model",
        "version": "2.0.0-cloud",
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
