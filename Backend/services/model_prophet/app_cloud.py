"""
Prophet Time Series Model Service - Cloud Run Version
=============================================================================
Optimized for Google Cloud Run with synchronous training.
=============================================================================
"""
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
import os
from typing import Optional, Dict
from datetime import datetime, timedelta
import traceback

from .prophet_model import ProphetModel
from utils.import_data import load_data
from utils.gcs_storage import (
    save_model_to_gcs,
    load_model_from_gcs,
    list_models_in_gcs,
    load_model_metadata,
    is_cloud_environment
)


app = FastAPI(
    title="Prophet Time Series Model Service",
    version="2.0.0-cloud",
    description="Cloud Run optimized Prophet forecasting service"
)


class TrainRequest(BaseModel):
    """Training request."""
    ticket: str = "NU"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    training_period: Optional[str] = None
    target_col: str = "Close"
    train_size: float = 0.8


LOCAL_MODEL_DIR = os.getenv("LOCAL_MODEL_DIR", "/app/models/prophet")
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

loaded_models: Dict[str, ProphetModel] = {}


def get_date_range(start_date, end_date, training_period, default_start="2020-12-10"):
    """Determine date range."""
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


def find_model_for_ticker(ticker: str) -> Optional[ProphetModel]:
    """Find and load model."""
    cache_key = f"prophet_{ticker}"
    if cache_key in loaded_models:
        return loaded_models[cache_key]
    
    model = load_model_from_gcs("prophet", ticker)
    if model is not None:
        loaded_models[cache_key] = model
        return model
    
    local_paths = [
        os.path.join(LOCAL_MODEL_DIR, f"prophet_model_{ticker}_latest.joblib"),
        os.path.join(LOCAL_MODEL_DIR, f"prophet_model_{ticker}.joblib"),
    ]
    
    for path in local_paths:
        if os.path.exists(path):
            try:
                model = ProphetModel.load(path)
                loaded_models[cache_key] = model
                return model
            except Exception as e:
                print(f"Error loading from {path}: {e}")
    
    return None


@app.get("/")
async def root():
    return {
        "service": "Prophet Time Series Model Service",
        "version": "2.0.0-cloud",
        "environment": "cloud" if is_cloud_environment() else "local"
    }


@app.post("/train")
async def train_model(request: TrainRequest):
    """Train Prophet model synchronously."""
    print(f"[CLOUD] Training Prophet for: {request.ticket}")
    
    try:
        start_date, end_date = get_date_range(
            request.start_date, request.end_date, request.training_period
        )
        
        data = load_stock_data(request.ticket, start_date, end_date)
        
        if len(data) < 260:
            raise HTTPException(status_code=400, detail="Need at least 260 data points")
        
        # Prepare data for Prophet
        prophet_data = data.reset_index()
        prophet_data = prophet_data.rename(columns={'Date': 'ds', request.target_col: 'y'})
        prophet_data = prophet_data[['ds', 'y']]
        
        # Train Prophet model
        model = ProphetModel()
        model.fit(prophet_data)
        
        # Calculate metrics on test set
        train_size = int(len(prophet_data) * request.train_size)
        train_data = prophet_data.iloc[:train_size]
        test_data = prophet_data.iloc[train_size:]
        
        if len(test_data) > 0:
            predictions = model.predict(test_data[['ds']])
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            import numpy as np
            
            y_true = test_data['y'].values
            y_pred = predictions['yhat'].values
            
            metrics = {
                "mse": float(mean_squared_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "mae": float(mean_absolute_error(y_true, y_pred))
            }
        else:
            metrics = {}
        
        metadata = {
            "ticker": request.ticket,
            "model_type": "Prophet",
            "training_start_date": start_date,
            "training_end_date": end_date,
            "target_col": request.target_col,
            "data_points": len(data),
            "metrics": metrics
        }
        
        model_path = save_model_to_gcs(
            model=model,
            model_type="prophet",
            ticker=request.ticket,
            metadata=metadata,
            start_date=start_date,
            end_date=end_date
        )
        
        loaded_models[f"prophet_{request.ticket}"] = model
        
        return {
            "status": "success",
            "message": f"Prophet model trained for {request.ticket}",
            "ticker": request.ticket,
            "model_type": "Prophet",
            "metrics": metrics,
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
    horizon: int = Query(10),
    target_col: str = Query("Close"),
    history_days: int = Query(365)
):
    """Get predictions."""
    try:
        model = find_model_for_ticker(ticket)
        if model is None:
            raise HTTPException(status_code=404, detail=f"No model found for {ticket}")
        
        metadata = load_model_metadata("prophet", ticket)
        end_date = datetime.now()
        if metadata and "training_end_date" in metadata:
            try:
                end_date = datetime.strptime(metadata["training_end_date"], "%Y-%m-%d")
            except Exception:
                pass
        
        start_date = end_date - timedelta(days=365*3)
        data = load_stock_data(ticket, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        
        # Generate future dates
        last_date = data.index[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=horizon,
            freq='B'
        )
        
        future_df = pd.DataFrame({'ds': future_dates})
        forecast = model.predict(future_df)
        
        predictions = [
            {
                "date": forecast['ds'].iloc[i].strftime('%Y-%m-%d'),
                "prediction": float(forecast['yhat'].iloc[i]),
                "lower_bound": float(forecast['yhat_lower'].iloc[i]),
                "upper_bound": float(forecast['yhat_upper'].iloc[i])
            }
            for i in range(len(forecast))
        ]
        
        historical_data = data.iloc[-history_days:]
        
        return {
            "status": "success",
            "ticker": ticket,
            "forecast_horizon": horizon,
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
    models = list_models_in_gcs("prophet")
    return {"total_models": len(models), "models": models}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "prophet-model", "version": "2.0.0-cloud"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
