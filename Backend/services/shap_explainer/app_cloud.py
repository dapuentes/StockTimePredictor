"""
SHAP Explainer Service - Cloud Run Version
=============================================================================
Cloud-optimized service for model interpretability using SHAP values.
=============================================================================
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict
import os
from datetime import datetime, timedelta

from .shap_explainer import SHAPExplainer
from utils.import_data import load_data
from utils.gcs_storage import load_model_from_gcs, is_cloud_environment


app = FastAPI(
    title="SHAP Explainer API",
    version="2.0.0-cloud",
    description="Cloud Run optimized model interpretability service"
)


class ExplainRequest(BaseModel):
    """Request for prediction explanation."""
    ticker: str = "NU"
    model_type: str = "xgboost"
    top_features: int = 10


class GlobalImportanceRequest(BaseModel):
    """Request for global feature importance."""
    ticker: str = "NU"
    model_type: str = "xgboost"
    max_samples: int = 500


def load_model(model_type: str, ticker: str):
    """Load model from GCS or local storage."""
    # Map model type names
    type_map = {"xgboost": "xgb", "xgb": "xgb", "random_forest": "rf", "rf": "rf"}
    storage_type = type_map.get(model_type.lower(), model_type.lower())
    
    model = load_model_from_gcs(storage_type, ticker)
    if model is not None:
        return model.model if hasattr(model, 'model') else model, model
    
    raise FileNotFoundError(f"No {model_type} model found for {ticker}")


def load_data_for_explanation(ticker: str, model_wrapper, days: int = 365):
    """Load and prepare data for SHAP explanation."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    data = load_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    
    if data.empty:
        raise ValueError(f"No data available for {ticker}")
    
    if hasattr(model_wrapper, 'prepare_data'):
        prepared_data = model_wrapper.prepare_data(data)
    else:
        prepared_data = data
    
    return prepared_data.dropna()


@app.get("/")
def root():
    return {
        "service": "SHAP Explainer API",
        "version": "2.0.0-cloud",
        "environment": "cloud" if is_cloud_environment() else "local",
        "supported_models": ["xgboost", "random_forest"]
    }


@app.post("/explain")
async def explain_prediction(request: ExplainRequest):
    """Explain model predictions using SHAP values."""
    try:
        model, model_wrapper = load_model(request.model_type, request.ticker)
        data = load_data_for_explanation(request.ticker, model_wrapper)
        
        feature_names = data.columns.tolist()
        X = data.tail(min(10, len(data)))
        
        if hasattr(model_wrapper, 'feature_scaler') and model_wrapper.feature_scaler is not None:
            X_scaled = model_wrapper.feature_scaler.transform(X)
        else:
            X_scaled = X.values
        
        explainer = SHAPExplainer(model, model_type=request.model_type)
        explanations = explainer.explain_prediction(X_scaled, feature_names)
        
        explanations["ticker"] = request.ticker
        explanations["model_type"] = request.model_type
        explanations["explanation_date"] = datetime.now().isoformat()
        
        return explanations
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/global-importance")
async def get_global_importance(request: GlobalImportanceRequest):
    """Get global feature importance using SHAP values."""
    try:
        model, model_wrapper = load_model(request.model_type, request.ticker)
        data = load_data_for_explanation(request.ticker, model_wrapper, days=365*2)
        
        feature_names = data.columns.tolist()
        
        if hasattr(model_wrapper, 'feature_scaler') and model_wrapper.feature_scaler is not None:
            X_scaled = model_wrapper.feature_scaler.transform(data)
        else:
            X_scaled = data.values
        
        explainer = SHAPExplainer(model, model_type=request.model_type)
        importance = explainer.get_global_importance(
            X_scaled, 
            feature_names=feature_names,
            max_samples=request.max_samples
        )
        
        importance["ticker"] = request.ticker
        importance["model_type"] = request.model_type
        importance["samples_used"] = min(request.max_samples, len(data))
        importance["generated_at"] = datetime.now().isoformat()
        
        return importance
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/summary-plot")
async def get_summary_plot(
    ticker: str = Query("NU"),
    model_type: str = Query("xgboost"),
    plot_type: str = Query("bar"),
    max_features: int = Query(15)
):
    """Get SHAP summary plot as base64 image."""
    try:
        model, model_wrapper = load_model(model_type, ticker)
        data = load_data_for_explanation(ticker, model_wrapper, days=365)
        
        feature_names = data.columns.tolist()
        
        if hasattr(model_wrapper, 'feature_scaler') and model_wrapper.feature_scaler is not None:
            X_scaled = model_wrapper.feature_scaler.transform(data)
        else:
            X_scaled = data.values
        
        explainer = SHAPExplainer(model, model_type=model_type)
        plot_data = explainer.generate_summary_plot(
            X_scaled,
            feature_names=feature_names,
            plot_type=plot_type,
            max_display=max_features
        )
        
        return {
            "ticker": ticker,
            "model_type": model_type,
            "plot_type": plot_type,
            "plot_image": plot_data,
            "generated_at": datetime.now().isoformat()
        }
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/waterfall-plot")
async def get_waterfall_plot(
    ticker: str = Query("NU"),
    model_type: str = Query("xgboost"),
    prediction_index: int = Query(0),
    max_features: int = Query(10)
):
    """Get SHAP waterfall plot for single prediction."""
    try:
        model, model_wrapper = load_model(model_type, ticker)
        data = load_data_for_explanation(ticker, model_wrapper, days=365)
        
        feature_names = data.columns.tolist()
        X = data.tail(min(10, len(data)))
        
        if hasattr(model_wrapper, 'feature_scaler') and model_wrapper.feature_scaler is not None:
            X_scaled = model_wrapper.feature_scaler.transform(X)
        else:
            X_scaled = X.values
        
        explainer = SHAPExplainer(model, model_type=model_type)
        plot_data = explainer.generate_waterfall_plot(
            X_scaled,
            feature_names=feature_names,
            index=prediction_index,
            max_display=max_features
        )
        
        return {
            "ticker": ticker,
            "model_type": model_type,
            "prediction_index": prediction_index,
            "plot_image": plot_data,
            "generated_at": datetime.now().isoformat()
        }
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "shap-explainer", "version": "2.0.0-cloud"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
