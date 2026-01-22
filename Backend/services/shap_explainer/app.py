"""
FastAPI Service for SHAP Model Explanations
Provides endpoints to explain predictions from XGBoost and Random Forest models
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import sys
from datetime import datetime, timedelta

# Add parent path for imports
sys.path.insert(0, '/app')

app = FastAPI(
    title="SHAP Explainer API",
    description="Model Interpretability Service using SHAP values",
    version="1.0.0",
)


class ExplainRequest(BaseModel):
    """Request model for prediction explanation."""
    ticker: str = "NU"
    model_type: str = "xgboost"  # "xgboost" or "random_forest"
    prediction_date: Optional[str] = None
    top_features: int = 10


class GlobalImportanceRequest(BaseModel):
    """Request model for global feature importance."""
    ticker: str = "NU"
    model_type: str = "xgboost"
    max_samples: int = 500


# Model directories
XGB_MODEL_DIR = "/app/services_code/model_xgb/models"
RF_MODEL_DIR = "/app/services_code/model_rf/models"


def load_model(model_type: str, ticker: str):
    """Load a trained model based on type and ticker."""
    if model_type in ["xgboost", "xgb"]:
        from model_xgb.xgb_model import XGBoostModel
        model_path = os.path.join(XGB_MODEL_DIR, f"xgb_model_{ticker}.joblib")
        if not os.path.exists(model_path):
            model_path = os.path.join(XGB_MODEL_DIR, "xgb_model.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No XGBoost model found for {ticker}")
        model_wrapper = XGBoostModel.load(model_path)
        return model_wrapper.model, model_wrapper
        
    elif model_type in ["random_forest", "rf"]:
        from model_rf.rf_model import RandomForestModel
        model_path = os.path.join(RF_MODEL_DIR, f"rf_model_{ticker}.joblib")
        if not os.path.exists(model_path):
            model_path = os.path.join(RF_MODEL_DIR, "rf_model.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No Random Forest model found for {ticker}")
        model_wrapper = RandomForestModel.load(model_path)
        return model_wrapper.model, model_wrapper
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def load_data_for_explanation(ticker: str, model_wrapper, days: int = 365):
    """Load and prepare data for SHAP explanation."""
    from utils.import_data import load_data
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    data = load_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    
    if data.empty:
        raise ValueError(f"No data available for {ticker}")
    
    # Prepare data using model's prepare_data method
    if hasattr(model_wrapper, 'prepare_data'):
        prepared_data = model_wrapper.prepare_data(data)
    else:
        prepared_data = data
    
    # Drop NaN values
    prepared_data = prepared_data.dropna()
    
    return prepared_data


@app.get("/")
def read_root():
    """Root endpoint."""
    return {
        "service": "SHAP Explainer API",
        "version": "1.0.0",
        "supported_models": ["xgboost", "random_forest"],
        "endpoints": [
            "/explain - Explain individual predictions",
            "/global-importance - Get global feature importance",
            "/summary-plot - Generate SHAP summary plot",
            "/waterfall-plot - Generate waterfall plot for single prediction"
        ]
    }


@app.post("/explain")
async def explain_prediction(request: ExplainRequest):
    """
    Explain model predictions using SHAP values.
    
    Returns feature contributions for the most recent predictions.
    """
    try:
        from shap_explainer.shap_explainer import SHAPExplainer
        
        # Load model
        model, model_wrapper = load_model(request.model_type, request.ticker)
        
        # Load and prepare data
        data = load_data_for_explanation(request.ticker, model_wrapper)
        
        # Get feature names
        feature_names = data.columns.tolist()
        
        # Use last N rows for explanation (most recent data)
        X = data.tail(min(10, len(data)))
        
        # Scale data if scaler exists
        if hasattr(model_wrapper, 'feature_scaler') and model_wrapper.feature_scaler is not None:
            X_scaled = model_wrapper.feature_scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Create explainer and get explanations
        explainer = SHAPExplainer(model, model_type=request.model_type)
        explanations = explainer.explain_prediction(X_scaled, feature_names)
        
        # Add metadata
        explanations["ticker"] = request.ticker
        explanations["model_type"] = request.model_type
        explanations["explanation_date"] = datetime.now().isoformat()
        explanations["data_dates"] = X.index.strftime("%Y-%m-%d").tolist()
        
        return explanations
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating explanation: {str(e)}")


@app.post("/global-importance")
async def get_global_importance(request: GlobalImportanceRequest):
    """
    Get global feature importance using SHAP values.
    
    Shows which features are most important across all predictions.
    """
    try:
        from shap_explainer.shap_explainer import SHAPExplainer
        
        # Load model
        model, model_wrapper = load_model(request.model_type, request.ticker)
        
        # Load and prepare data
        data = load_data_for_explanation(request.ticker, model_wrapper, days=365*2)
        
        # Get feature names
        feature_names = data.columns.tolist()
        
        # Scale data if scaler exists
        if hasattr(model_wrapper, 'feature_scaler') and model_wrapper.feature_scaler is not None:
            X_scaled = model_wrapper.feature_scaler.transform(data)
        else:
            X_scaled = data.values
        
        # Create explainer and get global importance
        explainer = SHAPExplainer(model, model_type=request.model_type)
        importance = explainer.get_global_importance(
            X_scaled, 
            feature_names=feature_names,
            max_samples=request.max_samples
        )
        
        # Add metadata
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
        raise HTTPException(status_code=500, detail=f"Error calculating importance: {str(e)}")


@app.get("/summary-plot")
async def get_summary_plot(
    ticker: str = Query("NU", description="Stock ticker"),
    model_type: str = Query("xgboost", description="Model type: xgboost or random_forest"),
    plot_type: str = Query("bar", description="Plot type: bar or dot"),
    max_features: int = Query(15, description="Maximum features to display")
):
    """
    Generate SHAP summary plot as base64 image.
    """
    try:
        from shap_explainer.shap_explainer import SHAPExplainer
        
        # Load model
        model, model_wrapper = load_model(model_type, ticker)
        
        # Load and prepare data
        data = load_data_for_explanation(ticker, model_wrapper)
        
        # Get feature names
        feature_names = data.columns.tolist()
        
        # Scale data if scaler exists
        if hasattr(model_wrapper, 'feature_scaler') and model_wrapper.feature_scaler is not None:
            X_scaled = model_wrapper.feature_scaler.transform(data)
        else:
            X_scaled = data.values
        
        # Create explainer and generate plot
        explainer = SHAPExplainer(model, model_type=model_type)
        image_base64 = explainer.generate_summary_plot(
            X_scaled,
            feature_names=feature_names,
            max_display=max_features,
            plot_type=plot_type
        )
        
        return {
            "ticker": ticker,
            "model_type": model_type,
            "plot_type": plot_type,
            "image_base64": image_base64,
            "generated_at": datetime.now().isoformat()
        }
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating plot: {str(e)}")


@app.get("/waterfall-plot")
async def get_waterfall_plot(
    ticker: str = Query("NU", description="Stock ticker"),
    model_type: str = Query("xgboost", description="Model type: xgboost or random_forest"),
    prediction_index: int = Query(0, description="Index of prediction to explain (0 = most recent)"),
    max_features: int = Query(10, description="Maximum features to display")
):
    """
    Generate SHAP waterfall plot for a single prediction.
    
    Shows how each feature contributes to moving the prediction from the base value.
    """
    try:
        from shap_explainer.shap_explainer import SHAPExplainer
        
        # Load model
        model, model_wrapper = load_model(model_type, ticker)
        
        # Load and prepare data
        data = load_data_for_explanation(ticker, model_wrapper)
        
        # Get feature names
        feature_names = data.columns.tolist()
        
        # Get last N rows
        X = data.tail(10)
        
        # Scale data if scaler exists
        if hasattr(model_wrapper, 'feature_scaler') and model_wrapper.feature_scaler is not None:
            X_scaled = model_wrapper.feature_scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Create explainer and generate plot
        explainer = SHAPExplainer(model, model_type=model_type)
        image_base64 = explainer.generate_waterfall_plot(
            X_scaled,
            prediction_index=prediction_index,
            feature_names=feature_names,
            max_display=max_features
        )
        
        return {
            "ticker": ticker,
            "model_type": model_type,
            "prediction_index": prediction_index,
            "prediction_date": X.index[prediction_index].strftime("%Y-%m-%d") if prediction_index < len(X) else None,
            "image_base64": image_base64,
            "generated_at": datetime.now().isoformat()
        }
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating plot: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "shap-explainer"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
