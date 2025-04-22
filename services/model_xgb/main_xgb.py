'''

This app.py file powers the RESTful API for interacting with the XGBoostModel. It supports endpoints for:

Training a model (/train)

Evaluating performance (/evaluate)

Making future predictions (/predict)

Listing available models (/models)

The app uses FastAPI and integrates with utils/ and services/ modules.


API Endpoints

GET /

Basic welcome message.

GET /train

Train a new XGBoost model on locally loaded stock data (default params). Saves model + metrics.

Uses:

load_data()

feature_engineering()

split_data()

scale_data()

XGBoostModel().fit()

Output: {
  "message": "Model trained and saved successfully.",
  "evaluation_results": {"mse": ..., "mae": ...}
}
GET /predict

Predict future values using latest available model.

Query parameters:

ticket: stock ticker

forecast_horizon: number of days to forecast

target_col: column name (usually 'Close')

Returns JSON array of future values + model metadata used.

GET /models

List trained models in /models dir with metadata, sizes, paths.

GET /evaluate

Evaluates latest model on holdout test data with:

Scaled metrics

Inverse-transformed original-scale metrics

Returns a snapshot:
{
  "evaluation_results_scaled": {...},
  "evaluation_results_original": {...},
  "sample_predictions": {
    "predicted": [...],
    "actual": [...]
  }
}

'''
from pathlib import Path
import sys
import os
from datetime import datetime, timedelta
import glob

current_dir = Path(__file__).parent
project_dir = current_dir.parent.parent  # Navigate up to the project root
sys.path.append(str(project_dir))
from services.model_xgb.forecast import forecast_future_prices
from fastapi import FastAPI, HTTPException, Query
import pandas as pd
from pydantic import BaseModel

import joblib
import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
from contextlib import asynccontextmanager
from services.model_xgb.xgb_model import XGBoostModel  # Import from the same directory
from utils.import_data import load_data
from utils.preprocessing import feature_engineering, split_data, scale_data
import traceback
#
app = FastAPI(
    title="XGBoost Model API",
    description="API for XGBoost Regression model",
    version="1.0.0",
)

# Define the model path
MODEL_DIR = "services/model_xgb/models"
os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, "model_xgb.joblib")
json_path = os.path.join(MODEL_DIR, "model_metrics.json")



class TrainRequest(BaseModel):
    ticket: str = "NU"
    start_date: str = "2020-12-10"
    end_date: str = datetime.now().strftime("%Y-%m-%d")
    n_lags: int = 10
    target_col: str = "Close"
    train_size: float = 0.8
    save_model_path: str = None



# Diccionario global para almacenar los modelos entrenados
loaded_models = {}

def get_default_model_path(ticket):
    """Genera la ruta predeterminada para guardar un modelo entrenado"""
    return os.path.join(MODEL_DIR, f"xgb_model_{ticket}.joblib") # CAMBIAR

def get_generic_model_path():
    """ Obtiene la ruta del modelo genérico entrenado previamente """
    return os.path.join(MODEL_DIR, "xgb_model.joblib")


def find_model_for_ticket(ticket):
    """
    Busca el modelo entrenado para un ticket específico
    Retorna la ruta del modelo si se encuentra, de lo contrario None
    """
    specific_model_path = get_default_model_path(ticket)
    if os.path.exists(specific_model_path):
        return specific_model_path
    
    # Si no hay modelo específico, busca un modelo genérico
    generic_model_path = get_generic_model_path()
    if os.path.exists(generic_model_path):
        return generic_model_path
    
    # Busca cualquier modelo en el directorio de modelos como último recurso
    avaliable_models = glob.glob(os.path.join(MODEL_DIR, "xgb_model_*.joblib"))
    if avaliable_models:
        return avaliable_models[0]
    return None

def load(model_path):
    """Recupera un modelo de la memoria caché o lo carga desde el disco"""
    if model_path in loaded_models:
        return loaded_models[model_path]
    
    try:
        model = XGBoostModel.load(model_path)
        loaded_models[model_path] = model
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

def load_stock_data(ticket, start_date, end_date):
    """Carga los datos de stock para un ticket específico entre las fechas dadas"""
    try:
        data = load_data(ticket, start_date, end_date)
        if data.empty:
            raise HTTPException(status_code=404, detail="No data found for the given ticket and date range.")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")


@app.get("/")
def read_root():
    return {"message": "Welcome to the XGBoost Model API"}


@app.get("/train") 
def train_model():
    """_
    Endpoint to train the model
    """
    global model, feature_scaler, target_scaler
    try:
        print("Manual model training started.")
        data = load_data()
        print("Data loaded successfully. Shape:", data.shape)

        processed_data = feature_engineering(data)
        
        print("Feature engineering completed. Shape:", processed_data.shape)

        # Split data
        X_train, X_test, y_train, y_test = split_data(processed_data, train_size=0.8)
        print("Data split completed. Training data shape:", X_train.shape)
        
        # Escalar los datos
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, feature_scaler, target_scaler = scale_data(
            X_train, X_test, 
            y_train.values.reshape(-1, 1), 
            y_test.values.reshape(-1, 1)
        )
        print("Data scaling completed.")

        # Train model
        model = XGBoostModel()
        model.fit(X_train_scaled, y_train_scaled)
        print("Model training completed.")

        # Asignar los escaladores al modelo
        model.feature_scaler = feature_scaler
        model.target_scaler = target_scaler

        
        # Save model
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_file_path = os.path.join(MODEL_DIR, "xgb_model.joblib")
        model.save(model_file_path)
        print(f"Model saved to {MODEL_DIR}")
        
        # Evaluate model
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_test_original = target_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
        
        evaluation_results = model.evaluate(X_test_scaled, y_test_scaled)
        print("Model evaluation results:", evaluation_results) 
        
        # Save metadata
        metadata = {
            'model_name': 'XGBoost',
            'best_params': model.best_params_,
            'evaluation': evaluation_results,
            'timestamp': pd.Timestamp.now().isoformat() # se puede añadir ya el scaler
        }
        
        # Convert any numpy types to Python native types for JSON serialization
        metadata_serializable = {}
        for k, v in metadata.items():
            if isinstance(v, dict):
                metadata_serializable[k] = {sub_k: float(sub_v) if isinstance(sub_v, np.float64) else sub_v 
                                           for sub_k, sub_v in v.items()}
            else:
                metadata_serializable[k] = float(v) if isinstance(v, np.float64) else v
        
        with open(json_path, 'w') as f:
            import json
            json.dump(metadata_serializable, f, indent=4)
            
        return {
            "message": "Model trained and saved successfully.",
            "evaluation_results": evaluation_results
        }
    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"Error during training: {traceback_str}")
        raise HTTPException(status_code=500, detail=f"Error during model training: {str(e)}")


@app.get("/predict")
async def predict(
    ticket: str = Query("NU", description="Ticker of the stock to predict"),
    forecast_horizon: int = Query(10, description="Forecast horizon in days"),
    target_col: str = Query("Close", description="Target column for prediction")
):
    """
    Endpoint para hacer predicciones usando el modelo entrenado.
    
    Intenta utilizar un modelo específico para el ticket proporcionado. Si no se encuentra, busca un modelo genérico.
    Ejemplo de uso:
    GET /predict?ticket=NU&forecast_horizon=10&target_col=Close
    """
    try:
        # Verificar que el ticket no esté vacío
        model_path = find_model_for_ticket(ticket)
        if model_path is None:
            raise HTTPException(
                status_code=404, 
                detail=f"No trained model found for {ticket}. Train a model first.")
        
        try:
            model = XGBoostModel.load(model_path)
            print(f"Model loaded successfully from {model_path}")
            print(f"Model type: {type(model)}")
            
            # Verificar que el modelo tiene los atributos necesarios
            if not hasattr(model, 'n_lags'):
                raise ValueError("Model missing n_lags attribute")

                
        except Exception as model_error:
            raise HTTPException(
                status_code=500,
                detail=f"Error loading model: {str(model_error)}"
            )
        
        try:
            end_date = datetime.now()
            # Usar un período más largo para asegurar suficientes datos
            start_date = end_date - timedelta(days=365*3)  # Tres años de datos históricos
            data = load_stock_data(ticket, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            
            print(f"Data loaded: {len(data)} rows, columns: {data.columns.tolist()}")
            
            # Verificar que tenemos datos suficientes
            if data.empty:
                raise HTTPException(
                    status_code=404,
                    detail=f"No data available for ticker {ticket}"
                )
                
            if target_col not in data.columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Target column '{target_col}' not found in data. Available columns: {data.columns.tolist()}"
                )
                
            # Verificar que hay suficientes filas para los rezagos
            if len(data) <= model.n_lags * 2:  
                raise HTTPException(
                    status_code=400,
                    detail=f"Not enough historical data for prediction. Need at least {model.n_lags * 2} rows, but got {len(data)}."
                )
                
        except HTTPException:
            raise
        except Exception as data_error:
            raise HTTPException(
                status_code=500,
                detail=f"Error loading stock data: {str(data_error)}"
            )
        
        try:
            forecast = forecast_future_prices(
                model=model,
                data=data,
                forecast_horizon=forecast_horizon,
                target_col=target_col
            )
            print('Predictions:', forecast)
            last_date = data.index[-1]
            forecast_dates = [(last_date + timedelta(days=i+1)).strftime("%Y-%m-%d") 
                            for i in range(forecast_horizon)]
            
            predictions = [{"date": date, "prediction": float(pred)} 
                        for date, pred in zip(forecast_dates, forecast)]
            
            model_info = "Modelo específico para el ticker" if ticket in model_path else "Modelo genérico"
            
            return {
                "status": "success",
                "ticker": ticket,
                "target_column": target_col,
                "forecast_horizon": forecast_horizon,
                "predictions": predictions,
                "last_actual_date": last_date.strftime("%Y-%m-%d"),
                "last_actual_value": float(data[target_col].iloc[-1]),
                "model_used": os.path.basename(model_path),
                "model_info": model_info
            }
            
        except Exception as forecast_error:
            import traceback
            error_details = traceback.format_exc()
            print(f"Forecast error details: {error_details}")
            
            if "Found array with 0 sample(s)" in str(forecast_error):
                raise HTTPException(
                    status_code=500,
                    detail="Error during data scaling: Empty array encountered. This typically happens when data preparation removes all rows. Check if your historical data matches the format expected by the model."
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error making prediction: {str(forecast_error)}"
                )
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/models")
async def list_models():
    """
    Endpoint para listar los modelos entrenados disponibles.
    
    Devuelve una lista de modelos con su ruta y tipo (específico o genérico).
    """
    try:
        models = glob.glob(os.path.join(MODEL_DIR, "xgb_model*.joblib"))
        model_info = []

        for model_path in models:
            model_name = os.path.basename(model_path)
            metadata = model_path.replace(',joblib', "_metadata.json")
            metadata = None

            if os.path.exists(metadata):
                try:
                    import json
                    with open(metadata, 'r') as f:
                        metadata = json.load(f)
                except:
                    metadata = {"error": "Cloud not load metadata"}

            model_info.append({
                "name": model_name,
                "path": model_path,
                "metadata": metadata,
                "size_mb": round(os.path.getsize(model_path) / (1024 * 1024), 2),  # Tamaño en MB
            })
            return {
                "total_models": len(model_info),
                "models": model_info
            }

    except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")
@app.get("/evaluate") 
def evaluate_model():
    """
    Endpoint to evaluate the model on test data
    """
    try:
        # Check if model exists
        model_path = get_generic_model_path()
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model not found. Please train the model first.")
        
        # Load model
        model = XGBoostModel.load(model_path)
        
        # Load data
        data = load_data()
        processed_data = feature_engineering(data)
        X_train, X_test, y_train, y_test = split_data(processed_data, train_size=0.8)
        
        # Create new scalers for this evaluation
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, feature_scaler, target_scaler = scale_data(
            X_train, X_test, 
            y_train.values.reshape(-1, 1), 
            y_test.values.reshape(-1, 1)
        )
        
        # Make predictions using scaled data
        y_pred_scaled = model.predict(X_test_scaled)
        
        # Inverse transform predictions and actual values
        y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_test_original = target_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
        
        # Evaluate using both scaled and original data
        evaluation_results = model.evaluate(X_test_scaled, y_test_scaled)
        original_metrics = model.evaluate(X_test, y_test.values)
        
        return {
            "evaluation_results_scaled": evaluation_results,
            "evaluation_results_original": original_metrics,
            "sample_predictions": {
                "predicted": y_pred[:5].tolist(),
                "actual": y_test_original[:5].tolist()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during model evaluation: {str(e)}")
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


#script to run the server
'''
.venv\Scripts\activate 
python -m services.model_xgb.app

'''