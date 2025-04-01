from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import joblib
import glob
from datetime import datetime, timedelta

# Importar módulos personalizados
from services.model_rf.rf_model2 import TimeSeriesRandomForestModel
from services.model_rf.train import train_ts_model
from services.model_rf.forecast import forecast_future_prices
from utils.import_data import load_data

app = FastAPI(title="Random Forest Time Series Model Service", version="1.0.0")

# Definir el modelo de datos para la solicitud de entrenamiento
class TrainRequest(BaseModel):
    ticket: str = "NU"
    start_date: str = "2020-12-10"
    end_date: str = datetime.now().strftime("%Y-%m-%d")
    n_lags: int = 10
    target_col: str = "Close"
    train_size: float = 0.8
    save_model_path: str = None

# Rutas para el almacenamiento de modelos
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Diccionario global para almacenar los modelos entrenados
loaded_models = {}

def get_default_model_path(ticket):
    """Genera la ruta predeterminada para guardar un modelo entrenado"""
    return os.path.join(MODEL_DIR, f"rf_model_{ticket}.joblib")

def get_generic_model_path():
    """ Obtiene la ruta del modelo genérico entrenado previamente """
    return os.path.join(MODEL_DIR, "rf_model.joblib")

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
    avaliable_models = glob.glob(os.path.join(MODEL_DIR, "rf_model_*.joblib"))
    if avaliable_models:
        return avaliable_models[0]
    return None

def load_model(model_path):
    """Recupera un modelo de la memoria caché o lo carga desde el disco"""
    if model_path in loaded_models:
        return loaded_models[model_path]
    
    try:
        model = TimeSeriesRandomForestModel.load_model(model_path)
        loaded_models[model_path] = model
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

def load_stock_data(ticket, start_date, end_date):
    """Carga datos históricos de acciones"""
    try:
        data = load_data(ticker=ticket, start_date=start_date, end_date=end_date)
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for ticket {ticket}")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading data: {str(e)}")

@app.get("/")
async def read_root():
    return {"message": "Random Forest Time Series Model Service"}

@app.post("/train")
async def train_model(request: TrainRequest):
    """
    Endpoint para entrenar un modelo de Random Forest para series temporales.
    
    Si save_model_path no se proporciona, el modelo se guarda en la ruta predeterminada.
    """
    try:
        # Cargar datos históricos
        data = load_stock_data(request.ticket, request.start_date, request.end_date)
        
        # Establecer ruta para guardar el modelo
        save_path = request.save_model_path
        if save_path is None or save_path == "":
            # Si no se proporciona una ruta, usar la ruta predeterminada
            save_path = get_default_model_path(request.ticket)
        
        # Entrenar modelo
        model = train_ts_model(
            data=data,
            n_lags=request.n_lags,
            target_col=request.target_col,
            train_size=request.train_size,
            save_model_path=save_path
        )

        # Agregar el modelo a la memoria caché
        loaded_models[save_path] = model
        
        # Devolver métricas y mejores parámetros
        return {
            "status": "success",
            "message": f"Model trained successfully for {request.ticket}",
            "metrics": model.metrics,
            "best_params": model.best_params_,
            "model_path": save_path
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

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
        # 1. Buscar el modelo para el ticket
        model_path = find_model_for_ticket(ticket)
        if model_path is None:
            raise HTTPException(
                status_code=404, 
                detail=f"No trained model found for {ticket}. Train a model first.")
        
        # 2. Cargar el modelo
        try:
            model = TimeSeriesRandomForestModel.load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
            print(f"Model type: {type(model)}")
            
            # Verificar que el modelo tiene los atributos necesarios
            if not hasattr(model, 'n_lags'):
                raise ValueError("Model missing n_lags attribute")
            if not hasattr(model, 'best_pipeline_'):
                raise ValueError("Model missing best_pipeline_ attribute")
                
        except Exception as model_error:
            raise HTTPException(
                status_code=500,
                detail=f"Error loading model: {str(model_error)}"
            )
        
        # 3. Cargar datos con un período más largo para asegurar suficientes datos
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
            if len(data) <= model.n_lags * 2:  # Multiplicamos por 2 para tener un margen
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
        
        # 4. Preparar datos y hacer predicción
        try:
            # Usamos directamente la función forecast_future_prices sin modificarla
            forecast = forecast_future_prices(
                model=model,
                data=data,
                forecast_horizon=forecast_horizon,
                target_col=target_col
            )
            
            # 5. Preparar respuesta
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
            # Capturar detalles específicos del error para depuración
            import traceback
            error_details = traceback.format_exc()
            print(f"Forecast error details: {error_details}")
            
            # Proporcionar un mensaje de error más descriptivo
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
        # Capturar cualquier otro error no manejado
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
        models = glob.glob(os.path.join(MODEL_DIR, "rf_model_*.joblib"))
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

@app.get("/health")
async def health_check():
    """Endpoint de verificación de salud para el microservicio."""
    return {"status": "Ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)  # El puerto 8001 coincide con el microservicio de RF en el API Gateway