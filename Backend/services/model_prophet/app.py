from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import os
import glob
import json
from datetime import datetime, timedelta
import sys

# --- Ajuste del sys.path para importaciones ---
try:
    # Intenta importar asumiendo que el PYTHONPATH está correctamente configurado
    from utils.import_data import load_data
    from services.model_prophet.prophet_model import ProphetModel
    from services.model_prophet.train_prophet import train_model as train_prophet_pipeline
    from services.model_prophet.forecast import forecast_future_prices_prophet
except ImportError:
    # Si falla, ajusta el path manualmente y reintenta
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
    if backend_dir not in sys.path:
        sys.path.append(backend_dir)
    from utils.import_data import load_data
    from services.model_prophet.prophet_model import ProphetModel
    from services.model_prophet.train_prophet import train_model as train_prophet_pipeline
    from services.model_prophet.forecast import forecast_future_prices_prophet

# --- Configuración de la App FastAPI ---
app = FastAPI(
    title="Prophet Time Series Model Service",
    version="1.0.3",
    description="Un servicio para entrenar y realizar predicciones con modelos Prophet para series temporales."
)

# --- Modelos Pydantic para Requests ---
class TrainRequestProphet(BaseModel):
    ticker: str = "NU"
    start_date: str = "2020-01-01"
    end_date: str = datetime.now().strftime('%Y-%m-%d')
    target_col: str = "Close"
    train_size: float = 0.8
    # El path se genera automáticamente, pero se puede sobreescribir
    model_path_prefix: Optional[str] = None

# --- Gestión de Modelos y Caching ---
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)
loaded_prophet_models_cache: Dict[str, ProphetModel] = {}

def get_default_prophet_model_path_prefix(ticker: str) -> str:
    """Genera el prefijo de ruta por defecto para un modelo de un ticker específico."""
    return os.path.join(MODEL_DIR, f"prophet_model_{ticker.upper()}")

def find_prophet_model_path_prefix(ticker: str) -> Optional[str]:
    """Encuentra el prefijo de un modelo para un ticker, con fallback al primero disponible."""
    specific_prefix = get_default_prophet_model_path_prefix(ticker)
    # Comprueba si existe el archivo de metadatos para el modelo específico
    if os.path.exists(f"{specific_prefix}_metadata.json"):
        return specific_prefix
    
    # Si no, busca cualquier otro modelo Prophet como fallback
    metadata_files = glob.glob(os.path.join(MODEL_DIR, "prophet_model_*_metadata.json"))
    if metadata_files:
        first_metadata_file = sorted(metadata_files)[0]
        prefix = first_metadata_file.replace("_metadata.json", "")
        print(f"ADVERTENCIA: No se encontró modelo Prophet para {ticker}. Usando el primero disponible: {os.path.basename(prefix)}")
        return prefix
    return None

def load_prophet_model_from_prefix(prefix: str) -> ProphetModel:
    """Carga un modelo Prophet desde un prefijo, usando caché para eficiencia."""
    if prefix in loaded_prophet_models_cache:
        print(f"Retornando modelo Prophet desde caché para el prefijo: {os.path.basename(prefix)}")
        return loaded_prophet_models_cache[prefix]
    
    try:
        print(f"Cargando modelo Prophet desde el prefijo: {os.path.basename(prefix)}")
        model = ProphetModel.load_model(model_path_prefix=prefix)
        loaded_prophet_models_cache[prefix] = model
        return model
    except FileNotFoundError as fnf_error:
        raise HTTPException(status_code=404, detail=f"Archivos de modelo no encontrados para el prefijo {os.path.basename(prefix)}: {fnf_error}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error cargando modelo Prophet desde {os.path.basename(prefix)}: {e}")

def load_stock_data_helper(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Función de ayuda para cargar datos de acciones, con manejo de errores."""
    try:
        data_df = load_data(ticker=ticker, start_date=start_date, end_date=end_date)
        if data_df.empty:
            raise HTTPException(status_code=404, detail=f"No se encontraron datos para el ticker {ticker} en el rango {start_date} a {end_date}.")
        return data_df
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=f"Error descargando o procesando datos para {ticker}: {e}")

# --- Endpoints de la API ---

@app.get("/", tags=["General"])
async def read_root_prophet():
    return {"message": "Servicio de Modelos de Series de Tiempo Prophet"}


@app.post("/train", tags=["Training & Management"])
async def train_model_endpoint(request: TrainRequestProphet):
    """
    Entrena un modelo de series de tiempo Prophet según los parámetros proporcionados.
    """
    try:
        print(f"Solicitud de entrenamiento Prophet recibida para el ticker: {request.ticker}")
        data_df = load_stock_data_helper(request.ticker, request.start_date, request.end_date)
        
        # --- LÍNEA CORREGIDA ---
        # El prefijo ahora se genera aquí y se pasa directamente, evitando la doble concatenación.
        save_prefix = request.model_path_prefix or get_default_prophet_model_path_prefix(request.ticker)
        
        print(f"Iniciando entrenamiento del modelo Prophet. Se guardará con prefijo: {os.path.basename(save_prefix)}")
        
        # El pipeline de entrenamiento ahora recibe el prefijo final y no lo modifica.
        trained_model, metrics = train_prophet_pipeline(
            ticker=request.ticker,
            start_date=request.start_date,
            end_date=request.end_date,
            target_col=request.target_col,
            train_size=request.train_size,
            model_path_prefix=save_prefix # Pasamos el prefijo ya completo
        )

        # Cache the newly trained model
        loaded_prophet_models_cache[save_prefix] = trained_model

        return {
            "status": "success",
            "message": f"Modelo Prophet entrenado exitosamente para {request.ticker}",
            "model_type": "Prophet",
            "model_path_prefix": os.path.basename(save_prefix),
            "metrics": metrics
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error entrenando modelo Prophet: {str(e)}")

@app.get("/predict", tags=["Prediction"])
async def predict_endpoint(
    ticker: str = Query("NU", description="Ticker de la acción a predecir"),
    forecast_horizon: int = Query(15, description="Horizonte de pronóstico en días"),
    target_col: str = Query("Close", description="Columna objetivo para la predicción"),
    history_days: int = Query(90, description="Número de días de datos históricos a devolver")
):
    """
    Realiza una predicción utilizando un modelo Prophet entrenado.
    Carga el historial de datos completo que necesita el modelo para asegurar
    la correcta alineación de fechas y el cálculo de regresores.
    """
    try:
        print(f"Solicitud de predicción Prophet recibida para el ticker: {ticker}")
        
        # 1. Encontrar el prefijo del modelo para el ticker solicitado.
        model_prefix = find_prophet_model_path_prefix(ticker)
        if not model_prefix:
            raise HTTPException(status_code=404, detail=f"No se encontró un modelo Prophet entrenado para {ticker}.")

        # 2. Cargar el modelo desde el archivo (o desde la caché si ya está cargado).
        model = load_prophet_model_from_prefix(model_prefix)
        
        # --- LÓGICA DE CARGA DE DATOS SINCRONIZADA ---
        # 3. Obtener el rango de fechas directamente del historial del modelo cargado.
        #    Esto es CRUCIAL para evitar desajustes de fechas entre el modelo y los datos.
        start_date_for_prediction = model.model.history['ds'].min()
        end_date_for_historical_load = model.model.history['ds'].max()
        
        print(f"Rango de datos requerido por el modelo: {start_date_for_prediction.strftime('%Y-%m-%d')} a {end_date_for_historical_load.strftime('%Y-%m-%d')}")

        # 4. Cargar el historial completo que el modelo necesita para sus cálculos.
        print(f"Cargando datos históricos para predicción...")
        historical_data_df = load_stock_data_helper(
            ticker,
            start_date_for_prediction.strftime("%Y-%m-%d"),
            end_date_for_historical_load.strftime("%Y-%m-%d")
        )
        
        # 5. Llamar a la función de pronóstico con los datos correctamente alineados.
        print("Realizando pronóstico Prophet...")
        forecast_values, lower_bounds, upper_bounds = forecast_future_prices_prophet(
            model=model,
            data=historical_data_df.copy(),
            forecast_horizon=forecast_horizon,
            target_col=target_col
        )

        # 6. Preparar la respuesta JSON.
        last_actual_date_in_data = historical_data_df.index.max()
        # Se generan las fechas para el pronóstico usando días hábiles ('B').
        forecast_dates = pd.date_range(
            start=last_actual_date_in_data + pd.tseries.offsets.BDay(1),
            periods=len(forecast_values), # Se usa la longitud real del pronóstico.
            freq='B'
        ).strftime('%Y-%m-%d').tolist()
        
        predictions_list = [{
            "date": forecast_dates[i],
            "prediction": float(forecast_values[i]),
            "lower_bound": float(lower_bounds[i]),
            "upper_bound": float(upper_bounds[i])
        } for i in range(len(forecast_dates))]

        # Se seleccionan los últimos N días de historial para incluir en la respuesta.
        historical_data_to_return = historical_data_df.iloc[-history_days:]

        return {
            "status": "success", "ticker": ticker, "model_type": "Prophet",
            "target_column": target_col, "forecast_horizon": forecast_horizon,
            "historical_dates": historical_data_to_return.index.strftime('%Y-%m-%d').tolist(),
            "historical_values": [val if not np.isnan(val) else None for val in historical_data_to_return[target_col].tolist()],
            "predictions": predictions_list,
            "last_actual_date": last_actual_date_in_data.strftime("%Y-%m-%d"),
            "last_actual_value": float(historical_data_df[target_col].iloc[-1]),
            "model_used_prefix": os.path.basename(model_prefix)
        }
        
    except HTTPException:
        # Re-lanzar excepciones HTTP para que FastAPI las maneje.
        raise
    except Exception as e:
        # Capturar cualquier otro error y devolver una respuesta de error 500.
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error en predicción Prophet: {str(e)}")

@app.get("/models", tags=["Training & Management"])
async def list_prophet_models():
    """Lista todos los modelos Prophet entrenados disponibles."""
    try:
        metadata_files = glob.glob(os.path.join(MODEL_DIR, "prophet_model_*_metadata.json"))
        models_info = []
        for meta_file_path in metadata_files:
            model_prefix = meta_file_path.replace("_metadata.json", "")
            try:
                with open(meta_file_path, 'r') as f:
                    metadata = json.load(f)
                models_info.append({
                    "name": os.path.basename(model_prefix),
                    "metadata": metadata
                })
            except Exception as e:
                print(f"Error al leer metadatos de {meta_file_path}: {e}")
        return {"total_models": len(models_info), "models": models_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listando modelos Prophet: {str(e)}")

@app.get("/health", tags=["General"])
async def health_check_prophet():
    return {"status": "Ok", "service": "Prophet Time Series Model Service"}

if __name__ == "__main__":
    import uvicorn
    # El puerto 8004 es el asignado para Prophet en el docker-compose y api-gateway
    uvicorn.run(app, host="0.0.0.0", port=8004)
