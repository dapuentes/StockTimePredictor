# Backend/services/model_xgb/main.py

from fastapi import FastAPI, HTTPException, Query, Depends # Asegúrate de tener Depends si usas GET con Pydantic model
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import os
import glob
import json
from datetime import datetime, timedelta

# Import modules specific to XGBoost service (relative imports)
from .xgb_model import TimeSeriesXGBoostModel # Usa la versión de xgb_model_py_v2
from .train_xgb import train_xgb_model
from .forecast import forecast_future_prices_xgb

try:
    from utils.import_data import load_data
except ImportError:
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..')) 
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from utils.import_data import load_data


app = FastAPI(
    title="XGBoost Time Series Model Service",
    version="1.0.2", # Incremented version
    description="A service for training and making predictions with XGBoost models for time series."
)

class TrainRequestXGB(BaseModel):
    ticket: str = "NU" 
    start_date: Optional[str] = None 
    end_date: Optional[str] = None
    n_lags: int = 10
    target_col: str = "Close" 
    train_size_ratio: float = 0.7
    save_model_path_prefix: Optional[str] = None

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)
loaded_xgb_models_cache: Dict[str, TimeSeriesXGBoostModel] = {}

def get_default_xgb_model_path_prefix(ticket: str) -> str:
    return os.path.join(MODEL_DIR, f"xgb_model_{ticket.upper()}")

def find_xgb_model_path_prefix(ticket: str) -> Optional[str]:
    specific_prefix = get_default_xgb_model_path_prefix(ticket)
    if os.path.exists(f"{specific_prefix}_metadata.json"):
        return specific_prefix
    metadata_files = glob.glob(os.path.join(MODEL_DIR, "xgb_model_*_metadata.json"))
    if metadata_files:
        first_metadata_file = metadata_files[0]
        prefix = first_metadata_file.replace("_metadata.json", "")
        print(f"Advertencia: No se encontró modelo XGBoost específico para {ticket}. Usando el primero disponible: {os.path.basename(prefix)}")
        return prefix
    return None

def load_xgb_model_from_prefix(prefix: str) -> TimeSeriesXGBoostModel:
    if prefix in loaded_xgb_models_cache:
        print(f"Retornando modelo XGBoost desde caché para el prefijo: {prefix}")
        return loaded_xgb_models_cache[prefix]
    try:
        print(f"Cargando modelo XGBoost desde el prefijo: {prefix}")
        model = TimeSeriesXGBoostModel.load_model(model_path_prefix=prefix)
        loaded_xgb_models_cache[prefix] = model
        return model
    except FileNotFoundError as fnf_error:
        print(f"Error de archivo no encontrado al cargar modelo XGBoost: {fnf_error}")
        raise HTTPException(status_code=404, detail=f"Archivos de modelo no encontrados para el prefijo {prefix}: {str(fnf_error)}")
    except Exception as e:
        print(f"Error detallado al cargar modelo XGBoost desde {prefix}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error cargando modelo XGBoost desde {prefix}: {str(e)}")

def load_stock_data_helper(ticket: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        data_df = load_data(ticker=ticket, start_date=start_date, end_date=end_date)
        if data_df.empty:
            raise HTTPException(status_code=404,
                                detail=f"No se encontraron datos para el ticker {ticket} en el rango {start_date} a {end_date}.")
        return data_df
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        print(f"Error al cargar datos de acciones para {ticket}: {e}")
        raise HTTPException(status_code=500, detail=f"Error descargando o procesando datos para {ticket}: {str(e)}")

@app.get("/", tags=["General"])
async def read_root_xgb():
    return {"message": "Servicio de Modelos de Series de Tiempo XGBoost"}

# CORRECCIÓN AQUÍ: Cambiar a @app.post y 'request' vendrá del cuerpo de la petición
@app.post("/train", tags=["XGBoost Training & Management"])
async def train_model_endpoint(request: TrainRequestXGB): # 'request' ahora es el cuerpo JSON
    """
    Trains an XGBoost time series model based on the provided request parameters (sent in JSON body).
    """
    try:
        print(f"Solicitud de entrenamiento XGBoost recibida para el ticker: {request.ticket}")
        
        # Determinar start_date y end_date
        # Prioridad: request.start_date/end_date, luego request.training_period, luego defaults.
        start_date_to_load = request.start_date
        end_date_to_load = request.end_date

        data_df = load_stock_data_helper(request.ticket, start_date_to_load, end_date_to_load)

        save_prefix = request.save_model_path_prefix or get_default_xgb_model_path_prefix(request.ticket)
        
        print(f"Iniciando entrenamiento del modelo XGBoost. Se guardará con prefijo: {save_prefix}")
        
        trained_model_obj, feature_names, residuals, residual_dates, acf_vals, pacf_vals, confint_acf, confint_pacf = train_xgb_model(
            data=data_df,
            target_col=request.target_col,
            n_lags=request.n_lags,
            train_size_ratio=request.train_size_ratio,
            save_model_path_prefix=save_prefix
        )

        loaded_xgb_models_cache[save_prefix] = trained_model_obj

        response = {
            "status": "success",
            "message": f"Modelo XGBoost entrenado exitosamente para {request.ticket}",
            "model_type": "XGBoost",
            "model_path_prefix": os.path.basename(save_prefix),
            "metrics": trained_model_obj.metrics if hasattr(trained_model_obj, 'metrics') else "Métricas no disponibles.",
            "feature_names_used": feature_names,
            "residuals_length": len(residuals) if residuals is not None else 0,
            "acf_values_length": len(acf_vals) if acf_vals is not None else 0,
            "pacf_values_length": len(pacf_vals) if pacf_vals is not None else 0
        }
        if hasattr(trained_model_obj, 'best_params_') and trained_model_obj.best_params_:
            serializable_best_params = {}
            for k, v in trained_model_obj.best_params_.items():
                if isinstance(v, (np.ndarray, list)):
                    serializable_best_params[k] = [item.item() if hasattr(item, 'item') else item for item in v]
                elif hasattr(v, 'item'): 
                    serializable_best_params[k] = v.item()
                else:
                    serializable_best_params[k] = v
            response["best_hyperparameters"] = serializable_best_params
        
        return response

    except HTTPException:
        raise 
    except Exception as e:
        print(f"Error detallado durante el entrenamiento XGBoost: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error entrenando modelo XGBoost: {str(e)}")


@app.get("/predict", tags=["XGBoost Prediction"])
async def predict_endpoint(
    ticket: str = Query("NU", description="Ticker de la acción a predecir"),
    forecast_horizon: int = Query(10, description="Horizonte de pronóstico en días"),
    target_col: str = Query("Close", description="Columna objetivo para la predicción"),
    history_days: int = Query(252, description="Número de días hábiles históricos a devolver en la respuesta (aprox. 1 año)")
):
    try:
        print(f"Solicitud de predicción XGBoost recibida para el ticker: {ticket}")
        model_prefix = find_xgb_model_path_prefix(ticket)
        if not model_prefix:
            raise HTTPException(status_code=404,
                                detail=f"No se encontró un modelo XGBoost entrenado para {ticket} o un modelo de fallback.")

        print(f"Cargando modelo XGBoost desde el prefijo: {model_prefix}")
        model = load_xgb_model_from_prefix(model_prefix)
        
        # Lógica de fechas para predicción mejorada
        # Usar la fecha de fin de entrenamiento del modelo si está disponible, si no, hoy.
        end_date_for_historical_load = datetime.now()
        metadata_path_for_predict = f"{model_prefix}_metadata.json"
        if os.path.exists(metadata_path_for_predict):
            try:
                with open(metadata_path_for_predict, 'r') as f_meta:
                    metadata = json.load(f_meta)
                    training_end_str = metadata.get('training_end_date')
                    if training_end_str:
                        end_date_for_historical_load = datetime.strptime(training_end_str, "%Y-%m-%d")
                        print(f"Usando fecha de fin de entrenamiento del modelo para cargar datos históricos: {training_end_str}")
            except Exception as meta_err:
                print(f"Advertencia: No se pudo leer la fecha de fin de entrenamiento de los metadatos ({meta_err}). Usando fecha actual.")


        days_to_load_for_prediction = model.n_lags + 252 # n_lags + approx 1 year for context
        start_date_for_prediction = end_date_for_historical_load - timedelta(days=days_to_load_for_prediction * 1.5) # Cargar un poco más por días no hábiles

        print(f"Cargando datos históricos para predicción ({days_to_load_for_prediction} días) desde {start_date_for_prediction.strftime('%Y-%m-%d')} hasta {end_date_for_historical_load.strftime('%Y-%m-%d')}...")
        historical_data_df = load_stock_data_helper(
            ticket,
            start_date_for_prediction.strftime("%Y-%m-%d"),
            end_date_for_historical_load.strftime("%Y-%m-%d") # Cargar hasta la fecha de fin de entrenamiento o hoy
        )
        
        if len(historical_data_df) < model.n_lags + 5:
             raise HTTPException(
                status_code=400,
                detail=f"No hay suficientes datos históricos ({len(historical_data_df)} filas) para la predicción. Se necesitan al menos {model.n_lags + 5} después de cargar hasta {end_date_for_historical_load.strftime('%Y-%m-%d')}."
            )

        print("Realizando pronóstico XGBoost...")
        forecast_values, lower_bounds, upper_bounds = forecast_future_prices_xgb(
            model=model,
            data=historical_data_df.copy(), 
            forecast_horizon=forecast_horizon,
            target_col=target_col
        )

        last_actual_date_in_data = historical_data_df.index[-1]
        forecast_dates = pd.date_range(
            start=last_actual_date_in_data + pd.tseries.offsets.BDay(1), 
            periods=forecast_horizon,
            freq='B' 
        ).strftime('%Y-%m-%d').tolist()
        
        if len(forecast_values) != len(forecast_dates):
             print(f"Advertencia: La longitud de los valores pronosticados ({len(forecast_values)}) no coincide con las fechas de pronóstico ({len(forecast_dates)}). Ajustando...")
             min_len = min(len(forecast_values), len(forecast_dates))
             forecast_values = forecast_values[:min_len]
             lower_bounds = lower_bounds[:min_len]
             upper_bounds = upper_bounds[:min_len]
             forecast_dates = forecast_dates[:min_len]

        predictions_list = [
            {
                "date": forecast_dates[i],
                "prediction": float(forecast_values[i]),
                "lower_bound": float(lower_bounds[i]) if not np.isnan(lower_bounds[i]) else None, 
                "upper_bound": float(upper_bounds[i]) if not np.isnan(upper_bounds[i]) else None, 
            } for i in range(len(forecast_dates))
        ]

        historical_data_to_return = historical_data_df.iloc[-history_days:]

        return {
            "status": "success",
            "ticker": ticket,
            "model_type": "XGBoost",
            "target_column": target_col,
            "forecast_horizon": forecast_horizon,
            "historical_dates": historical_data_to_return.index.strftime('%Y-%m-%d').tolist(),
            "historical_values": [val if not np.isnan(val) else None for val in historical_data_to_return[target_col].tolist()],
            "predictions": predictions_list,
            "last_actual_date": last_actual_date_in_data.strftime("%Y-%m-%d"),
            "last_actual_value": float(historical_data_df[target_col].iloc[-1]) if not np.isnan(historical_data_df[target_col].iloc[-1]) else None,
            "model_used_prefix": os.path.basename(model_prefix)
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error detallado durante la predicción XGBoost: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error en predicción XGBoost: {str(e)}")


@app.get("/models", tags=["XGBoost Training & Management"])
async def list_xgb_models():
    try:
        metadata_files = glob.glob(os.path.join(MODEL_DIR, "xgb_model_*_metadata.json"))
        models_info = []

        for meta_file_path in metadata_files:
            model_prefix = meta_file_path.replace("_metadata.json", "")
            model_name = os.path.basename(model_prefix) 
            
            metadata_content = {"error": "No se pudo cargar el archivo de metadatos."}
            try:
                with open(meta_file_path, 'r') as f:
                    metadata_content = json.load(f)
            except Exception as e:
                print(f"Error al cargar metadatos desde {meta_file_path}: {e}")
            
            total_size_bytes = 0
            # Construct full paths for size calculation based on prefix and metadata content
            base_dir_for_model_files = os.path.dirname(meta_file_path) # Directory where metadata file is
            
            pipeline_filename = metadata_content.get('pipeline_file') # Should be just basename
            components_filename = metadata_content.get('components_file') # Should be just basename

            if pipeline_filename:
                pipeline_full_path = os.path.join(base_dir_for_model_files, pipeline_filename)
                if os.path.exists(pipeline_full_path):
                    total_size_bytes += os.path.getsize(pipeline_full_path)
            if components_filename:
                components_full_path = os.path.join(base_dir_for_model_files, components_filename)
                if os.path.exists(components_full_path):
                    total_size_bytes += os.path.getsize(components_full_path)

            if os.path.exists(meta_file_path):
                 total_size_bytes += os.path.getsize(meta_file_path)

            models_info.append({
                "name": model_name,
                "path_prefix_basename": os.path.basename(model_prefix), 
                "metadata_file": os.path.basename(meta_file_path),
                "metadata_content": metadata_content,
                "estimated_size_mb": round(total_size_bytes / (1024 * 1024), 3) if total_size_bytes > 0 else "N/A",
            })

        return {
            "total_models": len(models_info),
            "models": models_info
        }
    except Exception as e:
        print(f"Error listando modelos XGBoost: {e}")
        raise HTTPException(status_code=500, detail=f"Error listando modelos XGBoost: {str(e)}")

@app.get("/health", tags=["General"])
async def health_check_xgb():
    return {"status": "Ok", "service": "XGBoost Time Series Model Service"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
