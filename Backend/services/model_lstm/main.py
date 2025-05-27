from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import Optional, List
import os
import glob
from datetime import datetime, timedelta
import json 

from Backend.services.model_lstm.lstm_model import TimeSeriesLSTMModel
from Backend.services.model_lstm.train import train_lstm_model
from Backend.services.model_lstm.forecast import forecast_future_prices_lstm

from Backend.utils.import_data import load_data

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    print("ADVERTENCIA: La librería google-cloud-storage no está instalada. El endpoint /models no podrá listar desde GCS.")



app = FastAPI(
    title="LSTM Time Series Model Service",
    version="1.0.0",
    description="Un servicio para entrenar y realizar pronósticos con modelos LSTM para series de tiempo."
)

# --- Modelos Pydantic para los Requests ---

class TrainRequestLSTM(BaseModel):
    ticket: str = "NU"
    start_date: str = "2020-12-10"
    end_date: str = "2024-10-01"
    n_lags: int = 10
    target_col: str = "Close"
    train_size: float = 0.8
    model_base_name: Optional[str] = None 

    sequence_length: int = 60
    epochs: int = 50
    lstm_units: int = 50
    dropout_rate: float = 0.2
    optimize_params: bool = True
    patience: int = 10



MODEL_STORAGE_BASE_PATH = "lstm_models" 

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
if not GCS_BUCKET_NAME:
    print("ADVERTENCIA: La variable de entorno GCS_BUCKET_NAME no está configurada. El guardado/carga en GCS no funcionará como se espera en Cloud Run.")

loaded_lstm_models_cache = {}


def get_model_gcs_dir_path(ticket: str, model_base_name: Optional[str] = None, start_date_str: Optional[str] = None, end_date_str: Optional[str] = None) -> str:
    """
    Construye la ruta GCS para un directorio de modelo LSTM específico.
    Ejemplo: lstm_models/NU/lstm_model_NU_20201210_20231001/
    """
    if model_base_name:
        dir_name = model_base_name
    elif start_date_str and end_date_str:
        dir_name = f"lstm_model_{ticket}_{start_date_str.replace('-', '')}_{end_date_str.replace('-', '')}"
    else:
        dir_name = f"lstm_model_{ticket}_latest"
    return os.path.join(MODEL_STORAGE_BASE_PATH, ticket, dir_name)


def find_latest_model_gcs_dir_path_for_ticket(ticket: str) -> Optional[str]:
    """
    Encuentra la ruta GCS del directorio del modelo LSTM más reciente para un ticker.
    Simplificado por ahora.
    """
    generic_dir_name = f"lstm_model_{ticket}_latest"
    return os.path.join(MODEL_STORAGE_BASE_PATH, ticket, generic_dir_name)


def load_stock_data_helper(ticket: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        data = load_data(ticker=ticket, start_date=start_date, end_date=end_date)
        if data.empty:
            raise HTTPException(status_code=404,
                                detail=f"No se encontraron datos para el ticker {ticket} en el rango {start_date} a {end_date}.")
        return data
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Error descargando datos para {ticket}: {str(e)}")


# --- Endpoints de la API ---

@app.get("/", tags=["General"])
async def read_root_lstm():
    return {"message": "Servicio de Modelos de Series de Tiempo LSTM"}


@app.post("/train", tags=["LSTM Training & Management"])
async def train_lstm_model_endpoint(request: TrainRequestLSTM):
    try:
        print(f"Solicitud de entrenamiento LSTM recibida para el ticker: {request.ticket}")
        data = load_stock_data_helper(request.ticket, request.start_date, request.end_date)

        # Construir la ruta GCS para guardar el directorio del modelo
        model_save_gcs_dir_path = get_model_gcs_dir_path(
            request.ticket,
            model_base_name=request.model_base_name, 
            start_date_str=request.start_date,
            end_date_str=request.end_date
        )
        print(f"Directorio del modelo se guardará en GCS (si está configurado) en: {model_save_gcs_dir_path}")

        min_rows_needed = request.sequence_length + request.n_lags + 30 # Margen para lags, secuencias y splits
        if len(data) < min_rows_needed:
            raise HTTPException(
                status_code=400,
                detail=f"No hay suficientes datos históricos para entrenar. Se necesitan al menos {min_rows_needed} filas, pero se obtuvieron {len(data)}."
            )

        trained_model, residuals, residual_dates, acf_vals, pacf_vals, confint_acf, confint_pacf = train_lstm_model(
            data=data,
            target_col=request.target_col,
            sequence_length=request.sequence_length,
            n_lags=request.n_lags,
            lstm_units=request.lstm_units,
            dropout_rate=request.dropout_rate,
            train_size=request.train_size,
            epochs=request.epochs,
            optimize_params=request.optimize_params,
            patience=request.patience,
            save_model_path=model_save_gcs_dir_path, 
            bucket_name=GCS_BUCKET_NAME            
        )


        response_payload = {
            "status": "success",
            "message": f"Modelo LSTM entrenado exitosamente para {request.ticket}",
            "model_type": "LSTM",
            "metrics": trained_model.metrics if hasattr(trained_model, 'metrics') else "Métricas no disponibles.",
            "best_params": trained_model.best_params_ if hasattr(trained_model, 'best_params_') else "Parámetros no disponibles.",
            "residuals": residuals.tolist() if residuals is not None else [],
            "residual_dates": [d.strftime("%Y-%m-%d") for d in residual_dates] if residual_dates is not None else [],
            "acf": {
                "values": acf_vals.tolist() if acf_vals is not None else [],
                "confint_lower": confint_acf[:, 0].tolist() if confint_acf is not None else [],
                "confint_upper": confint_acf[:, 1].tolist() if confint_acf is not None else []
            } if acf_vals is not None else None,
            "pacf": {
                "values": pacf_vals.tolist() if pacf_vals is not None else [],
                "confint_lower": confint_pacf[:, 0].tolist() if confint_pacf is not None else [],
                "confint_upper": confint_pacf[:, 1].tolist() if confint_pacf is not None else []
            } if pacf_vals is not None else None,
            "model_reference_path": f"gs://{GCS_BUCKET_NAME}/{model_save_gcs_dir_path}" if GCS_BUCKET_NAME else model_save_gcs_dir_path
        }
        
        if hasattr(trained_model, 'best_params_') and trained_model.best_params_:
            serializable_best_params = {
                k: v.item() if isinstance(v, np.generic) else v 
                for k, v in trained_model.best_params_.items()
            }
            response_payload["best_params"] = serializable_best_params
            
        return response_payload

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Error en procesamiento de datos LSTM: {str(ve)}")
    except FileNotFoundError as fnfe:
        raise HTTPException(status_code=404, detail=str(fnfe))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error entrenando modelo LSTM: {str(e)}")


@app.get("/predict", tags=["LSTM Prediction"])
async def predict_lstm_endpoint(
        ticket: str = Query("NU", description="Ticker de la acción a predecir"),
        forecast_horizon: int = Query(10, gt=0, description="Horizonte de pronóstico en días"),
        target_col: str = Query("Close", description="Columna objetivo para la predicción"),
        history_days: int = Query(365, description="Número de días históricos a devolver en la respuesta"),
        model_dir_name_override: Optional[str] = Query(None, description="Nombre exacto del directorio del modelo a usar (ej. lstm_model_NU_20201210_20231001)")
):
    try:
        print(f"Solicitud de predicción LSTM recibida para el ticker: {ticket}")

        if model_dir_name_override:
            model_to_load_gcs_dir_path = os.path.join(MODEL_STORAGE_BASE_PATH, ticket, model_dir_name_override)
        else:
            model_to_load_gcs_dir_path = find_latest_model_gcs_dir_path_for_ticket(ticket)
            if model_to_load_gcs_dir_path is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"No se pudo determinar una ruta de directorio de modelo LSTM para {ticket}."
                )
        
        print(f"Intentando cargar directorio de modelo LSTM desde GCS (si está configurado): {model_to_load_gcs_dir_path}")

        cache_key = f"gs://{GCS_BUCKET_NAME}/{model_to_load_gcs_dir_path}" if GCS_BUCKET_NAME else model_to_load_gcs_dir_path
        
        if cache_key in loaded_lstm_models_cache:
            model = loaded_lstm_models_cache[cache_key]
            print(f"Modelo LSTM cargado desde caché para: {ticket}")
        else:
            try:
                model = TimeSeriesLSTMModel.load_model(
                    dir_path=model_to_load_gcs_dir_path,
                    bucket_name=GCS_BUCKET_NAME
                )
                loaded_lstm_models_cache[cache_key] = model
                print(f"Modelo LSTM cargado exitosamente para: {ticket} desde {model_to_load_gcs_dir_path}")
            except FileNotFoundError:
                 raise HTTPException(status_code=404, detail=f"Directorio de modelo LSTM no encontrado en '{model_to_load_gcs_dir_path}'. Verifica el nombre o entrena uno nuevo.")
            except Exception as load_error:
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Error cargando modelo LSTM: {str(load_error)}")

        end_date_history = datetime.now()
        required_initial_rows = model.preprocessor.sequence_length + model.preprocessor.n_lags + 50 # + buffer
        start_date_history = end_date_history - timedelta(days=max(history_days, required_initial_rows) + 250) 

        print(f"Cargando datos históricos para predicción LSTM desde {start_date_history.strftime('%Y-%m-%d')} hasta {end_date_history.strftime('%Y-%m-%d')}")
        historical_data = load_stock_data_helper(
            ticket,
            start_date_history.strftime("%Y-%m-%d"),
            end_date_history.strftime("%Y-%m-%d")
        )

        if len(historical_data) < required_initial_rows:
            raise HTTPException(
                status_code=400,
                detail=f"No hay suficientes datos históricos ({len(historical_data)}) para construir la secuencia inicial de predicción LSTM (se necesitan al menos {required_initial_rows})."
            )

        print("Realizando pronóstico LSTM...")
        forecast_values, lower_bounds, upper_bounds = forecast_future_prices_lstm(
            model=model,
            data=historical_data.copy(),
            forecast_horizon=forecast_horizon,
            target_col=target_col
        )

        last_actual_date_in_data = historical_data.index[-1]
        forecast_dates = pd.date_range(
            start=last_actual_date_in_data + timedelta(days=1),
            periods=forecast_horizon,
            freq='B'
        ).strftime('%Y-%m-%d').tolist()

        predictions_list = [
            {
                "date": forecast_dates[i],
                "prediction": float(forecast_values[i]),
                "lower_bound": float(lower_bounds[i]) if lower_bounds is not None else None,
                "upper_bound": float(upper_bounds[i]) if upper_bounds is not None else None
            } for i in range(len(forecast_dates))
        ]

        historical_data_to_return = historical_data.iloc[-history_days:]
        model_ref_path = f"gs://{GCS_BUCKET_NAME}/{model_to_load_gcs_dir_path}" if GCS_BUCKET_NAME else model_to_load_gcs_dir_path

        return {
            "status": "success",
            "ticker": ticket,
            "model_type": "LSTM",
            "target_column": target_col,
            "forecast_horizon": forecast_horizon,
            "historical_dates": historical_data_to_return.index.strftime('%Y-%m-%d').tolist(),
            "historical_values": historical_data_to_return[target_col].tolist(),
            "predictions": predictions_list,
            "last_actual_date": last_actual_date_in_data.strftime("%Y-%m-%d"),
            "last_actual_value": float(historical_data[target_col].iloc[-1]),
            "model_used_reference": model_ref_path
        }

    except HTTPException:
        raise
    except FileNotFoundError as fnfe:
        raise HTTPException(status_code=404, detail=str(fnfe))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error en predicción LSTM: {str(e)}")


@app.get("/models", tags=["LSTM Training & Management"], summary="Listar modelos LSTM disponibles (directorios)")
async def list_lstm_models_endpoint(ticket_filter: Optional[str] = Query(None, description="Filtrar modelos por ticker")):
    if not GCS_BUCKET_NAME or not GCS_AVAILABLE:
        return {"message": "GCS_BUCKET_NAME no configurado o google-cloud-storage no disponible. No se pueden listar modelos de GCS."}

    models_info = []
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        
        prefix_to_list = f"{MODEL_STORAGE_BASE_PATH}/"
        if ticket_filter:
            prefix_to_list = f"{MODEL_STORAGE_BASE_PATH}/{ticket_filter}/"

        blobs = bucket.list_blobs(prefix=prefix_to_list, delimiter='/') # delimiter ayuda a obtener prefijos
        
        # Iterar sobre los prefijos (que simulan directorios)
        for prefix_obj in blobs.prefixes: # blobs.prefixes contiene los "subdirectorios"
            dir_name_full = prefix_obj.strip('/') # ej: lstm_models/NU/lstm_model_NU_20230101_20230101
            dir_name_simple = os.path.basename(dir_name_full)

            metadata_content = {"info": "Metadatos no implementados para listado de directorios LSTM aún."}
       
            models_info.append({
                "directory_name": dir_name_simple,
                "full_gcs_directory_path": f"gs://{GCS_BUCKET_NAME}/{dir_name_full}",
                "metadata_status": "No implementado para directorios LSTM"
            })
        
        if not models_info and ticket_filter:
             return {"message": f"No se encontraron directorios de modelos LSTM para el ticker '{ticket_filter}' en GCS bajo el prefijo '{prefix_to_list}'."}
        elif not models_info:
            return {"message": f"No se encontraron directorios de modelos LSTM en GCS bajo el prefijo '{prefix_to_list}'."}

        return {
            "total_model_directories_found": len(models_info),
            "bucket_queried": GCS_BUCKET_NAME,
            "prefix_used": prefix_to_list,
            "model_directories": models_info
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error listando directorios de modelos LSTM desde GCS: {str(e)}")


@app.get("/health", tags=["General"])
async def health_check_lstm():
    return {"status": "Ok", "service": "LSTM Time Series Model Service"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8002)) # Puerto para LSTM
    uvicorn.run(app, host="0.0.0.0", port=port)