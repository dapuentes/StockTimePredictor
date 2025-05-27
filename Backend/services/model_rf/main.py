from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
import os
from typing import Optional, List 
import glob
from datetime import datetime, timedelta
from google.cloud import storage
import json

from Backend.services.model_rf.rf_model import TimeSeriesRandomForestModel
from Backend.services.model_rf.train import train_ts_model
from Backend.services.model_rf.forecast import forecast_future_prices
from Backend.utils.import_data import load_data

app = FastAPI(title="Random Forest Time Series Model Service", version="1.0.0")


class TrainRequest(BaseModel):
    ticket: str = "NU"
    start_date: str = "2020-12-10"
    end_date: str = "2023-10-01" 
    n_lags: int = 10
    target_col: str = "Close"
    train_size: float = 0.8
    model_base_name: Optional[str] = None


MODEL_STORAGE_BASE_PATH = "rf_models" # Directorio base dentro del bucket para modelos RF


GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
if not GCS_BUCKET_NAME:
    print("ADVERTENCIA: La variable de entorno GCS_BUCKET_NAME no está configurada. El guardado/carga en GCS no funcionará como se espera en Cloud Run.")


loaded_models_cache = {}


def get_model_gcs_path(ticket: str, model_base_name: Optional[str] = None, start_date_str: Optional[str] = None, end_date_str: Optional[str] = None) -> str:
    """
    Construye la ruta GCS para un modelo específico.
    Ejemplo: rf_models/NU/rf_model_NU_20201210_20231001.joblib
    """
    if model_base_name:
        file_name = model_base_name
    elif start_date_str and end_date_str:
        file_name = f"rf_model_{ticket}_{start_date_str.replace('-', '')}_{end_date_str.replace('-', '')}.joblib"
    else:
        file_name = f"rf_model_{ticket}_latest.joblib"
    return os.path.join(MODEL_STORAGE_BASE_PATH, ticket, file_name)


def find_latest_model_gcs_path_for_ticket(ticket: str) -> Optional[str]:
    """
    Encuentra la ruta GCS del modelo más reciente para un ticker.
    Esta función necesitaría listar objetos en GCS, lo cual es más complejo.
    Por ahora, simplificaremos asumiendo un nombre de modelo predecible o
    que se usa el modelo más genérico si no se encuentra uno específico.

    """
    generic_model_name = f"rf_model_{ticket}_latest.joblib" # Un nombre que podrías usar al guardar el "mejor" o más reciente
    
  
    return os.path.join(MODEL_STORAGE_BASE_PATH, ticket, generic_model_name)


def load_stock_data(ticket, start_date, end_date):
    try:
        data = load_data(ticker=ticket, start_date=start_date, end_date=end_date) #
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for ticket {ticket} in range {start_date} to {end_date}")
        return data
    except Exception as e:
        # Si load_data ya lanza HTTPException, esto podría ser redundante
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Error downloading data for {ticket}: {str(e)}")


@app.get("/")
async def read_root():
    return {"message": "Random Forest Time Series Model Service"}


@app.post("/train")
async def train_model_endpoint(request: TrainRequest): # Renombrado para claridad
    try:
        print(f"Solicitud de entrenamiento RF recibida para el ticker: {request.ticket}")
        data = load_stock_data(request.ticket, request.start_date, request.end_date)

        model_save_gcs_path = get_model_gcs_path(
            request.ticket,
            start_date_str=request.start_date,
            end_date_str=request.end_date
        )
        print(f"Modelo se guardará en GCS (si está configurado) en: {model_save_gcs_path}")

        min_days = 260 
        if len(data) < min_days:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough historical data for training. Need at least {min_days} rows, but got {len(data)}."
            )

        model, feature_names, residuals, residual_dates, acf_values, pacf_values, confint_acf, confint_pacf = train_ts_model(
            data=data,
            n_lags=request.n_lags,
            target_col=request.target_col,
            train_size=request.train_size,
            save_model_path=model_save_gcs_path, # Esta es la RUTA DENTRO del bucket
            bucket_name=GCS_BUCKET_NAME          # Nombre del bucket
        )

        return {
            "status": "success",
            "message": f"Modelo RF entrenado exitosamente para {request.ticket}",
            "model_type": "RandomForest",
            "metrics": model.metrics if hasattr(model, 'metrics') else "Métricas no disponibles.",
            "features_names": feature_names,
            "best_params": model.best_params_ if hasattr(model, 'best_params_') else "Parámetros no disponibles.",
            "residuals": residuals.tolist() if residuals is not None else [],
            "residual_dates": [d.strftime("%Y-%m-%d") for d in residual_dates] if residual_dates is not None else [],
            "acf": {
                "values": acf_values.tolist() if acf_values is not None else [],
                "confint_lower": confint_acf[:, 0].tolist() if confint_acf is not None else [],
                "confint_upper": confint_acf[:, 1].tolist() if confint_acf is not None else []
            } if acf_values is not None else None,
            "pacf": {
                "values": pacf_values.tolist() if pacf_values is not None else [],
                "confint_lower": confint_pacf[:, 0].tolist() if confint_pacf is not None else [],
                "confint_upper": confint_pacf[:, 1].tolist() if confint_pacf is not None else []
            } if pacf_values is not None else None,
            # Devuelve la ruta GCS para referencia, no la ruta local absoluta
            "model_reference_path": f"gs://{GCS_BUCKET_NAME}/{model_save_gcs_path}" if GCS_BUCKET_NAME else model_save_gcs_path
        }

    except ValueError as ve:
        # Errores de validación o preparación de datos
        raise HTTPException(status_code=400, detail=f"Error en procesamiento de datos: {str(ve)}")
    except FileNotFoundError as fnfe:
        # Error si un archivo esperado no se encuentra (más probable en carga que en entreno)
        raise HTTPException(status_code=404, detail=str(fnfe))
    except Exception as e:
        # Otros errores inesperados
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error entrenando modelo RF: {str(e)}")


@app.get("/predict")
async def predict_endpoint( # Renombrado para claridad
        ticket: str = Query("NU", description="Ticker de la acción a predecir"),
        forecast_horizon: int = Query(10, gt=0, description="Horizonte de pronóstico en días"),
        target_col: str = Query("Close", description="Columna objetivo para la predicción"),
        history_days: int = Query(365, description="Número de días históricos a considerar para el gráfico y la predicción inicial"),
        model_name_override: Optional[str] = Query(None, description="Nombre exacto del archivo del modelo a usar (ej. rf_model_NU_20201210_20231001.joblib)")
):
    try:
        print(f"Solicitud de predicción RF recibida para el ticker: {ticket}")

        # Determinar la ruta GCS del modelo a cargar
        if model_name_override:
            model_to_load_gcs_path = os.path.join(MODEL_STORAGE_BASE_PATH, ticket, model_name_override)
        else:
            model_to_load_gcs_path = find_latest_model_gcs_path_for_ticket(ticket)
            if model_to_load_gcs_path is None: #  find_latest_model_gcs_path_for_ticket ahora devuelve la ruta completa
                 raise HTTPException(
                    status_code=404,
                    detail=f"No se pudo determinar una ruta de modelo para {ticket}. Considere usar model_name_override."
                )

        print(f"Intentando cargar modelo desde GCS (si está configurado): {model_to_load_gcs_path}")

        cache_key = f"gs://{GCS_BUCKET_NAME}/{model_to_load_gcs_path}" if GCS_BUCKET_NAME else model_to_load_gcs_path
        
        if cache_key in loaded_models_cache:
            model = loaded_models_cache[cache_key]
            print(f"Modelo RF cargado desde caché para: {ticket}")
        else:
            try:
                model = TimeSeriesRandomForestModel.load_model(
                    model_path=model_to_load_gcs_path, # Ruta DENTRO del bucket
                    bucket_name=GCS_BUCKET_NAME
                )
                loaded_models_cache[cache_key] = model
                print(f"Modelo RF cargado exitosamente para: {ticket} desde {model_to_load_gcs_path}")
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail=f"Modelo no encontrado en '{model_to_load_gcs_path}'. Verifica el nombre o entrena uno nuevo.")
            except Exception as load_error:
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Error cargando modelo RF: {str(load_error)}")

        end_date_history = datetime.now()
        
        days_for_features_and_lags = model.n_lags + 70 #  10 lags + ventana de 60 días para alguna feature
        start_date_history = end_date_history - timedelta(days=max(history_days, days_for_features_and_lags) + 250) # 250 días de datos para calcular las features del predict_future.

        print(f"Cargando datos históricos para predicción RF desde {start_date_history.strftime('%Y-%m-%d')} hasta {end_date_history.strftime('%Y-%m-%d')}")
        historical_data = load_stock_data(ticket, start_date_history.strftime("%Y-%m-%d"), end_date_history.strftime("%Y-%m-%d"))

        if len(historical_data) < days_for_features_and_lags:
             raise HTTPException(
                status_code=400,
                detail=f"No hay suficientes datos históricos ({len(historical_data)}) para construir las características y lags iniciales (se necesitan al menos {days_for_features_and_lags})."
            )

        # Realizar el pronóstico
        forecast_values, lower_bounds, upper_bounds = forecast_future_prices(
            model=model,
            data=historical_data.copy(), # Importante pasar una copia
            forecast_horizon=forecast_horizon,
            target_col=target_col
        )

        # Formatear la respuesta
        last_actual_date_in_data = historical_data.index[-1]
        forecast_dates = pd.date_range(
            start=last_actual_date_in_data + timedelta(days=1),
            periods=forecast_horizon,
            freq='B'  # 'B' para días hábiles
        ).strftime('%Y-%m-%d').tolist()

        predictions_list = []
        for i in range(len(forecast_dates)):
            predictions_list.append({
                "date": forecast_dates[i],
                "prediction": float(forecast_values[i]),
                "lower_bound": float(lower_bounds[i]) if lower_bounds is not None else None, # Manejar si no hay bounds
                "upper_bound": float(upper_bounds[i]) if upper_bounds is not None else None
            })

        # Devolver solo la porción de datos históricos solicitada por `history_days`
        historical_data_to_return = historical_data.iloc[-history_days:]

        model_ref_path = f"gs://{GCS_BUCKET_NAME}/{model_to_load_gcs_path}" if GCS_BUCKET_NAME else model_to_load_gcs_path

        return {
            "status": "success",
            "ticker": ticket,
            "model_type": "RandomForest",
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
        raise HTTPException(status_code=500, detail=f"Error en predicción RF: {str(e)}")


@app.get("/models", summary="Listar modelos RF disponibles (simplificado)")
async def list_rf_models(ticket_filter: Optional[str] = Query(None, description="Filtrar modelos por ticker")):
    """
    Lista modelos RF disponibles.
    Por ahora, solo devolverá un ejemplo o información estática.
    """
    if not GCS_BUCKET_NAME:
        return {"message": "GCS_BUCKET_NAME no está configurado. No se pueden listar modelos de GCS."}

    models_info = []
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        
        # Construir el prefijo para listar (directorio base de modelos RF)
        prefix_to_list = f"{MODEL_STORAGE_BASE_PATH}/"
        if ticket_filter:
            prefix_to_list = f"{MODEL_STORAGE_BASE_PATH}/{ticket_filter}/"

        blobs = bucket.list_blobs(prefix=prefix_to_list)
        
        for blob in blobs:
            if blob.name.endswith(".joblib"): # Considerar solo archivos de modelo
                # Intentar cargar metadatos si existen
                metadata_path = blob.name.replace(".joblib", "_metadata.json")
                metadata_blob = bucket.blob(metadata_path)
                metadata_content = None
                if metadata_blob.exists():
                    try:
                        metadata_content = json.loads(metadata_blob.download_as_text())
                    except Exception as e:
                        metadata_content = {"error": f"No se pudieron cargar metadatos: {str(e)}"}
                
                models_info.append({
                    "name": os.path.basename(blob.name),
                    "full_gcs_path": f"gs://{GCS_BUCKET_NAME}/{blob.name}",
                    "size_bytes": blob.size,
                    "last_modified": blob.updated.isoformat() if blob.updated else None,
                    "metadata": metadata_content
                })
        
        if not models_info and ticket_filter:
             return {"message": f"No se encontraron modelos RF para el ticker '{ticket_filter}' en GCS bajo el prefijo '{prefix_to_list}'. Verifica la ruta y los nombres de archivo."}
        elif not models_info:
            return {"message": f"No se encontraron modelos RF en GCS bajo el prefijo '{prefix_to_list}'."}


        return {
            "total_models_found": len(models_info),
            "bucket_queried": GCS_BUCKET_NAME,
            "prefix_used": prefix_to_list,
            "models": models_info
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error listando modelos RF desde GCS: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "Ok", "service": "Random Forest Time Series Model Service"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001)) 
    uvicorn.run(app, host="0.0.0.0", port=port)