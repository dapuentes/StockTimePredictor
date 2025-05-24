# Este archivo debe estar en: Backend/services/model_lstm/main.py

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import Optional
import os
import glob
from datetime import datetime, timedelta

# Importar módulos específicos de LSTM (relativos a la ubicación de este main.py)
from .lstm_model import TimeSeriesLSTMModel
from .train import train_lstm_model
from .forecast import forecast_future_prices_lstm

# Importar módulos de utilidades (asumiendo que 'Backend' está en PYTHONPATH)
from utils.import_data import load_data

app = FastAPI(
    title="LSTM Time Series Model Service",
    version="1.0.0",
    description="Un servicio para entrenar y realizar pronósticos con modelos LSTM para series de tiempo."
)


# --- Modelos Pydantic para los Requests ---

class BaseTrainRequest(BaseModel):
    """Configuración base para el entrenamiento de modelos."""
    ticket: str = "NU"
    start_date: str = "2020-12-10"
    end_date: str = "2024-10-01"
    n_lags: int = 10  # Relevante para la creación de lags en el preprocesador
    target_col: str = "Close"
    train_size: float = 0.8
    save_model_path: Optional[str] = None  # Para LSTM, esto será un directorio


class TrainRequestLSTM(BaseTrainRequest):
    """Configuración específica para entrenar un modelo LSTM."""
    sequence_length: int = 30
    epochs: int = 50
    lstm_units: int = 50
    dropout_rate: float = 0.2
    optimize_params: bool = True


# --- Configuración de Rutas y Caching ---

# Directorio de modelos específico para este servicio LSTM
# Estará en: Backend/services/model_lstm/models/
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Caché para los modelos LSTM cargados
loaded_lstm_models_cache = {}


# --- Funciones Auxiliares (Helpers) para el Manejo de Modelos LSTM ---

def get_default_lstm_model_dir(ticket: str) -> str:
    """Genera la ruta al directorio por defecto para un modelo LSTM de un ticket."""
    return os.path.join(MODEL_DIR, f"lstm_model_{ticket}")


def find_lstm_model_dir(ticket: str) -> Optional[str]:
    """
    Encuentra el directorio de un modelo LSTM entrenado para un ticket.
    Los modelos LSTM se guardan como directorios.
    """
    model_dir_path = get_default_lstm_model_dir(ticket)
    if os.path.exists(model_dir_path) and os.path.isdir(model_dir_path):
        return model_dir_path

    # Fallback: buscar cualquier directorio que parezca un modelo LSTM
    # (Esta lógica puede ser más sofisticada si es necesario)
    available_model_dirs = [d for d in glob.glob(os.path.join(MODEL_DIR, "lstm_model_*")) if os.path.isdir(d)]
    if available_model_dirs:
        print(
            f"Advertencia: No se encontró modelo LSTM específico para {ticket}. Usando el primero disponible: {available_model_dirs[0]}")
        return available_model_dirs[0]
    return None


def load_lstm_model_from_dir(dir_path: str) -> TimeSeriesLSTMModel:
    """Carga un modelo TimeSeriesLSTMModel desde el directorio especificado y lo cachea."""
    if dir_path in loaded_lstm_models_cache:
        return loaded_lstm_models_cache[dir_path]

    try:
        model = TimeSeriesLSTMModel.load_model(dir_path)  # El método de clase se encarga de cargar componentes
        loaded_lstm_models_cache[dir_path] = model
        return model
    except Exception as e:
        print(f"Error detallado al cargar modelo LSTM desde {dir_path}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error cargando modelo LSTM desde {dir_path}: {str(e)}")


def load_stock_data_helper(ticket: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Función auxiliar para cargar datos de acciones con manejo de errores HTTP."""
    try:
        data = load_data(ticker=ticket, start_date=start_date, end_date=end_date)
        if data.empty:
            raise HTTPException(status_code=404,
                                detail=f"No se encontraron datos para el ticker {ticket} en el rango especificado.")
        return data
    except Exception as e:
        # Si load_data ya lanza HTTPException, esto podría ser redundante,
        # pero es bueno tener un catch-all aquí.
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Error descargando datos para {ticket}: {str(e)}")


# --- Endpoints de la API ---

@app.get("/", tags=["General"])
async def read_root_lstm():
    """Endpoint raíz del servicio de modelos LSTM."""
    return {"message": "Servicio de Modelos de Series de Tiempo LSTM"}


@app.post("/train", tags=["LSTM Training & Management"])
async def train_lstm_endpoint(request: TrainRequestLSTM):
    """Entrena y guarda un modelo LSTM basado en la configuración proporcionada."""
    try:
        print(f"Solicitud de entrenamiento LSTM recibida para el ticker: {request.ticket}")
        data = load_stock_data_helper(request.ticket, request.start_date, request.end_date)

        # La ruta de guardado para LSTM es un directorio.
        # Si no se proporciona, se usa una ruta por defecto.
        save_dir = request.save_model_path or get_default_lstm_model_dir(request.ticket)

        # Validación simple de la cantidad de datos
        min_rows_needed = request.sequence_length + 30  # Un margen para lags y splits
        if len(data) < min_rows_needed:
            raise HTTPException(
                status_code=400,
                detail=f"No hay suficientes datos históricos para entrenar. Se necesitan al menos {min_rows_needed} filas, pero se obtuvieron {len(data)}."
            )

        print(f"Iniciando entrenamiento del modelo LSTM. Se guardará en: {save_dir}")
        trained_model, residuals, residual_dates = train_lstm_model(
            data=data,
            target_col=request.target_col,
            sequence_length=request.sequence_length,
            n_lags=request.n_lags,  # Usado por el preprocesador LSTM
            lstm_units=request.lstm_units,
            dropout_rate=request.dropout_rate,
            train_size=request.train_size,
            epochs=request.epochs,
            optimize_params=request.optimize_params,
            save_model_path=save_dir  # train_lstm_model espera un directorio aquí
        )

        # Opcional: añadir el modelo recién entrenado al caché
        loaded_lstm_models_cache[save_dir] = trained_model

        response = {
            "status": "success",
            "message": f"Modelo LSTM entrenado exitosamente para {request.ticket}",
            "model_type": "LSTM",
            "metrics": trained_model.metrics if hasattr(trained_model, 'metrics') else "Métricas no disponibles.",
            "model_path": os.path.basename(save_dir),
            "residuals": residuals.tolist(),
            "residual_dates": residual_dates.strftime('%Y-%m-%d').tolist(),
        }

        # Añadir los mejores parámetros a la respuesta si el atributo existe y no está vacío
        if hasattr(trained_model, 'best_params_') and trained_model.best_params_:
            # Serializar los parámetros para evitar problemas 
            best_params_serializable = {k: v.item() if isinstance(v, np.generic) else v for k, v in
                                        trained_model.best_params_.items()}
            response["best_params"] = best_params_serializable

        return response

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error detallado durante el entrenamiento LSTM: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error entrenando modelo LSTM: {str(e)}")


@app.get("/predict", tags=["LSTM Prediction"])
async def predict_lstm_endpoint(
        ticket: str = Query("NU", description="Ticker de la acción a predecir"),
        forecast_horizon: int = Query(10, description="Horizonte de pronóstico en días"),
        target_col: str = Query("Close", description="Columna objetivo para la predicción"),
        history_days: int = Query(365, description="Número de días históricos a devolver en la respuesta")
):
    """Genera un pronóstico usando un modelo LSTM entrenado, incluyendo intervalos de confianza."""
    try:
        print(f"Solicitud de predicción LSTM recibida para el ticker: {ticket}")
        model_dir_path = find_lstm_model_dir(ticket)
        if not model_dir_path:
            raise HTTPException(status_code=404,
                                detail=f"No se encontró un directorio de modelo LSTM entrenado para {ticket}.")

        print(f"Cargando modelo LSTM desde: {model_dir_path}")
        model = load_lstm_model_from_dir(model_dir_path)

        # Cargar datos históricos suficientes para la secuencia inicial
        # El preprocesador LSTM tiene el atributo sequence_length
        required_sequence_length = model.preprocessor.sequence_length
        # Cargar un poco más para asegurar que después del preprocesamiento y lags queden suficientes datos
        days_to_load_for_sequence = 365 * 3

        current_date = datetime.now()
        start_date_for_prediction = current_date - timedelta(days=days_to_load_for_sequence)

        print(f"Cargando datos históricos para la predicción ({days_to_load_for_sequence} días)...")
        historical_data = load_stock_data_helper(
            ticket,
            start_date_for_prediction.strftime("%Y-%m-%d"),
            current_date.strftime("%Y-%m-%d")
        )

        # Validación de datos para predicción
        # Después del preprocesamiento en predict_future, se necesitará al menos sequence_length filas
        # Esta es una comprobación previa más simple.
        if len(historical_data) < required_sequence_length + model.preprocessor.n_lags:
            raise HTTPException(
                status_code=400,
                detail=f"No hay suficientes datos históricos ({len(historical_data)}) para construir la secuencia inicial de predicción (se necesitan al menos {required_sequence_length + model.preprocessor.n_lags})."
            )

        print("Realizando pronóstico LSTM...")
        forecast, lower_bounds, upper_bounds = forecast_future_prices_lstm(
            model=model,
            data=historical_data.copy(),  # Pasar una copia para evitar modificaciones
            forecast_horizon=forecast_horizon,
            target_col=target_col
            # n_iter_mc se puede añadir como parámetro Query si se desea configurar desde la API
        )

        # Formateo de la respuesta (similar al de RF para consistencia)
        last_actual_date_in_data = historical_data.index[-1]
        forecast_dates = pd.date_range(
            start=last_actual_date_in_data + timedelta(days=1),
            periods=forecast_horizon,
            freq='B'  # 'B' para días hábiles
        ).strftime('%Y-%m-%d').tolist()

        predictions_list = [
            {
                "date": forecast_dates[i],
                "prediction": float(forecast[i]),
                "lower_bound": float(lower_bounds[i]),
                "upper_bound": float(upper_bounds[i])
            } for i in range(len(forecast_dates))
        ]

        historical_data_to_return = historical_data.iloc[-history_days:]

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
            "model_used": os.path.basename(model_dir_path)
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error detallado durante la predicción LSTM: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error en predicción LSTM: {str(e)}")


@app.get("/models", tags=["LSTM Training & Management"])
async def list_lstm_models():
    """Lista todos los modelos LSTM disponibles (directorios de modelos)."""
    try:
        model_dirs = [d for d in glob.glob(os.path.join(MODEL_DIR, "lstm_model_*")) if os.path.isdir(d)]
        models_info = []

        for model_dir_path in model_dirs:
            model_name = os.path.basename(model_dir_path)
            # Podrías añadir lógica para leer metadatos si los guardas (ej. un json en el dir)
            # Por ahora, solo información básica del directorio
            dir_size_bytes = sum(
                os.path.getsize(os.path.join(dirpath, filename)) for dirpath, _, filenames in os.walk(model_dir_path)
                for filename in filenames)

            models_info.append({
                "name": model_name,
                "path_type": "directory",
                "full_path": model_dir_path,  # Considera si quieres exponer la ruta completa
                "size_mb": round(dir_size_bytes / (1024 * 1024), 2),
            })

        return {
            "total_models": len(models_info),
            "models": models_info
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listando modelos LSTM: {str(e)}")


@app.get("/health", tags=["General"])
async def health_check_lstm():
    """Verifica el estado de salud del servicio de modelos LSTM."""
    return {"status": "Ok", "service": "LSTM Time Series Model Service"}


# --- Punto de Entrada para Uvicorn (si se ejecuta este archivo directamente) ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)