import os
import glob
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import sys  # Importar sys para manipulación de path si es necesario

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
import numpy as np
import json

# --- AJUSTE DE RUTAS DE IMPORTACIÓN ---
# Si main_lstm.py está en services/model_lstm/

try:
    # Importaciones relativas para módulos en el mismo paquete (model_lstm)
    from .lstm_model import TimeSeriesLSTMModel  # Asumiendo que el archivo actualizado es lstm_model_refactored.py
    from .train import train_lstm_model
    from .forecast import forecast_future_prices

    from utils.import_data import load_data
except ImportError as e:
    print(f"Error inicial importando módulos con rutas relativas: {e}.")
    print("Intentando fallback o añadiendo ruta al sistema para módulos...")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Añadido {project_root} a sys.path")

    try:
        from services.model_lstm.lstm_model import \
            TimeSeriesLSTMModel  # Asumiendo que el archivo actualizado es lstm_model_refactored.py
        from services.model_lstm.train import train_lstm_model
        from services.model_lstm.forecast import forecast_future_prices
        from utils.import_data import load_data

        print("Importaciones exitosas después de ajustar sys.path.")
    except ImportError as e_fallback:
        print(f"Error importando módulos incluso después de ajustar sys.path: {e_fallback}")
        print("Usando importaciones directas como último recurso...")
        from lstm_model import TimeSeriesLSTMModel
        from train import train_lstm_model
        from forecast import forecast_future_prices

        try:
            from utils.import_data import load_data
        except ImportError:
            def load_data(ticker, start_date, end_date):  # Dummy
                print(f"ADVERTENCIA: Usando función load_data ficticia para {ticker} de {start_date} a {end_date}")
                dates = pd.date_range(start_date, end_date, freq='B');
                if dates.empty and start_date == end_date: dates = pd.date_range(start_date, periods=1)
                if dates.empty: return pd.DataFrame()
                data = pd.DataFrame(index=dates);
                data['Open'] = np.random.rand(len(dates)) * 100 + 50
                data['High'] = data['Open'] + np.random.rand(len(dates)) * 10;
                data['Low'] = data['Open'] - np.random.rand(len(dates)) * 10
                data['Close'] = data['Open'] + (np.random.rand(len(dates)) - 0.5) * 5;
                data['Volume'] = np.random.randint(100000, 10000000, size=len(dates))
                data['GreenDay'] = (data['Close'] > data['Open']).astype(int);
                return data


            print("Función load_data ficticia creada.")

app = FastAPI(title="LSTM Time Series Model Service", version="1.0.5")  # Versión actualizada


# --- Modelos Pydantic para Solicitudes y Respuestas ---
class TrainRequestLSTM(BaseModel):
    ticket: str = "NU"
    start_date: str = "2021-01-01"
    end_date: str = "2023-12-31"
    n_lags: int = 20
    target_col: str = "Close"
    train_size_ratio: float = 0.8
    save_model_path_prefix: Optional[str] = None
    epochs: int = 50
    batch_size: int = 32
    use_hyperparameter_optimization: bool = True
    hp_strategy: str = "bayesian"
    hp_max_trials: int = 5
    hp_epochs_per_trial: int = 10
    hp_project_name: str = "lstm_fastapi_tuning"
    initial_lstm_units: int = 50
    initial_lstm_layers: int = 1
    initial_dropout_rate: float = 0.2
    initial_learning_rate: float = 0.001


class PredictionResponseLSTM(BaseModel):
    status: str;
    ticker: str;
    target_column: str;
    forecast_horizon: int
    historical_dates: List[str];
    historical_values: List[float]
    predictions: List[Dict[str, Any]];
    last_actual_date: str;
    last_actual_value: float
    model_used: str;
    model_info: str


class TrainResponseLSTM(BaseModel):
    status: str;
    message: str
    metrics: Optional[Dict[str, float]] = None
    best_hyperparameters: Optional[Dict[str, Any]] = None
    feature_names: Optional[List[str]] = None
    residuals: Optional[List[float]] = None
    residual_dates: Optional[List[str]] = None
    model_path_prefix: Optional[str] = None


MODEL_DIR_LSTM = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR_LSTM, exist_ok=True)
print(f"Directorio de modelos LSTM configurado en: {MODEL_DIR_LSTM}")
loaded_lstm_models: Dict[str, TimeSeriesLSTMModel] = {}


def get_default_lstm_model_path_prefix(ticket: str) -> str:
    return os.path.join(MODEL_DIR_LSTM, f"lstm_model_{ticket.upper()}")


def get_generic_lstm_model_path_prefix() -> str:
    return os.path.join(MODEL_DIR_LSTM, "lstm_model_generic")


def find_lstm_model_path_prefix_for_ticket(ticket: str) -> Optional[str]:
    print(f"[FIND_MODEL] Buscando modelo para el ticket: '{ticket}'")
    print(f"[FIND_MODEL] Directorio de modelos LSTM: {MODEL_DIR_LSTM}")
    specific_prefix = get_default_lstm_model_path_prefix(ticket)
    specific_model_file_keras = specific_prefix + ".keras"
    print(f"[FIND_MODEL] Comprobando ruta específica: {specific_model_file_keras}")
    specific_exists = os.path.exists(specific_model_file_keras)
    print(f"[FIND_MODEL] ¿Existe el modelo específico? {specific_exists}")
    if specific_exists:
        print(f"[FIND_MODEL] Modelo específico encontrado: {specific_prefix}")
        return specific_prefix
    generic_prefix = get_generic_lstm_model_path_prefix()
    generic_model_file_keras = generic_prefix + ".keras"
    print(f"[FIND_MODEL] Comprobando ruta genérica: {generic_model_file_keras}")
    generic_exists = os.path.exists(generic_model_file_keras)
    print(f"[FIND_MODEL] ¿Existe el modelo genérico? {generic_exists}")
    if generic_exists:
        print(f"[FIND_MODEL] Modelo genérico encontrado: {generic_prefix}")
        return generic_prefix
    glob_pattern = os.path.join(MODEL_DIR_LSTM, "lstm_model_*.keras")
    print(f"[FIND_MODEL] Buscando cualquier modelo con el patrón: {glob_pattern}")
    available_models_keras = glob.glob(glob_pattern)
    print(f"[FIND_MODEL] Modelos encontrados por glob: {available_models_keras}")
    if available_models_keras:
        first_available_prefix = os.path.splitext(available_models_keras[0])[0]
        print(f"[FIND_MODEL] Usando primer modelo disponible como fallback: {first_available_prefix}")
        return first_available_prefix
    print(f"[FIND_MODEL] No se encontró ningún modelo para el ticket '{ticket}'.")
    return None


def load_lstm_model_from_cache_or_disk(model_path_prefix: str) -> TimeSeriesLSTMModel:
    if model_path_prefix in loaded_lstm_models:
        print(f"Cargando modelo LSTM '{os.path.basename(model_path_prefix)}' desde caché.")
        return loaded_lstm_models[model_path_prefix]
    try:
        print(
            f"Cargando modelo LSTM '{os.path.basename(model_path_prefix)}' desde disco (prefijo: {model_path_prefix}).")
        model = TimeSeriesLSTMModel.load_model(model_path_prefix)
        loaded_lstm_models[model_path_prefix] = model
        return model
    except Exception as e:
        import traceback
        print(f"Traceback de error cargando modelo: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error cargando modelo LSTM desde '{model_path_prefix}': {str(e)}")


def load_stock_data_api(ticket: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        data = load_data(ticker=ticket, start_date=start_date, end_date=end_date)
        if data.empty:
            raise HTTPException(status_code=404,
                                detail=f"No se encontraron datos para el ticket {ticket} en el rango {start_date} - {end_date}")
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error descargando datos para {ticket}: {str(e)}")


@app.get("/")
async def read_root(): return {"message": "Servicio de Modelo LSTM para Series Temporales"}


@app.post("/train", response_model=TrainResponseLSTM)
async def train_model_api(request: TrainRequestLSTM):
    try:
        print(f"Solicitud de entrenamiento recibida para el ticket: {request.ticket}")
        data_df = load_stock_data_api(request.ticket, request.start_date, request.end_date)
        save_prefix = request.save_model_path_prefix
        if not save_prefix:
            save_prefix = get_default_lstm_model_path_prefix(request.ticket)
        elif not os.path.isabs(save_prefix):
            save_prefix = os.path.join(MODEL_DIR_LSTM, os.path.basename(save_prefix))
        save_prefix_dir = os.path.dirname(save_prefix)
        if save_prefix_dir and not os.path.exists(save_prefix_dir):
            os.makedirs(save_prefix_dir, exist_ok=True)
            print(f"Directorio creado para el prefijo de guardado: {save_prefix_dir}")
        min_rows_needed = request.n_lags + 60 + 20
        if len(data_df) < min_rows_needed:
            raise HTTPException(status_code=400,
                                detail=f"Datos históricos insuficientes para {request.ticket}. Necesarios: {min_rows_needed}, obtenidos: {len(data_df)}.")
        print(f"Iniciando train_lstm_model para {request.ticket}. Guardando en prefijo: {save_prefix}")
        model, feature_names, residuals, residual_dates = train_lstm_model(
            data=data_df, target_col=request.target_col, n_lags=request.n_lags,
            train_size_ratio=request.train_size_ratio, save_model_path_prefix=save_prefix,
            epochs=request.epochs, batch_size=request.batch_size,
            use_hyperparameter_optimization=request.use_hyperparameter_optimization,
            hp_strategy=request.hp_strategy, hp_max_trials=request.hp_max_trials,
            hp_epochs_per_trial=request.hp_epochs_per_trial, hp_project_name=request.hp_project_name,
            initial_lstm_units=request.initial_lstm_units, initial_lstm_layers=request.initial_lstm_layers,
            initial_dropout_rate=request.initial_dropout_rate, initial_learning_rate=request.initial_learning_rate)
        print(f"Entrenamiento completado para {request.ticket}. Métricas: {model.metrics}")
        loaded_lstm_models[save_prefix] = model
        return TrainResponseLSTM(status="success", message=f"Modelo LSTM entrenado exitosamente para {request.ticket}",
                                 metrics=model.metrics, best_hyperparameters=model.best_hyperparameters,
                                 feature_names=feature_names,
                                 residuals=residuals.tolist() if residuals is not None and residuals.size > 0 else [],
                                 residual_dates=residual_dates if residual_dates else [],
                                 model_path_prefix=os.path.basename(save_prefix))
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Error en datos/parámetros de entrenamiento: {str(ve)}")
    except Exception as e:
        import traceback; print(traceback.format_exc()); raise HTTPException(status_code=500,
                                                                             detail=f"Error entrenando modelo LSTM: {str(e)}")


@app.get("/predict", response_model=PredictionResponseLSTM)
async def predict_api(ticket: str = Query("NU"), forecast_horizon: int = Query(10),
                      target_col: str = Query("Close"), history_days: int = Query(365 * 2)):
    try:
        print(f"Solicitud de predicción recibida para ticket: {ticket}")
        model_path_prefix = find_lstm_model_path_prefix_for_ticket(ticket)
        if model_path_prefix is None:
            raise HTTPException(status_code=404, detail=f"No se encontró modelo LSTM para {ticket}. Entrenar primero.")
        model = load_lstm_model_from_cache_or_disk(model_path_prefix)
        print(f"Modelo LSTM cargado desde prefijo: {model_path_prefix}")
        training_end_date_str = None;
        metadata_file = model_path_prefix + "_metadata.json"
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f: metadata = json.load(f); training_end_date_str = metadata.get(
                "training_end_date")
        end_date_hist = datetime.now()
        if training_end_date_str:
            try:
                end_date_hist = datetime.strptime(training_end_date_str, "%Y-%m-%d")
            except ValueError:
                print(
                    f"Advertencia: Formato de fecha inválido ('{training_end_date_str}') en metadatos. Usando fecha actual.")
        else:
            print(
                "Advertencia: training_end_date no encontrado en metadatos. Usando fecha actual para datos históricos.")
        start_date_hist = end_date_hist - timedelta(days=max(history_days, model.n_lags + 250))
        print(
            f"Cargando datos históricos para {ticket} de {start_date_hist.strftime('%Y-%m-%d')} a {end_date_hist.strftime('%Y-%m-%d')}")
        historical_data_df = load_stock_data_api(ticket, start_date_hist.strftime('%Y-%m-%d'),
                                                 end_date_hist.strftime('%Y-%m-%d'))
        if len(historical_data_df) < model.n_lags:
            raise HTTPException(status_code=400,
                                detail=f"Datos históricos insuficientes ({len(historical_data_df)} filas) para n_lags ({model.n_lags}).")
        print(f"Iniciando forecast_future_prices para {ticket}...")
        forecast_values, _, _ = forecast_future_prices(model=model, historical_data_df=historical_data_df,
                                                       forecast_horizon=forecast_horizon, target_col=target_col)
        if forecast_values is None: raise HTTPException(status_code=500,
                                                        detail="Función de pronóstico no devolvió valores.")

        last_historical_date = historical_data_df.index[-1]
        forecast_dates = []

        if isinstance(last_historical_date, pd.Timestamp):
            current_date_iterator = last_historical_date
            # Intentar obtener la frecuencia del índice histórico
            date_offset = None
            # Primero, intentar usar la frecuencia explícita del índice si existe
            if historical_data_df.index.freq:
                date_offset = historical_data_df.index.freq
            # Si no, intentar inferirla
            elif len(historical_data_df.index) >= 2:
                inferred_freq_str = pd.infer_freq(historical_data_df.index)
                if inferred_freq_str:
                    try:
                        date_offset = pd.tseries.frequencies.to_offset(inferred_freq_str)
                    except ValueError:
                        print(
                            f"Advertencia: No se pudo convertir la frecuencia inferida '{inferred_freq_str}' a DateOffset. Usando timedelta(days=1).")
                        date_offset = timedelta(days=1)
                else:  # Si no se puede inferir
                    print("Advertencia: No se pudo inferir la frecuencia del índice. Usando timedelta(days=1).")
                    date_offset = timedelta(days=1)
            else:  # Si no hay suficientes datos para inferir
                print(
                    "Advertencia: No hay suficientes datos en el índice para inferir frecuencia. Usando timedelta(days=1).")
                date_offset = timedelta(days=1)

            for _ in range(forecast_horizon):
                current_date_iterator = current_date_iterator + date_offset
                forecast_dates.append(current_date_iterator.strftime('%Y-%m-%d'))
        else:  # Para índices no-fecha (numéricos)
            current_idx_iterator = last_historical_date
            for i in range(forecast_horizon):
                current_idx_iterator += 1  # Asumir incremento de 1 para índices numéricos
                forecast_dates.append(f"Periodo_{current_idx_iterator}")

        predictions_output = [{"date": dt, "prediction": float(val)} for dt, val in
                              zip(forecast_dates, forecast_values)]
        historical_to_return = historical_data_df.iloc[-history_days:]
        hist_dates_out = historical_to_return.index.strftime('%Y-%m-%d').tolist() if isinstance(
            historical_to_return.index, pd.DatetimeIndex) else historical_to_return.index.astype(str).tolist()
        hist_vals_out = historical_to_return[target_col].tolist()
        model_info = "Modelo LSTM específico para el ticket" if ticket.upper() in model_path_prefix.upper() else "Modelo LSTM genérico"
        return PredictionResponseLSTM(status="success", ticker=ticket, target_column=target_col,
                                      forecast_horizon=forecast_horizon,
                                      historical_dates=hist_dates_out, historical_values=hist_vals_out,
                                      predictions=predictions_output,
                                      last_actual_date=historical_data_df.index[-1].strftime('%Y-%m-%d') if isinstance(
                                          historical_data_df.index[-1], pd.Timestamp) else str(
                                          historical_data_df.index[-1]),
                                      last_actual_value=float(historical_data_df[target_col].iloc[-1]),
                                      model_used=os.path.basename(model_path_prefix), model_info=model_info)
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Error en datos/parámetros de predicción: {str(ve)}")
    except Exception as e:
        import traceback; print(traceback.format_exc()); raise HTTPException(status_code=500,
                                                                             detail=f"Error realizando predicción LSTM: {str(e)}")


@app.get("/models")
async def list_lstm_models():
    try:
        keras_files = glob.glob(os.path.join(MODEL_DIR_LSTM, "lstm_model_*.keras"))
        models_info = []
        for model_file_keras in keras_files:
            prefix = model_file_keras.replace(".keras", "")
            model_name = os.path.basename(prefix)
            meta_file = prefix + "_metadata.json"
            pkl_file = prefix + "_scalers_params.pkl"
            size_mb = 0
            if os.path.exists(model_file_keras): size_mb += os.path.getsize(model_file_keras)
            if os.path.exists(pkl_file): size_mb += os.path.getsize(pkl_file)
            size_mb = round(size_mb / (1024 * 1024), 2)
            metadata_content = None
            if os.path.exists(meta_file):
                try:
                    with open(meta_file, 'r') as f:
                        metadata_content = json.load(f)
                except Exception:
                    metadata_content = {"error": "No se pudieron cargar los metadatos"}
            else:
                metadata_content = {"error": "Archivo de metadatos no encontrado"}
            models_info.append(
                {"name": model_name, "path_prefix": prefix, "metadata": metadata_content, "size_mb": size_mb})
        return {"total_models": len(models_info), "models": models_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listando modelos LSTM: {str(e)}")


@app.get("/health")
async def health_check(): return {"status": "Ok", "message": "Servicio LSTM está operativo."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
