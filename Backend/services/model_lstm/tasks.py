from .celery_app import celery_app
from .train import train_lstm_model 
from .main import actual_date_range, load_stock_data_helper, get_default_lstm_model_dir # Importa helpers de tu main.py

from .main import BaseTrainRequest, TrainRequestLSTM # Importa tus modelos Pydantic
import os
import pandas as pd
import numpy as np
from datetime import datetime 
from celery import Task
from typing import Dict, Any

@celery_app.task(bind=True, name="train_lstm_model_task")
def train_lstm_model_task(self: Task, request_data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tarea de Celery para entrenar el modelo LSTM en segundo plano.
    """
    request_data = BaseTrainRequest(**request_data_dict)
    request_data_train = TrainRequestLSTM(**request_data_dict)
    job_id = self.request.id

    print(f"CELERY WORKER LSTM: Iniciando entrenamiento para job_id: {job_id}, ticker: {request_data.ticket}")
    self.update_state(
        state='PROGRESS', 
        meta={
            'current_step': 'Iniciando procesamiento', 
            'progress': 5,
            'timestamp': datetime.now().isoformat()
        }
    )

    try:
        actual_start_date = request_data.start_date if request_data.start_date else "2022-01-01"
        actual_end_date = request_data.end_date if request_data.end_date else "2025-01-01"
        
        print(f"🔍 DEBUG: Usando fechas: {actual_start_date} a {actual_end_date}")
        
        self.update_state(state='PROGRESS', meta={'current_step': 'Cargando datos', 'progress': 20})
        data = load_stock_data_helper(request_data.ticket, actual_start_date, actual_end_date)

        save_dir = request_data.save_model_path or get_default_lstm_model_dir(request_data.ticket)
        os.makedirs(save_dir, exist_ok=True)

        min_rows_needed = request_data_train.sequence_length + 30
        if len(data) < min_rows_needed:
            raise ValueError(f"Datos insuficientes ({len(data)} filas) para {min_rows_needed} requeridas.")

        self.update_state(state='PROGRESS', meta={'current_step': 'Entrenando modelo', 'progress': 50})
        
        trained_model, residuals, residual_dates, acf_vals, pacf_vals, confint_acf, confint_pacf = train_lstm_model(
            data=data,
            target_col=request_data.target_col,
            sequence_length=request_data_train.sequence_length,
            n_lags=request_data.n_lags,
            lstm_units=request_data_train.lstm_units,
            dropout_rate=request_data_train.dropout_rate,
            train_size=request_data.train_size,
            epochs=request_data_train.epochs,
            optimize_params=request_data_train.optimize_params,
            save_model_path=save_dir
        )
        
        self.update_state(state='PROGRESS', meta={'current_step': 'Entrenamiento finalizado, generando resultados', 'progress': 90})

        result_payload = {
            "status": "success",
            "message": f"Modelo LSTM entrenado exitosamente para {request_data.ticket} con datos de {actual_start_date} a {actual_end_date}",
            "model_type": "LSTM",
            "metrics": trained_model.metrics if hasattr(trained_model, 'metrics') else "Métricas no disponibles.",
            "model_path": os.path.basename(save_dir),
            "residuals": residuals.tolist(),
            "residual_dates": residual_dates.strftime('%Y-%m-%d').tolist(),
            "acf": {"values": acf_vals.tolist(), "confint_lower": confint_acf[:, 0].tolist(), "confint_upper": confint_acf[:, 1].tolist()},
            "pacf": {"values": pacf_vals.tolist(), "confint_lower": confint_pacf[:, 0].tolist(), "confint_upper": confint_pacf[:, 1].tolist()}
        }
        
        if hasattr(trained_model, 'best_params_') and trained_model.best_params_:
            best_params_serializable = {k: v.item() if isinstance(v, np.generic) else v for k, v in trained_model.best_params_.items()}
            result_payload["best_params"] = best_params_serializable

        print(f"CELERY WORKER LSTM: Entrenamiento completado para job_id: {job_id}")
        return result_payload 

    except Exception as e:
        print(f"CELERY WORKER LSTM: Error durante el entrenamiento para job_id: {job_id} - {e}")
        self.update_state(state='FAILURE', meta={'error_type': type(e).__name__, 'error_message': str(e)})
        raise