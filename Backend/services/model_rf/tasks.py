from .celery_app import celery_app
from .train import train_ts_model
from .main import actual_date_range, load_stock_data, get_default_model_path
from .main import TrainRequest 
import os
import pandas as pd
import numpy as np
from datetime import datetime
from celery import Task

@celery_app.task(bind=True, name="train_rf_model_task", queue="rf_queue")
def train_rf_model_task(self: Task, request_data_dict: dict):
    request_data = TrainRequest(**request_data_dict)
    job_id = self.request.id

    print(f"CELERY WORKER RF: Iniciando entrenamiento para job_id: {job_id}, ticker: {request_data.ticket}")
    self.update_state(
        state='PROGRESS', 
        meta={
            'current_step': 'Iniciando procesamiento', 
            'progress': 5,
            'timestamp': datetime.now().isoformat()
        }
    )

    try:
        actual_start_date, actual_end_date = actual_date_range(
            start_date=request_data.start_date,
            end_date=request_data.end_date,
            training_period=request_data.training_period
        )
        self.update_state(state='PROGRESS', meta={'current_step': 'Cargando datos', 'progress': 20})
        data = load_stock_data(request_data.ticket, actual_start_date, actual_end_date)

        save_path = request_data.save_model_path or get_default_model_path(request_data.ticket)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        min_days_rf = max(request_data.n_lags * 2, 252)
        if len(data) < min_days_rf:
            raise ValueError(f"Datos insuficientes ({len(data)} filas) para {min_days_rf} requeridas.")

        self.update_state(state='PROGRESS', meta={'current_step': 'Entrenando modelo RF', 'progress': 50})
        
        # train_ts_model devuelve: model, feature_names, residuals, residual_dates, acf_values, pacf_values, confint_acf, confint_pacf
        model_obj, feature_names, residuals, residual_dates, acf_vals, pacf_vals, confint_acf, confint_pacf = train_ts_model(
            data=data,
            n_lags=request_data.n_lags,
            target_col=request_data.target_col,
            train_size=request_data.train_size,
            save_model_path=save_path
        )
        self.update_state(state='PROGRESS', meta={'current_step': 'Entrenamiento RF finalizado', 'progress': 90})

        result_payload = {
            "status": "success",
            "message": f"Modelo Random Forest entrenado para {request_data.ticket} con datos de {actual_start_date} a {actual_end_date}",
            "model_type": "Random Forest",
            "metrics": model_obj.metrics if hasattr(model_obj, 'metrics') else "Métricas no disponibles.",
            "features_names": feature_names, # Específico de RF
            "best_params": model_obj.best_params_ if hasattr(model_obj, 'best_params_') else "No disponible", # Específico de RF
            "model_path": os.path.basename(save_path),
            "residuals": residuals.tolist(),
            "residual_dates": [d.strftime("%Y-%m-%d") for d in residual_dates],
            "acf": {"values": acf_vals.tolist(), "confint_lower": confint_acf[:, 0].tolist(), "confint_upper": confint_acf[:, 1].tolist()},
            "pacf": {"values": pacf_vals.tolist(), "confint_lower": confint_pacf[:, 0].tolist(), "confint_upper": confint_pacf[:, 1].tolist()}
        }
        print(f"CELERY WORKER RF: Entrenamiento completado para job_id: {job_id}")
        return result_payload

    except Exception as e:
        print(f"CELERY WORKER RF: Error durante el entrenamiento para job_id: {job_id} - {e}")
        self.update_state(state='FAILURE', meta={'error_type': type(e).__name__, 'error_message': str(e)})
        raise