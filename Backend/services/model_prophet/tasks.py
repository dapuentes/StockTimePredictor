"""
Celery Tasks for Prophet Model Service
Handles asynchronous training operations
"""
import pandas as pd
from datetime import datetime
import os
import traceback
import sys

# Asegurar paths correctos para Docker
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/utils')
sys.path.insert(0, '/app/services_code')

from .celery_app import celery_app
from .prophet_model import ProphetModel


@celery_app.task(
    bind=True,
    name="train_prophet_model_task",
    max_retries=2,
    soft_time_limit=3600,  # 1 hora máximo
    time_limit=3660,
    queue="prophet_queue"
)
def train_prophet_model_task(
    self,
    ticker: str,
    historical_data: list,
    data_columns: list,
    data_index: list,
    model_params: dict = None,
    save_model_path: str = None
):
    """
    Tarea Celery para entrenar el modelo Prophet de forma asíncrona.
    
    Parameters:
    - ticker: Símbolo del ticker
    - historical_data: Lista de listas con los datos
    - data_columns: Nombres de las columnas
    - data_index: Índice temporal
    - model_params: Parámetros del modelo
    - save_model_path: Ruta para guardar el modelo
    
    Returns:
    - Diccionario con resultados del entrenamiento
    """
    task_id = self.request.id
    start_time = datetime.now()
    
    # Validación del tamaño del payload
    MAX_ROWS = 5000
    if len(historical_data) > MAX_ROWS:
        return {
            "status": "error",
            "task_id": task_id,
            "ticker": ticker,
            "error": f"Payload demasiado grande. Máximo {MAX_ROWS} filas permitidas, recibidas: {len(historical_data)}"
        }
    
    try:
        # Actualizar estado a "en progreso"
        self.update_state(
            state='PROGRESS',
            meta={
                'ticker': ticker,
                'stage': 'initializing',
                'progress': 5,
                'message': 'Iniciando entrenamiento Prophet...'
            }
        )
        
        # Reconstruir DataFrame desde los datos serializados
        df = pd.DataFrame(
            historical_data,
            columns=data_columns,
            index=pd.to_datetime(data_index)
        )
        df = df.sort_index()
        
        # Parámetros por defecto
        if model_params is None:
            model_params = {}
        
        target_col = model_params.get('target_col', 'Close')
        train_size = model_params.get('train_size', 0.8)
        optimize = model_params.get('optimize_hyperparameters', False)
        regressor_cols = model_params.get('regressor_cols', None)
        
        # Parámetros del modelo Prophet
        changepoint_prior_scale = model_params.get('changepoint_prior_scale', 0.05)
        seasonality_prior_scale = model_params.get('seasonality_prior_scale', 10.0)
        holidays_prior_scale = model_params.get('holidays_prior_scale', 10.0)
        seasonality_mode = model_params.get('seasonality_mode', 'additive')
        
        # Actualizar progreso
        self.update_state(
            state='PROGRESS',
            meta={
                'ticker': ticker,
                'stage': 'preparing_data',
                'progress': 15,
                'message': 'Preparando datos para Prophet...'
            }
        )
        
        # Crear modelo Prophet
        model = ProphetModel(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            seasonality_mode=seasonality_mode
        )
        
        # Preparar datos en formato Prophet
        prophet_data = model.prepare_data(df, target_col=target_col, regressor_cols=regressor_cols)
        
        # Dividir datos
        split_idx = int(len(prophet_data) * train_size)
        train_data = prophet_data.iloc[:split_idx].copy()
        test_data = prophet_data.iloc[split_idx:].copy()
        
        self.update_state(
            state='PROGRESS',
            meta={
                'ticker': ticker,
                'stage': 'training',
                'progress': 30,
                'message': 'Entrenando modelo Prophet...'
            }
        )
        
        # Entrenar
        if optimize:
            self.update_state(
                state='PROGRESS',
                meta={
                    'ticker': ticker,
                    'stage': 'optimizing',
                    'progress': 50,
                    'message': 'Optimizando hiperparámetros Prophet...'
                }
            )
            model.optimize_hyperparameters(train_data, n_iter=model_params.get('n_iter', 10))
        else:
            model.fit(train_data)
        
        self.update_state(
            state='PROGRESS',
            meta={
                'ticker': ticker,
                'stage': 'evaluating',
                'progress': 75,
                'message': 'Evaluando modelo...'
            }
        )
        
        # Evaluar modelo
        metrics = model.evaluate(test_data)
        
        # Hacer predicciones para visualización
        y_pred = model.predict(test_data)
        y_actual = test_data['y'].values
        dates = test_data['ds'].dt.strftime('%Y-%m-%d').tolist()
        
        # Limitar datos para el payload de respuesta
        max_return_points = 100
        
        self.update_state(
            state='PROGRESS',
            meta={
                'ticker': ticker,
                'stage': 'saving',
                'progress': 90,
                'message': 'Guardando modelo...'
            }
        )
        
        # Guardar modelo
        if save_model_path is None:
            save_model_path = f"models/prophet_model_{ticker}.joblib"
        
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        model.save(save_model_path)
        
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()
        
        return {
            "status": "success",
            "task_id": task_id,
            "ticker": ticker,
            "metrics": metrics,
            "best_params": model.best_params_,
            "model_path": save_model_path,
            "training_duration_seconds": training_duration,
            "data_points_used": len(prophet_data),
            "predictions": {
                "actual": y_actual[-max_return_points:].tolist(),
                "predicted": y_pred[-max_return_points:].tolist(),
                "dates": dates[-max_return_points:]
            }
        }
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in train_prophet_model_task: {error_trace}")
        
        return {
            "status": "error",
            "task_id": task_id,
            "ticker": ticker,
            "error": str(e),
            "traceback": error_trace
        }


@celery_app.task(name="get_prophet_task_status")
def get_prophet_task_status(task_id: str):
    """
    Obtiene el estado de una tarea de entrenamiento Prophet.
    """
    from celery.result import AsyncResult
    result = AsyncResult(task_id, app=celery_app)
    
    return {
        "task_id": task_id,
        "state": result.state,
        "info": result.info if result.info else {}
    }
