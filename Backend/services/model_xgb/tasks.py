"""
Celery Tasks for XGBoost Model Service
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
from .xgb_model import XGBoostModel

# Importar utilidades desde el path correcto en Docker
try:
    # Primero intentar desde utils (Docker path)
    from utils import scale_data, split_data, feature_engineering, add_lags
except ImportError:
    try:
        # Fallback - imports desde utils.preprocessing
        from utils.preprocessing import (
            split_data_universal as split_data,
            scale_data_universal as scale_data
        )
        from utils import feature_engineering, add_lags
    except ImportError:
        try:
            # Fallback para desarrollo local con Backend prefix
            from Backend.utils import scale_data, split_data, feature_engineering, add_lags
        except ImportError:
            print("WARNING: Could not import preprocessing utils. Training may fail.")
            scale_data = split_data = feature_engineering = add_lags = None


@celery_app.task(
    bind=True,
    name="train_xgb_model_task",
    max_retries=2,
    soft_time_limit=3600,  # 1 hora máximo
    time_limit=3660,
    queue="xgb_queue"
)
def train_xgb_model_task(
    self,
    ticker: str,
    historical_data: list,
    data_columns: list,
    data_index: list,
    model_params: dict = None,
    save_model_path: str = None
):
    """
    Tarea Celery para entrenar el modelo XGBoost de forma asíncrona.
    
    Parameters:
    - ticker: Símbolo del ticker
    - historical_data: Lista de listas con los datos (limitada para evitar payload excesivo)
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
                'message': 'Iniciando entrenamiento XGBoost...'
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
        
        n_lags = model_params.get('n_lags', 10)
        target_col = model_params.get('target_col', 'Close')
        train_size = model_params.get('train_size', 0.8)
        optimize = model_params.get('optimize_hyperparameters', True)
        
        # Actualizar progreso
        self.update_state(
            state='PROGRESS',
            meta={
                'ticker': ticker,
                'stage': 'preparing_data',
                'progress': 15,
                'message': 'Preparando datos y características...'
            }
        )
        
        # Verificar que las utilidades están disponibles
        if add_lags is None or split_data is None or scale_data is None:
            raise ImportError("No se pudieron importar las utilidades de preprocessing")
        
        # Preparar datos con lags
        df_with_lags = add_lags(df, target_col=target_col, n_lags=n_lags)
        
        try:
            df_prepared = feature_engineering(df_with_lags)
        except Exception as e:
            print(f"Warning: Feature engineering failed: {e}")
            df_prepared = df_with_lags
        
        # Eliminar filas con NaN (causadas por los lags)
        df_prepared = df_prepared.dropna()
        
        self.update_state(
            state='PROGRESS',
            meta={
                'ticker': ticker,
                'stage': 'splitting_data',
                'progress': 25,
                'message': 'Dividiendo datos en train/test...'
            }
        )
        
        # Dividir datos
        X_train, X_test, y_train, y_test = split_data(
            df_prepared, 
            train_size=train_size, 
            shuffle=False, 
            random_state=42
        )
        
        # Escalar datos
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, feature_scaler, target_scaler = scale_data(
            X_train, X_test, y_train, y_test
        )
        
        self.update_state(
            state='PROGRESS',
            meta={
                'ticker': ticker,
                'stage': 'training',
                'progress': 40,
                'message': 'Entrenando modelo XGBoost...'
            }
        )
        
        # Crear y entrenar modelo
        model = XGBoostModel(n_lags=n_lags)
        model.feature_scaler = feature_scaler
        model.target_scaler = target_scaler
        
        if optimize:
            self.update_state(
                state='PROGRESS',
                meta={
                    'ticker': ticker,
                    'stage': 'optimizing',
                    'progress': 50,
                    'message': 'Optimizando hiperparámetros...'
                }
            )
            model.optimize_hyperparameters(
                X_train_scaled, 
                y_train_scaled.ravel(),
                n_iter=model_params.get('n_iter', 20),
                cv=model_params.get('cv', 3)
            )
        else:
            model.fit(X_train_scaled, y_train_scaled.ravel())
        
        self.update_state(
            state='PROGRESS',
            meta={
                'ticker': ticker,
                'stage': 'evaluating',
                'progress': 80,
                'message': 'Evaluando modelo...'
            }
        )
        
        # Evaluar modelo
        metrics = model.evaluate(X_test_scaled, y_test_scaled)
        
        # Predicciones para visualización (limitadas a las últimas 100)
        y_pred = model.predict(X_test_scaled)
        if model.target_scaler:
            y_pred_original = model.target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            y_test_original = model.target_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
        else:
            y_pred_original = y_pred
            y_test_original = y_test_scaled.flatten()
        
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
            save_model_path = f"models/xgb_model_{ticker}.joblib"
        
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
            "data_points_used": len(df_prepared),
            "predictions": {
                "actual": y_test_original[-max_return_points:].tolist(),
                "predicted": y_pred_original[-max_return_points:].tolist(),
                "dates": X_test.index[-max_return_points:].strftime('%Y-%m-%d').tolist()
            },
            "feature_importances": model.feature_importances_.tolist() if model.feature_importances_ is not None else None
        }
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in train_xgb_model_task: {error_trace}")
        
        return {
            "status": "error",
            "task_id": task_id,
            "ticker": ticker,
            "error": str(e),
            "traceback": error_trace
        }


@celery_app.task(name="get_xgb_task_status")
def get_xgb_task_status(task_id: str):
    """
    Obtiene el estado de una tarea de entrenamiento XGBoost.
    """
    from celery.result import AsyncResult
    result = AsyncResult(task_id, app=celery_app)
    
    return {
        "task_id": task_id,
        "state": result.state,
        "info": result.info if result.info else {}
    }
