from .celery_app import celery_app
from .train_xgb import train_xgb_model
from .main_xgb import load_stock_data_helper, get_default_xgb_model_path_prefix, TrainRequestXGB

import os
import numpy as np
from datetime import datetime
from celery import Task
from typing import Dict, Any, Optional
from pydantic import BaseModel

class BaseTrainRequest(BaseModel):
    """
    Represents a base request model for training operations.

    This class is designed to standardize the input parameters required for 
    initiating training processes in various machine learning workflows. It 
    facilitates capturing metadata, preprocessing configurations, and model 
    storage paths. 

    Attributes:
        ticket (str): Identifier or code associated with the training request (stock ticker).
        start_date (str): The starting date for data selection.
        end_date (str): The ending date for data selection.
        n_lags (int): Number of time lags to include in preprocessing.
        target_col (str): Name of the target column for predictions.
        train_size (float): Proportion of the dataset to be used for training.
        save_model_path (Optional[str]): Path where the trained model should 
            be saved. Can be None if saving is not required.
    """
    ticket: str = "NU"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    training_period: Optional[str] = None  # Utilizado para definir rango especifico (ejm: 1 año, 3 años, todo el histórico)
    n_lags: int = 10  
    target_col: str = "Close"
    train_size: float = 0.8
    save_model_path: Optional[str] = None

@celery_app.task(bind=True, name="train_xgb_model_task")
def train_xgb_model_task(self: Task, request_data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tarea de Celery para entrenar el modelo XGBoost en segundo plano.
    
    Args:
        self: Instancia de la tarea de Celery (auto-inyectada por bind=True)
        request_data_dict: Diccionario con los parámetros de entrenamiento
        
    Returns:
        Dict con los resultados del entrenamiento, métricas, residuales y paths del modelo
    """
    request_data = BaseTrainRequest(**request_data_dict)
    request_data_train = TrainRequestXGB(**request_data_dict)
    job_id = self.request.id

    print(f"CELERY WORKER XGB: Iniciando entrenamiento para job_id: {job_id}, ticker: {request_data.ticket}")
    self.update_state(
        state='PROGRESS', 
        meta={
            'current_step': 'Iniciando procesamiento XGBoost', 
            'progress': 5,
            'timestamp': datetime.now().isoformat()
        }
    )

    try:
        # Determinar fechas de entrenamiento
        actual_start_date = request_data.start_date if request_data.start_date else "2022-01-01"
        actual_end_date = request_data.end_date if request_data.end_date else datetime.now().strftime("%Y-%m-%d")
        
        print(f"🔍 DEBUG XGB: Usando fechas: {actual_start_date} a {actual_end_date}")
        
        self.update_state(
            state='PROGRESS', 
            meta={
                'current_step': 'Cargando datos de mercado', 
                'progress': 15,
                'timestamp': datetime.now().isoformat()
            }
        )
        
        # Cargar datos usando el helper del módulo XGB
        data = load_stock_data_helper(request_data.ticket, actual_start_date, actual_end_date)

        # Determinar ruta de guardado usando la lógica específica de XGBoost
        save_path_prefix = request_data.save_model_path or get_default_xgb_model_path_prefix(request_data.ticket)
        save_dir = os.path.dirname(save_path_prefix)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Validar que tenemos suficientes datos para XGBoost
        min_rows_needed = request_data.n_lags + 50  # XGBoost necesita menos datos que LSTM
        if len(data) < min_rows_needed:
            raise ValueError(f"Datos insuficientes ({len(data)} filas) para entrenar XGBoost. Se requieren al menos {min_rows_needed} filas.")

        self.update_state(
            state='PROGRESS', 
            meta={
                'current_step': 'Preparando datos y entrenando modelo XGBoost', 
                'progress': 40,
                'timestamp': datetime.now().isoformat()
            }
        )
        
        # Entrenar el modelo XGBoost usando los parámetros específicos
        trained_model_obj, feature_names, residuals, residual_dates, acf_vals, pacf_vals, confint_acf, confint_pacf = train_xgb_model(
            data=data,
            target_col=request_data.target_col,
            n_lags=request_data.n_lags,
            train_size_ratio=request_data_train.train_size_ratio,
            save_model_path_prefix=save_path_prefix
        )
        
        self.update_state(
            state='PROGRESS', 
            meta={
                'current_step': 'Entrenamiento XGBoost finalizado, procesando resultados', 
                'progress': 85,
                'timestamp': datetime.now().isoformat()
            }
        )

        # Preparar el payload de respuesta con características específicas de XGBoost
        result_payload = {
            "status": "success",
            "message": f"Modelo XGBoost entrenado exitosamente para {request_data.ticket} con datos de {actual_start_date} a {actual_end_date}",
            "model_type": "XGBoost",
            "metrics": trained_model_obj.metrics if hasattr(trained_model_obj, 'metrics') else "Métricas no disponibles.",
            "model_path_prefix": os.path.basename(save_path_prefix),
            "feature_names_used": feature_names if feature_names else [],
            "n_lags_used": request_data.n_lags,
            "train_size_ratio": request_data_train.train_size_ratio,
            "data_shape": {"total_rows": len(data), "columns": list(data.columns)},
            "timestamp": datetime.now().isoformat()
        }
        
        # Agregar información de residuales si está disponible
        if residuals is not None and len(residuals) > 0:
            result_payload.update({
                "residuals": residuals.tolist() if hasattr(residuals, 'tolist') else list(residuals),
                "residual_dates": residual_dates.strftime('%Y-%m-%d').tolist() if hasattr(residual_dates, 'strftime') else [str(d) for d in residual_dates],
                "residuals_length": len(residuals)
            })
        
        # Agregar información de correlación autocorrelación si está disponible
        if acf_vals is not None and len(acf_vals) > 0:
            result_payload.update({
                "acf": {
                    "values": acf_vals.tolist() if hasattr(acf_vals, 'tolist') else list(acf_vals),
                    "confint_lower": confint_acf[:, 0].tolist() if confint_acf is not None else [],
                    "confint_upper": confint_acf[:, 1].tolist() if confint_acf is not None else []
                },
                "acf_values_length": len(acf_vals)
            })
        
        if pacf_vals is not None and len(pacf_vals) > 0:
            result_payload.update({
                "pacf": {
                    "values": pacf_vals.tolist() if hasattr(pacf_vals, 'tolist') else list(pacf_vals),
                    "confint_lower": confint_pacf[:, 0].tolist() if confint_pacf is not None else [],
                    "confint_upper": confint_pacf[:, 1].tolist() if confint_pacf is not None else []
                },
                "pacf_values_length": len(pacf_vals)
            })

        # Agregar mejores hiperparámetros si están disponibles (específico de XGBoost)
        if hasattr(trained_model_obj, 'best_params_') and trained_model_obj.best_params_:
            best_params_serializable = {}
            for k, v in trained_model_obj.best_params_.items():
                if isinstance(v, np.generic):
                    best_params_serializable[k] = v.item()
                elif isinstance(v, (np.ndarray, list)):
                    best_params_serializable[k] = [item.item() if hasattr(item, 'item') else item for item in v]
                else:
                    best_params_serializable[k] = v
            result_payload["best_hyperparameters"] = best_params_serializable

        # Agregar información específica del modelo XGBoost si está disponible
        if hasattr(trained_model_obj, 'feature_importances_'):
            try:
                importances = trained_model_obj.feature_importances_
                if importances is not None and len(importances) > 0:
                    # Combinar nombres de características con sus importancias
                    feature_importance_dict = {}
                    if feature_names and len(feature_names) == len(importances):
                        for i, importance in enumerate(importances):
                            feature_name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                            feature_importance_dict[feature_name] = float(importance)
                    else:
                        for i, importance in enumerate(importances):
                            feature_importance_dict[f"feature_{i}"] = float(importance)
                    
                    result_payload["feature_importances"] = feature_importance_dict
            except Exception as e:
                print(f"Warning: No se pudieron extraer las importancias de características: {e}")

        self.update_state(
            state='PROGRESS', 
            meta={
                'current_step': 'Finalizando y guardando resultados', 
                'progress': 95,
                'timestamp': datetime.now().isoformat()
            }
        )

        print(f"CELERY WORKER XGB: Entrenamiento completado exitosamente para job_id: {job_id}")
        return result_payload 

    except Exception as e:
        error_message = f"Error durante el entrenamiento XGBoost para job_id: {job_id} - {str(e)}"
        print(f"CELERY WORKER XGB: {error_message}")
        
        # Log detallado del error para debugging
        import traceback
        traceback.print_exc()
        
        self.update_state(
            state='FAILURE', 
            meta={
                'error_type': type(e).__name__, 
                'error_message': str(e),
                'timestamp': datetime.now().isoformat(),
                'job_id': job_id,
                'ticker': request_data.ticket if 'request_data' in locals() else 'unknown'
            }
        )
        raise
