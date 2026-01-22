from pathlib import Path
import sys
import os
from datetime import datetime, timedelta
import glob
import json
#current_dir = Path(__file__).parent
#project_dir = current_dir.parent.parent  # Navigate up to the project root
#sys.path.append(str(project_dir))
from services.model_xgb.forecast import forecast_future_prices
from fastapi import FastAPI, HTTPException, Query
import pandas as pd
from pydantic import BaseModel
import joblib
import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
from contextlib import asynccontextmanager
from services.model_prophet.prophet_model import ProphetModel, train_prophet_model
from utils.import_data import load_data
from utils.preprocessing import feature_engineering, split_data, scale_data
import traceback
from sklearn.model_selection import TimeSeriesSplit
import services.model_prophet.prophet_service as prophet_service
from typing import List, Dict, Optional
#
app = FastAPI(
    title="Prophet Model API",
    description="API for Prophet Regression model",
    version="1.0.0",
)

# Define the model path
MODEL_DIR = "services/model_prophet/models"
os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, "model_prophet.joblib")
json_path = os.path.join(MODEL_DIR, "model_prophet.json")

class TrainRequest(BaseModel):
    ticker: str = 'NU'
    start_date: str = '2020-12-10'
    end_date: str = datetime.now().strftime("%Y-%m-%d")
#    n_lags: int = 10 
    target_col: str = 'Close'
    regressor_cols: list = ['Open', 'High', 'Low', 'Volume']
    train_size: float = 0.8
    save_model_path: str = None


# Diccionario global para almacenar el modelo

loaded_models = {}

def get_default_model_path(ticket):
    """Genera la ruta predeterminada para guardar un modelo entrenado"""
    return os.path.join(MODEL_DIR, f"prophet_model{ticket}.joblib")


def get_generic_model_path():
    """ Obtiene la ruta del modelo genérico entrenado previamente """
    return os.path.join(MODEL_DIR, "model_prophet.joblib")


def find_model_for_ticket(ticket):
    """
    Busca el modelo entrenado para un ticket específico
    Retorna la ruta del modelo si se encuentra, de lo contrario None
    """
    specific_model_path = get_default_model_path(ticket)
    if os.path.exists(specific_model_path):
        return specific_model_path
    
    # Si no hay modelo específico, busca un modelo genérico
    generic_model_path = get_generic_model_path()
    if os.path.exists(generic_model_path):
        return generic_model_path
    
    # Busca cualquier modelo en el directorio de modelos como último recurso
    avaliable_models = glob.glob(os.path.join(MODEL_DIR, "model_prophet*.joblib"))
    if avaliable_models:
        return avaliable_models[0]
    return None

def load(model_path):
    """Recupera un modelo de la memoria caché o lo carga desde el disco"""
    if model_path in loaded_models:
        return loaded_models[model_path]
    
    try:
        model = ProphetModel.load(model_path)
        loaded_models[model_path] = model
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

def load_stock_data(ticket, start_date, end_date):
    """Carga los datos de stock para un ticket específico entre las fechas dadas"""
    try:
        data = load_data(ticket, start_date, end_date)
        if data.empty:
            raise HTTPException(status_code=404, detail="No data found for the given ticket and date range.")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")
    
def load_trained_model(path: str) -> ProphetModel:
    return ProphetModel.load(path)

@app.get('/')
def read_root():
    return {"message": "Welcome to the Prophet Model API!"}

@app.get('/train')
def train_model():
    '''
    Endpoint para entrenar el modelo Prophet
    '''
    global model, feature_scaler, target_scaler
    try:
        model = ProphetModel()
        print('Manual training of the model started')
        data = load_data()
        if data.empty:
            raise HTTPException(status_code=404, detail="No data found for the given ticket and date range.")

        #processed_data = feature_engineering(data, n_lags=10, target_col='Close')

        model, metrics, _ = prophet_service.train(
            data=data,
            target_col='Close',
            regressor_cols=['Open','High','Low','Volume'],
            train_size=0.8,
            optimize_hyperparams=True,
            save_model_path=os.path.join(MODEL_DIR, "model_prophet.joblib"),
            plot_results=False,         # Desactivar gráficos en API
            forecast_horizon=None       # No necesitamos forecast aquí
        )

        print("Model training completed.")

        # Guardar el modelo
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_file_path = os.path.join(MODEL_DIR, "model_prophet.joblib")
        model.save(model_file_path)
        print(f"Model saved to {MODEL_DIR}")

        # Metadata
        metadata = {
            "model_type": "Prophet",
            "model_path": model_file_path,
            'best_params': model.best_params_,
            "evaluation_results": metrics,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Convert any numpy types to Python native types for JSON serialization
        metadata_serializable = {}
        for k, v in metadata.items():
            if isinstance(v, dict):
                metadata_serializable[k] = {sub_k: float(sub_v) if isinstance(sub_v, np.float64) else sub_v 
                                            for sub_k, sub_v in v.items()}
            else:
                metadata_serializable[k] = float(v) if isinstance(v, np.float64) else v
        
        with open(json_path, 'w') as f:
            import json
            json.dump(metadata_serializable, f, indent=4)
            
        return {
            "message": "Modelo Prophet entrenado y guardado exitosamente.",
            "evaluation_results": metrics
        }


    except Exception as e:
        print("Error during training:", str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during training: {str(e)}")


# ============== NEW ASYNC TRAINING ENDPOINTS (CELERY) ==============

@app.post("/train", status_code=202)
async def train_model_async(request: TrainRequest):
    """
    Endpoint asíncrono para entrenar el modelo Prophet.
    
    Envía la tarea a Celery y devuelve un task_id para consultar el estado.
    Este es el método recomendado para entrenamiento en producción.
    
    Returns:
        - task_id: ID de la tarea para consultar estado
        - status: "queued"
    """
    import pandas as pd
    try:
        # Cargar datos
        data = load_stock_data(
            request.ticker, 
            request.start_date, 
            request.end_date
        )
        
        if data.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"No data found for ticker {request.ticker}"
            )
        
        # Limitar tamaño del payload para Celery
        MAX_ROWS = 5000
        if len(data) > MAX_ROWS:
            data = data.tail(MAX_ROWS)
            print(f"Warning: Data truncated to last {MAX_ROWS} rows")
        
        # Preparar datos para serialización
        historical_data = data.values.tolist()
        data_columns = data.columns.tolist()
        data_index = data.index.strftime('%Y-%m-%d').tolist()
        
        # Definir ruta del modelo
        save_path = request.save_model_path or get_default_model_path(request.ticker)
        
        # Parámetros del modelo
        model_params = {
            'target_col': request.target_col,
            'train_size': request.train_size,
            'regressor_cols': request.regressor_cols,
            'optimize_hyperparameters': False  # Prophet optimization is slow
        }
        
        # Importar tarea aquí para evitar problemas de importación circular
        from model_prophet.tasks import train_prophet_model_task
        
        # Enviar tarea a Celery
        task = train_prophet_model_task.delay(
            ticker=request.ticker,
            historical_data=historical_data,
            data_columns=data_columns,
            data_index=data_index,
            model_params=model_params,
            save_model_path=save_path
        )
        
        return {
            "status": "queued",
            "task_id": task.id,
            "message": f"Training task for {request.ticker} has been queued",
            "ticker": request.ticker,
            "data_points": len(data)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"Error queuing training task: {traceback_str}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error queuing training task: {str(e)}"
        )


@app.get("/train/status/{task_id}")
async def get_train_status(task_id: str):
    """
    Consulta el estado de una tarea de entrenamiento.
    
    Args:
        task_id: ID de la tarea de Celery
        
    Returns:
        - state: Estado actual (PENDING, PROGRESS, SUCCESS, FAILURE)
        - progress: Porcentaje de progreso (si disponible)
        - result: Resultado final (si completada)
    """
    from celery.result import AsyncResult
    from model_prophet.celery_app import celery_app
    
    result = AsyncResult(task_id, app=celery_app)
    
    response = {
        "task_id": task_id,
        "state": result.state
    }
    
    if result.state == 'PENDING':
        response.update({
            "progress": 0,
            "message": "Task is pending or unknown"
        })
    elif result.state == 'PROGRESS':
        info = result.info or {}
        response.update({
            "progress": info.get('progress', 0),
            "stage": info.get('stage', 'unknown'),
            "message": info.get('message', ''),
            "ticker": info.get('ticker', '')
        })
    elif result.state == 'SUCCESS':
        response.update({
            "progress": 100,
            "result": result.result
        })
    elif result.state == 'FAILURE':
        response.update({
            "progress": 0,
            "error": str(result.info) if result.info else "Unknown error"
        })
    
    return response


@app.get("/train/tasks")
async def list_active_tasks():
    """
    Lista las tareas de entrenamiento activas.
    """
    from model_prophet.celery_app import celery_app
    
    try:
        inspect = celery_app.control.inspect()
        
        active = inspect.active() or {}
        reserved = inspect.reserved() or {}
        scheduled = inspect.scheduled() or {}
        
        return {
            "active_tasks": active,
            "reserved_tasks": reserved,
            "scheduled_tasks": scheduled
        }
    except Exception as e:
        return {
            "error": str(e),
            "message": "Could not retrieve task information. Is Celery running?"
        }


@app.delete("/train/cancel/{task_id}")
async def cancel_training(task_id: str):
    """
    Cancela una tarea de entrenamiento en progreso.
    """
    from celery.result import AsyncResult
    from model_prophet.celery_app import celery_app
    
    result = AsyncResult(task_id, app=celery_app)
    
    if result.state in ['PENDING', 'PROGRESS']:
        result.revoke(terminate=True)
        return {
            "task_id": task_id,
            "status": "cancelled",
            "message": "Training task has been cancelled"
        }
    else:
        return {
            "task_id": task_id,
            "status": result.state,
            "message": f"Task cannot be cancelled. Current state: {result.state}"
        }


# ============== LEGACY SYNC ENDPOINT (for backward compatibility) ==============

@app.get('/train/sync')
def train_model_sync():
    '''
    Endpoint SÍNCRONO para entrenar el modelo Prophet (LEGACY - usar POST /train en producción).
    '''
    global model, feature_scaler, target_scaler
    try:
        model = ProphetModel()
        print('Manual training of the model started (sync)')
        data = load_data()
        if data.empty:
            raise HTTPException(status_code=404, detail="No data found for the given ticket and date range.")

        model, metrics, _ = prophet_service.train(
            data=data,
            target_col='Close',
            regressor_cols=['Open','High','Low','Volume'],
            train_size=0.8,
            optimize_hyperparams=True,
            save_model_path=os.path.join(MODEL_DIR, "model_prophet.joblib"),
            plot_results=False,
            forecast_horizon=None
        )

        print("Model training completed.")
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_file_path = os.path.join(MODEL_DIR, "model_prophet.joblib")
        model.save(model_file_path)

        return {
            "message": "Modelo Prophet entrenado y guardado exitosamente (sync).",
            "evaluation_results": metrics
        }

    except Exception as e:
        print("Error during training:", str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during training: {str(e)}")


######### VERSION PRE MODIFICACIONES ##########

@app.get("/evaluate")
def evaluate_model(train_size: float = Query(0.8, ge=0.1, le=0.99)):
    # 1. carga modelo
    path = get_generic_model_path()
    if not os.path.exists(path):
        raise HTTPException(404, "No hay modelo entrenado.")
    model = load_trained_model(path)

    # 2. carga datos
    data = load_data()  # o usa ticket/start/end si lo defines
    if data.empty:
        raise HTTPException(404, "No hay datos para evaluar.")

    # 3. delega en el servicio
    result = prophet_service.evaluate(model, data, train_size=train_size)
    return result



@app.get("/predict")
def predict(
    ticket: str = Query("NU"),
    horizon: int = Query(10, gt=0),
    target_col: str = Query("Close"),
    regressor_cols: List[str] = Query(['Open','High','Low','Volume'])
):
    # 1) Carga modelo
    path = find_model_for_ticket(ticket)
    if path is None:
        raise HTTPException(404, f"No hay modelo para {ticket}")
    model = load_trained_model(path)

    # 2) Datos históricos
    end = datetime.now()
    start = end - timedelta(days=365*3)
    data = load_data(ticket, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    if data.empty:
        raise HTTPException(404, "No hay datos históricos.")

    # 3) Forecast usando el servicio
    preds = prophet_service.predict(
        model=model,
        data=data,
        forecast_horizon=horizon,
        regressor_cols=regressor_cols,
        target_col=target_col
    )

    return {
        "ticker": ticket,
        "target_column": target_col,
        "forecast_horizon": horizon,
        "predictions": preds,
        "last_actual_date": data.index[-1].strftime("%Y-%m-%d"),
        "last_actual_value": float(data[target_col].iloc[-1])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)


#script to run the server
'''
.venv\Scripts\activate 
python -m services.model_prophet.app

'''

