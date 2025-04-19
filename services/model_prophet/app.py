from pathlib import Path
import sys
import os
from datetime import datetime, timedelta
import glob

current_dir = Path(__file__).parent
project_dir = current_dir.parent.parent  # Navigate up to the project root
sys.path.append(str(project_dir))
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
    n_lags: int = 10 
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
    return os.path.join(MODEL_DIR, "prophet_model.joblib")


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
    avaliable_models = glob.glob(os.path.join(MODEL_DIR, "prophet_model*.joblib"))
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
        print('Manual training of the model started')
        data = load_data()
        print('Data loaded successfully, Shape:', data.shape)

    processed_data = feature_engineering(data, n_lags=10, target_col='Close')

    print('Feature engineering completed, Shape:', processed_data.shape)


    # Split data
    X_train, X_test, y_train, y_test = split_data(processed_data, train_size=0.8)
    print("Data split completed. Training data shape:", X_train.shape)
    
    # Escalar los datos
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, feature_scaler, target_scaler = scale_data(
        X_train, X_test, 
        y_train.values.reshape(-1, 1), 
        y_test.values.reshape(-1, 1)
    )
    print("Data scaling completed.")

    # Train model
    model = train_prophet_model(
        X_train_scaled, y_train_scaled, 
        X_test_scaled, y_test_scaled, 
        feature_scaler=feature_scaler, 
        target_scaler=target_scaler
    )