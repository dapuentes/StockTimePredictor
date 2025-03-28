from fastapi import FastAPI, HTTPException, Query
import pandas as pd 
from pydantic import BaseModel # Para validar los datos de entrada
import os
from models.random_forest import train_model, predict_future, evaluate_model # Importar las funciones del modelo
from apscheduler.schedulers.background import BackgroundScheduler # Para programar tareas en segundo plano

app = FastAPI(
    title="API de Random Forest",
    description="API para entrenar un modelo de predicción de precios de acciones y hacer predicciones.",
    version="0.1",
)

DATA_PATH = "data/NU_Historical_Data.csv"  # Ruta del archivo CSV con los datos de acciones
MODEL_PATH = "models/model.joblib"  # Ruta donde se guardará el modelo entrenado

# Convertir Close a float
data = pd.read_csv(DATA_PATH, parse_dates=["Date"])
data["Close"] = pd.to_numeric(data["Close"], errors='coerce') # Convertir Close a float
data["Date"] = pd.to_datetime(data["Date"], format="%m/%d/%Y", errors='coerce') # Convertir Date a datetime

# Configuración del programador de tareas
scheduler = BackgroundScheduler()

def scheduled_training():
    """
    Función para programar el entrenamiento del modelo
    """
    if os.path.exists(MODEL_PATH):
        try:
            result = train_model(DATA_PATH, "Close")
            print(f"Entrenamiento programado completado: {result}")
        except Exception as e:
            print(f"Error al entrenar el modelo: {e}")
    else:
        print("No se ha encontrado la base de datos para el entrenamiento.")

# Iniciar el programador de tareas
scheduler.add_job(scheduled_training, 'cron', hour=2, minute=0)  # Entrenamiento diario a las 2:00 AM
scheduler.start()

# Endpoint raíz
@app.get("/")
def read_root():
    """
    Endpoint raíz que muestra que la API está en funcionamiento
    """

    return {"message": "API de pronóstico Random Forest en funcionamiento"} 

# Evento de startup por si el modelo no está entrenado
@app.on_event("startup")
def startup_event():
    """
    Evento de inicio de la aplicación
    """
    if not os.path.exists(MODEL_PATH):
        print("Modelo no encontrado, iniciando entrenamiento...")
        # Entrenar el modelo al iniciar la aplicación
        try:
            result = train_model(DATA_PATH)
            print(f"Entrenamiento inicial completado: {result}")
        except Exception as e:
            print(f"Error al entrenar en startup: {e}")

# Endpoint para entrenar el modelo
@app.get("/train")
def train():
    """
    Endpoint para entrenar el modelo
    """
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=404, detail="Archivo de datos no encontrado")
    try:
        result = train_model(DATA_PATH)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al entrenar el modelo: {e}")

# Petición para predecir el futuro
class PredictionRequest(BaseModel):
    n_steps: int = Query(1, ge=1, le=50, description="Número de pasos a predecir (1-10)")

# Endpoint para predecir a futuro
@app.post("/predict")
def get_predict(request: PredictionRequest):
    """
    Endpoint para predecir el futuro usando el modelo entrenado
    """
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    try:
        result = predict_future(DATA_PATH, n_steps=request.n_steps)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir: {e}")

@app.get("/evaluate")
def evaluate(days_out: int = 30):
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=404, detail="Archivo de datos no encontrado")
    try:
        result = evaluate_model(DATA_PATH, days_out=days_out)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al evaluar el modelo: {e}")
