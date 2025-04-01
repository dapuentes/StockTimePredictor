from fastapi import FastAPI, HTTPException, Query, Form, Path
import httpx
from datetime import datetime

app = FastAPI(title="API Gateway", version="1.0.0")

# Diccionario para almacenar las rutas de los microservicios
microservices = {
    "model_rf": "http://localhost:8001",       # Microservicio de Random Forest
    "model_lstm": "http://localhost:8002",     # Microservicio de LSTM
    "model_xgboost": "http://localhost:8003",  # Microservicio de XGBoost
    "model_ensemble": "http://localhost:8004", # Microservicio de Ensemble
    "data_import": "http://localhost:8005"    # Microservicio de Importación de Datos
}

@app.get("/")
async def read_root():
    return {"message": "Welcome to the API Gateway"}

@app.post("/train/{model_type}")
async def train_model(
    model_type: str = Path(..., description="Tipo de modelo a entrenar (e.g., 'rf', 'lstm', 'xgboost')"),
    ticket: str = Form("NU"),
    start_date: str = Form("2020-12-10"),
    end_date: str = Form(datetime.now().strftime("%Y-%m-%d")),
    n_lags: int = Form(10),
    target_col: str = Form("Close"),
    train_size: float = Form(0.8),
    save_model_path: str = Form(None)
):
    """"
    Endpoint para entrenar un modelo de machine learning.

    Parámetros:
    - model_type (str): Tipo de modelo a entrenar (e.g., 'rf', 'lstm', 'xgboost').
    - ticket (str): ticket de la acción a importar.
    - start_date (str): Fecha de inicio para la importación de datos.
    - end_date (str): Fecha de fin para la importación de datos.
    - n_lags (int): Número de rezagos a usar en el modelo.
    - target_col (str): Columna objetivo para el modelo.
    - train_size (float): Proporción de datos para entrenamiento.
    - save_model_path (str): Ruta para guardar el modelo entrenado.

    Si no se proporcionan parámetros, se usarán los valores por defecto.
    """

    if model_type.lower() not in microservices:
        raise HTTPException(status_code=400, detail="Invalid model type. Supported types: rf, lstm, xgboost")
    
    # URL del microservicio correspondiente de entrenamiento
    service_url = f"{microservices[model_type.lower()]}/train"

    # Reenviar la petición con los parámetros necesarios
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                service_url,
                json={
                    "ticket": ticket,
                    "start_date": start_date,
                    "end_date": end_date,
                    "n_lags": n_lags,
                    "target_col": target_col,
                    "train_size": train_size,
                    "save_model_path": save_model_path
                }
            )
            response.raise_for_status()  # Lanza un error
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error in sending request to microservice: {e}")
        
    return response.json()


@app.get("/predict/{model_type}")
async def predict(
    model_type: str = Path(..., description="Type of model to use for prediction (e.g., 'rf', 'lstm', 'xgboost')"),
    ticket: str = Query("NU", description="ticket of the stock to import"),
    forecast_horizon: int = Query(10, description="Forecast horizon in days"),
    target_col: str = Query("Close", description="Target column for prediction")
):
    """
    Endpoint para hacer predicciones usando el modelo entrenado.
    Dependiento del parámetro 'model_type' se cargan datos recientes del ticket especificado (por defecto 'NU') y se usa
    un modelo preentrenado en la ruta especificada.
    """    

    if model_type.lower() not in microservices:
        raise HTTPException(status_code=400, detail="Invalid model type. Supported types: rf, lstm, xgboost")
    
    # URL del microservicio correspondiente de predicción
    service_url = f"{microservices[model_type.lower()]}/predict"

    # Parametros para la predicción
    params = {
        "ticket": ticket,
        "forecast_horizon": forecast_horizon,
        "target_col": target_col
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(service_url, params=params)
            response.raise_for_status()  # Lanza un error
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error in sending request to microservice: {e}")
        
    return response.json()

@app.get("/health")
async def health_check():
    """
    Endpoint de verificación de salud para el API Gateway.
    """
    return {"status": "Ok"}

# Crear conexión al localhost
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host = "0.0.0.0", port = 8000)
