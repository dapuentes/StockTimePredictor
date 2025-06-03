import os
import httpx
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Query, Path, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="API Gateway", version="1.1.0")

# Configuración de CORS para permitir solicitudes desde cualquier origen
origins = [
    "http://localhost:3000"  # Frontend local
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Diccionario para almacenar las rutas de los microservicios
microservices = {
    "rf": os.getenv("RF_SERVICE_URL", "http://localhost:8001"),  # Microservicio de Random Forest
    "lstm": os.getenv("LSTM_SERVICE_URL", "http://localhost:8002"),  # Microservicio de LSTM
    "xgboost": os.getenv("XGBOOST_SERVICE_URL", "http://localhost:8003"),  # Microservicio de XGBoost
    "prophet": os.getenv("PROPHET_SERVICE_URL", "http://localhost:8004"),  # Microservicio de Prophet
}


# Nueva clase para manejar la solicitud de entrenamiento
class TrainRequest(BaseModel):
    ticket: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    training_period: Optional[int] = None
    n_lags: Optional[int] = 10
    target_col: Optional[str] = "Close"
    train_size: Optional[float] = 0.8
    save_model_path: Optional[str] = None

    # Nuevos campos para el entrenamiento de cada modelo en particular (revisar mas adelante)
    # LSTM
    sequence_length: Optional[int] = None
    epochs: Optional[int] = None
    lstm_units: Optional[int] = None
    dropout_rate: Optional[float] = None
    optimize_params: Optional[bool] = None

    # Random Forest
    rf_n_estimators: Optional[int] = None # Prefijo 'rf_' para evitar colisiones si XGBoost también tiene n_estimators
    rf_max_depth: Optional[int] = None
    rf_min_samples_split: Optional[int] = None
    rf_min_samples_leaf: Optional[int] = None
    rf_max_features: Optional[str] = None # Puede ser "sqrt", "log2" o un float/int
    rf_cv_folds: Optional[int] = None # Si quieres controlar los folds del CV de RF desde el front

@app.get("/")
async def read_root():
    return {"message": "Welcome to the API Gateway"}


@app.post("/train/{model_type}", status_code=202) # Devolver 202 por defecto para el inicio de una tarea
async def train_model(
        model_type: str = Path(..., description="Tipo de modelo a entrenar (e.g., 'rf', 'lstm', 'xgboost')"),
        train_data: TrainRequest = Body(..., description="Datos de entrenamiento para el modelo")
):
    """
    Trains a machine learning model by forwarding the request to the appropriate microservice.
        This endpoint acts as a gateway that routes training requests to the corresponding
        microservice based on the model type. It handles the asynchronous communication
        with the training service and manages error responses.
        Args:
            model_type (str): The type of model to train. Must be one of the supported
                             types: 'rf' (Random Forest), 'lstm', or 'xgboost'.
            train_data (TrainRequest): The training data and configuration parameters
                                      required for model training.
        Returns:
            dict: Response from the microservice containing job information, typically
                  including a job_id for tracking the training process.
        Raises:
            HTTPException: 
                - 400: If model_type is not supported
                - 502/503: If the target microservice is unreachable or returns an error
                - 504: If the request times out waiting for microservice response
                - 500: For unexpected internal errors
         """

    if model_type.lower() not in microservices:
        raise HTTPException(status_code=400, detail="Invalid model type. Supported types: rf, lstm, xgboost")

    # URL del microservicio correspondiente de entrenamiento
    service_url = f"{microservices[model_type.lower()]}/train"

    # Parámetros de la solicitud
    payload_dict = train_data.model_dump(exclude_none=True) # Excluir campos con valor None (para el frontend)

    print(f"API Gateway: Reenviando solicitud de entrenamiento para {model_type.lower()} a {service_url} con payload: {payload_dict}")

    # Reenviar la petición con los parámetros necesarios
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(service_url, json=payload_dict, timeout=30.0) 

            if response.status_code == 202:
                response_data = response.json()
                print(f"API Gateway: Trabajo de entrenamiento para {model_type.lower()} encolado. Job ID: {response_data.get('job_id')}")
                return response_data 
            else:
                print(f"API Gateway: Error del microservicio {model_type.lower()} al encolar. Status: {response.status_code}, Body: {response.text}")
                try:
                    error_detail = response.json().get("detail", response.text)
                except Exception:
                    error_detail = response.text
                raise HTTPException(status_code=response.status_code, detail=error_detail)

        except httpx.ReadTimeout:
            print(f"API Gateway: Timeout esperando la confirmación de encolamiento de {service_url}")
            raise HTTPException(status_code=504, detail=f"Gateway timeout esperando que el servicio {model_type.lower()} encole la tarea.")
        except httpx.RequestError as exc:
            print(f"API Gateway: No se pudo conectar a {service_url}. Error: {exc}")
            raise HTTPException(status_code=503, detail=f"No se puede conectar al servicio de {model_type.lower()}.")
        except Exception as e:
            print(f"API Gateway: Error inesperado al procesar /train/{model_type.lower()}: {type(e).__name__} - {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail="Error interno del Gateway al iniciar el entrenamiento.")
        
@app.get("/train_status/{model_type}/{job_id}")
async def train_status(
        model_type: str = Path(..., description="Tipo de modelo para verificar el estado del entrenamiento (e.g., 'rf', 'lstm', 'xgboost')"),
        job_id: str = Path(..., description="ID del trabajo de entrenamiento para consultar su estado")
):
    
    if model_type.lower() not in microservices:
        raise HTTPException(status_code=400, detail=f"Invalid model type: {model_type}. Supported types: {list(microservices.keys())}")

    # El endpoint en el microservicio es /training_status/{job_id}
    service_url = f"{microservices[model_type.lower()]}/training_status/{job_id}"

    print(f"API Gateway: Consultando estado para job_id {job_id} del modelo {model_type.lower()} en {service_url}")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(service_url, timeout=30.0)
            response.raise_for_status() 
            status_data = response.json()
            print(f"API Gateway: Estado recibido para job_id {job_id}: {status_data}")
            return status_data

        except httpx.ReadTimeout:
            print(f"API Gateway: Timeout consultando estado de {service_url}")
            raise HTTPException(status_code=504, detail=f"Gateway timeout consultando estado del job en el servicio {model_type.lower()}.")
        except httpx.HTTPStatusError as exc:
            print(f"API Gateway: Error del microservicio {model_type.lower()} al consultar estado. Status: {exc.response.status_code}, Body: {exc.response.text}")
            error_detail = f"Error desde el servicio {model_type.lower()} consultando estado: {exc.response.status_code}"
            try:
                service_error_detail = exc.response.json().get("detail")
                if service_error_detail:
                    error_detail = service_error_detail
            except Exception:
                pass 
            raise HTTPException(status_code=exc.response.status_code, detail=error_detail)
        except httpx.RequestError as exc:
            print(f"API Gateway: No se pudo conectar a {service_url} para consultar estado. Error: {exc}")
            raise HTTPException(status_code=503, detail=f"No se puede conectar al servicio de {model_type.lower()} para consultar estado.")
        except Exception as e:
            print(f"API Gateway: Error inesperado al procesar /train_status/{model_type.lower()}/{job_id}: {type(e).__name__} - {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail="Error interno del Gateway consultando estado del entrenamiento.")


@app.get("/predict/{model_type}")
async def predict(
        model_type: str = Path(..., description="Type of model to use for prediction (e.g., 'rf', 'lstm', 'xgboost')"),
        ticket: str = Query("NU", description="ticket of the stock to import"),
        forecast_horizon: int = Query(10, description="Forecast horizon in days"),
        target_col: str = Query("Close", description="Target column for prediction"),
        historical_days: Optional[int] = Query(365, description="Number of historical days to return")
):
    if model_type.lower() not in microservices:
        raise HTTPException(status_code=400, detail="Invalid model type. Supported types: rf, lstm, xgboost")

    # URL del microservicio correspondiente de predicción
    service_url = f"{microservices[model_type.lower()]}/predict"

    # Parametros para la predicción
    params = {
        "ticket": ticket,
        "forecast_horizon": forecast_horizon,
        "target_col": target_col,
        "historical_days": historical_days
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(service_url, params=params, timeout=420.0)
            response.raise_for_status()
            return response.json()
        except httpx.ReadTimeout:
            raise HTTPException(status_code=504, detail=f"Gateway timeout esperando predicción del servicio {model_type.lower()}.")
        except httpx.HTTPStatusError as exc:
            error_detail = f"Error desde el servicio {model_type.lower()} en predicción: {exc.response.status_code}"
            try: service_error_detail = exc.response.json().get("detail"); error_detail = service_error_detail or error_detail
            except Exception: pass
            raise HTTPException(status_code=exc.response.status_code, detail=error_detail)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error interno del Gateway al solicitar predicción: {str(e)}")


@app.get("/models/{model_type}")
async def list_models(
        model_type: str = Path(..., description="Type of model to list (e.g., 'rf', 'lstm', 'xgboost', 'prophet')")
):
    """
    Lists available trained models for a specific model type by forwarding the request
    to the appropriate microservice.

    Args:
        model_type: Type of the model to list (e.g., 'rf', 'lstm', 'xgboost', 'prophet'). 
            This determines which microservice will handle the request.

    Returns:
        dict: JSON response from the microservice containing information about available models.
        Typically includes total count of models and a list with model details such as
        name, path, metadata, and size.

    Raises:
        HTTPException: If the model type is not supported (status 400) or if there's
            an issue communicating with the microservice (status 500, 503, or 504).
    """
    if model_type.lower() not in microservices:
        raise HTTPException(status_code=400, detail=f"Invalid model type: {model_type}. Supported types: {list(microservices.keys())}")

    # URL del microservicio correspondiente para listar modelos
    service_url = f"{microservices[model_type.lower()]}/models"

    print(f"API Gateway: Listando modelos para {model_type.lower()} en {service_url}")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(service_url, timeout=30.0)
            response.raise_for_status()
            models_data = response.json()
            print(f"API Gateway: Modelos recibidos para {model_type.lower()}: {models_data}")
            return models_data

        except httpx.ReadTimeout:
            print(f"API Gateway: Timeout listando modelos de {service_url}")
            raise HTTPException(status_code=504, detail=f"Gateway timeout listando modelos del servicio {model_type.lower()}.")
        except httpx.HTTPStatusError as exc:
            print(f"API Gateway: Error del microservicio {model_type.lower()} al listar modelos. Status: {exc.response.status_code}, Body: {exc.response.text}")
            error_detail = f"Error desde el servicio {model_type.lower()} listando modelos: {exc.response.status_code}"
            try:
                service_error_detail = exc.response.json().get("detail")
                if service_error_detail:
                    error_detail = service_error_detail
            except Exception:
                pass
            raise HTTPException(status_code=exc.response.status_code, detail=error_detail)
        except httpx.RequestError as exc:
            print(f"API Gateway: No se pudo conectar a {service_url} para listar modelos. Error: {exc}")
            raise HTTPException(status_code=503, detail=f"No se puede conectar al servicio de {model_type.lower()} para listar modelos.")
        except Exception as e:
            print(f"API Gateway: Error inesperado al procesar /models/{model_type.lower()}: {type(e).__name__} - {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail="Error interno del Gateway listando modelos.")


@app.get("/health")
async def health_check():
    return {"status": "Ok"}


# Crear conexión al localhost
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
