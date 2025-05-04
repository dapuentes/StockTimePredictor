import os
from google.cloud import pubsub_v1
from fastapi import FastAPI, HTTPException, Query, Form, Path
from fastapi.middleware.cors import CORSMiddleware
import httpx
from datetime import datetime

# Configuración de la variable de entorno GOOGLE_APPLICATION_CREDENTIALS
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID") # Cuando se cree el proyecto se reemplazará por el ID del proyecto
PUB_SUB_TOPIC_ID = os.getenv("PUB_SUB_TOPIC_ID", "training-request") # Cuando se cree el proyecto se reemplazará por el ID del topic

if GCP_PROJECT_ID is None:
    print("ERROR: La variable de entorno GCP_PROJECT_ID no está configurada.")
    raise ValueError("La variable de entorno GCP_PROJECT_ID no está configurada.")

publisher = None
topic_path = None

if GCP_PROJECT_ID and PUB_SUB_TOPIC_ID:
    try:
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(GCP_PROJECT_ID, PUB_SUB_TOPIC_ID)
        print(f"Publicador Pub/Sub inicializado para el tema: {topic_path}")
    except Exception as e:
        print(f"ERROR: No se pudo inicializar el cliente Pub/Sub: {e}")
        publisher = None # Deshabilita la publicación si falla la inicialización
else:
    print("ADVERTENCIA: GCP_PROJECT_ID o PUB_SUB_TOPIC_ID no configurados. La publicación Pub/Sub estará deshabilitada.")

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


@app.get("/")
async def read_root():
    """
    Handles the root endpoint of the API Gateway that serves as the entry point for API
    requests. Provides a welcome message to confirm the API's availability.

    Returns:
        dict: A dictionary containing a single key-value pair with a welcome
        message.
    """
    return {"message": "Welcome to the API Gateway"}


@app.post("/train/{model_type}")
async def request_training(
        model_type: str = Path(..., description="Tipo de modelo a entrenar (e.g., 'rf', 'lstm', 'xgboost')"),
        ticket: str = Form("NU"),
        start_date: str = Form("2020-12-10"),
        end_date: str = Form("2023-10-01"),
        n_lags: int = Form(10),
        target_col: str = Form("Close"),
        train_size: float = Form(0.8)
):
    """
    Handles a request to initiate a machine learning model training process. This endpoint
    validates the provided model type, constructs a training request message, and publishes
    it to a Pub/Sub topic for asynchronous processing. Returns confirmation details if the
    request is successfully handled.

    Parameters:
        model_type (str): The type of model to train (e.g., 'rf', 'lstm', 'xgboost').
        ticket (str): The identifier associated with the training ticket. Defaults to 'NU'.
        start_date (str): The start date for the training data window in YYYY-MM-DD format.
            Defaults to '2020-12-10'.
        end_date (str): The end date for the training data window in YYYY-MM-DD format.
            Defaults to '2023-10-01'.
        n_lags (int): The number of lagged values to use as input features for the model.
            Defaults to 10.
        target_col (str): The target column name in the dataset. Defaults to 'Close'.
        train_size (float): The proportion of the dataset to include in the training split.
            Defaults to 0.8.

    Returns:
        dict: A JSON-compatible dictionary that contains the status of the operation, a
            message confirming submission of the training request, and the identifier of
            the published message.

    Raises:
        HTTPException: If the Pub/Sub service is not available, the model type is invalid,
            or an error occurs while attempting to publish the message.
    """

    if not publisher or not topic_path:
        raise HTTPException(status_code=503, detail="Pub/Sub service is not available")

    # Validar el tipo de modelo
    supported_models = ["rf", "lstm", "xgboost", "prophet"]
    if model_type.lower() not in supported_models:
        raise HTTPException(status_code=400, detail=f"Invalid model type. Supported types: {', '.join(supported_models)}")

    # Para publicar en Pub/Sub
    try:
        # Crear el mensaje de entrenamiento
        message_data = {
            "model_type": model_type,
            "ticket": ticket,
            "start_date": start_date,
            "end_date": end_date,
            "n_lags": n_lags,
            "target_col": target_col,
            "train_size": train_size,
            "request_time": datetime.now().isoformat()
        }

        # Convertir el mensaje a bytes
        message_bytes = str(message_data).encode("utf-8")

        # Publicar el mensaje en el tema de Pub/Sub
        future = publisher.publish(topic_path, data=message_bytes)
        message_id = future.result() # Esperar a que se publique el mensaje

        print(f"Mensaje de entrenamiento para {model_type}/{ticket} publicado con ID: {message_id}")

        # Respuesta al frontend
        return {
            "status": "success",
            "message": f"Training request for {model_type} model with ticket {ticket} has been submitted.",
            "message_id": message_id
        }

    except Exception as e:
        print(f"Error al publicar el mensaje: {e}")
        raise HTTPException(status_code=500, detail="Error in sending request to Pub/Sub service")

@app.get("/predict/{model_type}")
async def predict(
        model_type: str = Path(..., description="Type of model to use for prediction (e.g., 'rf', 'lstm', 'xgboost')"),
        ticket: str = Query("NU", description="ticket of the stock to import"),
        forecast_horizon: int = Query(10, description="Forecast horizon in days"),
        target_col: str = Query("Close", description="Target column for prediction")
):
    """
    Handles predictions by routing requests to appropriate microservices based on the
    specified model type. Validates the model type and forwards user input as
    parameters to designated microservices for computation.

    Args:
        model_type (str): Type of model to use for prediction (e.g., 'rf', 'lstm', 'xgboost').
        ticket (str): Stock ticket identifier to import for prediction (default is 'NU').
        forecast_horizon (int): Forecast duration in days (default is 10).
        target_col (str): Target column for prediction (default is 'Close').

    Raises:
        HTTPException: If the specified model type is invalid.
        HTTPException: If any error occurs while sending the prediction request to the
            appropriate microservice.

    Returns:
        dict: JSON response from the respective microservice containing prediction results.
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
            print(f"Gateway sending request to {service_url} with params: {params}")
            response = await client.get(service_url, params=params, timeout=300.0)
            response.raise_for_status()  # Lanza un error
            return response.json()
        except httpx.RequestError as exc:
            print(f"Error in request to {exc.request.url!r}: {exc}")
            raise HTTPException(status_code=503, detail=f"Error in request to microservice: {model_type}: {exc}")
        except httpx.HTTPStatusError as exc:
            print(f"Error response {exc.response.status_code} from {exc.request.url!r}: {exc.response.text}")
            detail = f"Error {exc.response.status_code} from microservice {model_type}."
            try:
                detail += f" Detail: {exc.response.json().get('detail', exc.response.text)}"
            except:
                detail += f" Detail: {exc.response.text}"
            raise HTTPException(status_code=exc.response.status_code, detail=detail)
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise HTTPException(status_code=500, detail=f"Error in sending request to microservice: {e}")


@app.get("/health")
async def health_check():
    """
    Handles the health check endpoint to ensure the application is running correctly.

    Returns a dictionary indicating the status of the application.

    Returns:
        dict: A dictionary containing the application's health status.
    """
    return {"status": "Ok"}

