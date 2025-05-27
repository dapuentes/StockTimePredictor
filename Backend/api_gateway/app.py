import os

from fastapi import FastAPI, HTTPException, Query, Form, Path
from fastapi.middleware.cors import CORSMiddleware
import httpx
from datetime import datetime

app = FastAPI(title="API Gateway", version="1.0.0")

# Configuración de CORS para permitir solicitudes desde cualquier origen
origins = [
    "http://localhost:3000",  # Frontend local
    "https://stocktimepredictor.netlify.app",  # Frontend desplegado en Netlify
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
async def train_model(
        model_type: str = Path(..., description="Tipo de modelo a entrenar (e.g., 'rf', 'lstm', 'xgboost')"),
        ticket: str = Form("NU"),
        start_date: str = Form("2020-12-10"),
        end_date: str = Form("2023-10-01"),
        n_lags: int = Form(10),
        target_col: str = Form("Close"),
        train_size: float = Form(0.8),
        save_model_path: str = Form(None)
):
    """
    Handles the training of machine learning models by forwarding the request and parameters
    to the appropriate microservice based on the specified model type. Supported model types
    include 'rf', 'lstm', and 'xgboost'. The function validates the model type, constructs the
    necessary payload, and communicates with the targeted microservice to initiate the
    training process.

    Args:
        model_type: Type of the model to train (e.g., 'rf', 'lstm', 'xgboost'). This is a path
            parameter and defines the service endpoint to forward the training request to.
        ticket: Identifier or code for the dataset or data source to be used in training.
            It defaults to 'NU' and is passed as part of the request body.
        start_date: Start date for the training data, specified in 'YYYY-MM-DD' format. Defaults
            to '2020-12-10'.
        end_date: End date for the training data, specified in 'YYYY-MM-DD' format. Defaults to
            today's date.
        n_lags: Number of lagging or previous observations to consider in training. It is passed
            as an integer.
        target_col: The name of the target column in the dataset to perform predictions on. It
            defaults to 'Close'.
        train_size: Proportion of the dataset to be used for training. Specified as a float
            value (e.g., 0.8 for 80%).
        save_model_path: Path to save the trained model. If not specified, defaults to None,
            indicating the model will not be saved directly.

    Returns:
        dict: JSON response from the microservice being invoked for model training. Contains
        details about the training process, such as success status or any diagnostic
        information.

    Raises:
        HTTPException: If the passed model type is not supported, an error is raised with a
            status code 400.
        HTTPException: If there is an issue in sending requests (e.g., server errors or
            connectivity issues) to the targeted microservice, an error is raised with a
            status code 500.
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
                },
                timeout=300.0  # Timeout de 300 segundos
            )

            # Debugging: Imprimir el código de estado y el texto crudo de la respuesta
            print(f"DEBUG Gateway: Status Code recibido de {service_url}: {response.status_code}")
            print(f"DEBUG Gateway: Texto CRUDO recibido de {service_url}: ---START---\n{response.text}\n---END---")

            response.raise_for_status()  # Lanza un error

            json_response = response.json()
            # Debugging: Imprimir la respuesta JSON
            print(f"DEBUG Gateway: Respuesta JSON recibida de {service_url}: {json_response}")
            return json_response
        except Exception as e:
            print(f"ERROR FATAL en Gateway procesando respuesta de {service_url}: {type(e).__name__} - {e}")
            import traceback
            traceback.print_exc()  # <--- Imprime el error detallado
            raise HTTPException(status_code=500, detail=f"Error interno del Gateway...")


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
            response = await client.get(service_url, params=params, timeout=420.0)
            response.raise_for_status()  # Lanza un error
            return response.json()
        except Exception as e:
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


# Crear conexión al localhost
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
