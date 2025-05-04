import json

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
from typing import Optional
import glob
from datetime import datetime, timedelta
import traceback

from google.cloud import storage

# Importar módulos personalizados
from Backend.services.model_rf.rf_model2 import TimeSeriesRandomForestModel
from Backend.services.model_rf.forecast import forecast_future_prices
from Backend.utils.import_data import load_data

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
MODEL_BUCKET_NAME = os.getenv("MODEL_BUCKET_NAME")

if not GCP_PROJECT_ID or not MODEL_BUCKET_NAME:
    print("ERROR: Faltan variables de entorno GCP_PROJECT_ID o MODEL_BUCKET_NAME.")
    storage_client = None
else:
    storage_client = storage.Client(project=GCP_PROJECT_ID)

app = FastAPI(title="Random Forest Time Series Model Service", version="1.0.0")


# Definir el modelo de datos para la solicitud de entrenamiento
class TrainRequest(BaseModel):
    """
    Represents a request for training a model with specific configurations.

    This class defines the structure for a training request, containing
    parameters required for training a model, such as ticket information, dates,
    target column, number of lags, and other configurations. Instances of this
    class are used to pass data to the training process.

    Attributes:
        ticket (str): The identifier or code used for training, default is "NU".
        start_date (str): The starting date of the data range for training,
            default is "2020-12-10".
        end_date (str): The ending date of the data range for training, default
            is the current date in "YYYY-MM-DD" format.
        n_lags (int): The number of lag features to consider for the training
            data, default is 10.
        target_col (str): The name of the target column in the dataset, default
            is "Close".
        train_size (float): The proportion of the dataset to use for training,
            default is 0.8.
        save_model_path (str): The file path where the trained model should be
            saved, default is None.
    """
    ticket: str = "NU"
    start_date: str = "2020-12-10"
    end_date: str = "2023-10-01"
    n_lags: int = 10
    target_col: str = "Close"
    train_size: float = 0.8


# Diccionario global para almacenar los modelos entrenados
loaded_models = {}


def get_gcs_model_paths(ticket, model_type="rf"):
    """
    Generate Google Cloud Storage paths for model and metadata files.

    This function constructs the file paths for a model file and its
    corresponding metadata file, stored in a folder based on the given
    model type. The filenames include the provided ticket and model
    type as part of their naming convention.

    Parameters:
    ticket : str
        Unique identifier used to generate the file names for the model
        and metadata.
    model_type : str, optional
        Type of the model, such as "rf" (default is "rf").

    Returns:
    tuple
        A tuple containing two elements:
        - The Google Cloud Storage path for the model file as a string.
        - The Google Cloud Storage path for the metadata file as a string.
    """
    model_filename = f"{model_type}_model_{ticket}.joblib"
    metadata_filename = f"{model_type}_model_{ticket}_metadata.json"
    gcs_model_path = f"{model_type}_models/{model_filename}"
    gcs_metadata_path = f"{model_type}_models/{metadata_filename}"
    return gcs_model_path, gcs_metadata_path

def find_model_paths_in_gcs(ticket, model_type="rf"):
    """
    Finds the paths to specific or generic machine learning model files stored in Google Cloud Storage (GCS).

    This function searches for a model associated with a given ticket and model type in a GCS bucket.
    It first attempts to locate a specific model for the ticket, and if unavailable, it searches for
    a generic model. The function requires access to a valid GCS client and pre-defined bucket name
    through properly configured environment variables.

    Parameters:
    ticket: str
        Identifier used to search for the model associated with a specific entity or purpose.
    model_type: str, optional
        Type of the machine learning model. Defaults to "rf".

    Returns:
    tuple[str | None, str | None]
        A tuple containing the paths to the model file and its metadata file in the GCS bucket. If no model is found,
        both values will be None.
    """
    if not storage_client or not MODEL_BUCKET_NAME:
        print("ERROR: Faltan variables de entorno GCP_PROJECT_ID o MODEL_BUCKET_NAME.")
        return None, None

    bucket = storage_client.bucket(MODEL_BUCKET_NAME)

    # Buscar el modelo específico
    specific_model_path, specific_metadata_path = get_gcs_model_paths(ticket, model_type)
    model_blob = bucket.blob(specific_model_path)
    if model_blob.exists():
        print(f"Modelo específico encontrado en GCS: {specific_model_path}")
        return specific_model_path, specific_metadata_path

    # Buscar el modelo genérico
    generic_model_path, generic_metadata_path = get_gcs_model_paths("generic", model_type)
    model_blob = bucket.blob(generic_model_path)
    if model_blob.exists():
        print(f"Modelo genérico encontrado en GCS: {generic_model_path}")
        return generic_model_path, generic_metadata_path

    print(f"No se encontró ningún modelo {model_type} en GCS para {ticket} o genérico.")
    return None, None

def load_model_from_cache_or_gcs(gcs_model_path, gcs_metadata_path):
    """
    Loads a machine learning model from either a local cache or Google Cloud Storage (GCS).

    This function attempts to load a model from a local in-memory cache if it has already
    been loaded previously. If the model is not found in the cache, it proceeds to load the
    model from Google Cloud Storage (GCS). This allows efficient reuse of models and reduces
    load times by avoiding repeated downloads from GCS. Additionally, this function ensures
    proper error handling in cases where paths are invalid, the storage service is not
    configured, or the loading operation fails.

    Parameters:
        gcs_model_path (str): The path to the model file in Google Cloud Storage. This
            must be a valid path, as it is used to identify and fetch the model.
        gcs_metadata_path (str): The path to the metadata file associated with the model
            in Google Cloud Storage.

    Returns:
        TimeSeriesRandomForestModel: An instance of the loaded machine learning model.

    Raises:
        HTTPException: Raised with a status code of 404 if the model path provided is invalid.
        HTTPException: Raised with a status code of 503 if the storage service or bucket
            name is not configured properly.
        HTTPException: Raised with a status code of 500 if any other error occurs during
            the model loading process.
    """
    cache_key = gcs_model_path
    if cache_key in loaded_models:
        print(f"Modelo cargado desde cache: {cache_key}")
        return loaded_models[cache_key]

    if not gcs_model_path:
        raise HTTPException(status_code=404, detail="Ruta del modelo GCS no válida.")

    if not storage_client or not MODEL_BUCKET_NAME:
        raise HTTPException(status_code=503, detail="Servicio de almacenamiento no configurado.")

    try:
        model = TimeSeriesRandomForestModel.load_model_from_gcs(
            gcs_model_path,
            gcs_metadata_path,
            MODEL_BUCKET_NAME,
            GCP_PROJECT_ID
        )
        loaded_models[cache_key] = model  # Guardar en caché
        return model
    except Exception as e:
        # El error ya se loguea en load_model_from_gcs
        raise HTTPException(status_code=500, detail=f"Error al cargar el modelo desde GCS: {e}")


def get_generic_model_path():
    """
    Retrieve the file path for the generic machine learning model.

    This function constructs the file path for the default machine learning
    model using the predefined `MODEL_DIR` directory and a specified file name
    of the model. It returns the complete file path as a string.

    Returns:
        str: The path to the generic machine learning model file.
    """
    return os.path.join(MODEL_DIR, "rf_model.joblib")


def find_model_for_ticket(ticket):
    """
    Finds and returns the appropriate model path for a given ticket. The function prioritizes
    specific models related to the ticket, then checks for generic models, and finally looks
    for any available models in a designated directory as a last resort. If no model is found,
    it returns None.

    Args:
        ticket: The information or code related to the model being searched.

    Returns:
        str or None: The path to the selected model if found, otherwise None.
    """
    specific_model_path = get_default_model_path(ticket)
    if os.path.exists(specific_model_path):
        return specific_model_path

    # Si no hay modelo específico, busca un modelo genérico
    generic_model_path = get_generic_model_path()
    if os.path.exists(generic_model_path):
        return generic_model_path

    # Busca cualquier modelo en el directorio de modelos como último recurso
    avaliable_models = glob.glob(os.path.join(MODEL_DIR, "rf_model_*.joblib"))
    if avaliable_models:
        return avaliable_models[0]
    return None


def load_model(model_path):
    """
    Loads a time series random forest model from the specified path.

    This function retrieves a pre-loaded model from the `loaded_models` cache if it
    exists. Otherwise, it attempts to load the model from the provided file path
    using the `TimeSeriesRandomForestModel` class and caches the loaded model
    before returning it. If the model cannot be loaded due to an error, an
    `HTTPException` is raised with a relevant error message.

    Args:
        model_path: The file path to the serialized TimeSeriesRandomForestModel.

    Returns:
        The loaded TimeSeriesRandomForestModel instance.

    Raises:
        HTTPException: If there is an error during the model loading process.
    """
    if model_path in loaded_models:
        return loaded_models[model_path]

    try:
        model = TimeSeriesRandomForestModel.load_model(model_path)
        loaded_models[model_path] = model
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


def load_stock_data(ticket, start_date, end_date):
    """
    Loads stock data for a specific ticket and date range.

    This function retrieves stock market data for the provided ticket across a specified
    date range. If no data is found for the given ticket, a 404 HTTP exception is raised.
    In case of other errors, a 500 HTTP exception is triggered.

    Args:
        ticket: The stock ticker symbol identifying the financial instrument.
        start_date: The start date of the desired data time range.
        end_date: The end date of the desired data time range.

    Returns:
        pandas.DataFrame: A DataFrame containing the stock data for the given
        ticket and date range.

    Raises:
        HTTPException: If no stock data is found for the given ticket or if an
        error occurs during data loading.
    """
    try:
        data = load_data(ticker=ticket, start_date=start_date, end_date=end_date)
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for ticket {ticket}")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading data: {str(e)}")


@app.get("/")
async def read_root():
    """
    Handles the root endpoint of the API.

    This function is triggered when a GET request is made to the root URL ("/").
    It returns a JSON object containing a welcome message for the Random Forest
    Time Series Model Service.

    Returns:
        dict: A dictionary containing a single key-value pair with the message.

    """
    return {"message": "Random Forest Time Series Model Service"}


@app.post("/train")
async def train_model(request: TrainRequest):
    """
    Handles the training process initiation for a machine learning model via an API endpoint.

    This endpoint is designed to trigger the model training asynchronously. However, it raises an
    HTTP exception to indicate that training is initiated via a Pub/Sub mechanism rather than
    directly triggered by this API endpoint.

    Parameters:
        request (TrainRequest): A request object containing the necessary information for model
        training.

    Raises:
        HTTPException: Always raised with a status code of 501 and a specific detail message to
        specify that training is started via Pub/Sub, not directly through this API endpoint.

    Returns:
        None
    """
    raise HTTPException(status_code=501, detail="Entrenamiento iniciado vía Pub/Sub, no directamente por API.")


@app.get("/predict")
async def predict(
        ticket: str = Query("NU", description="Ticker of the stock to predict"),
        forecast_horizon: int = Query(10, description="Forecast horizon in days"),
        target_col: str = Query("Close", description="Target column for prediction"),
        history_days: int = Query(365, description="Number of historical days to consider for prediction")
):
    """
    Handles stock price predictions for a specified stock ticker using historical data and a pre-trained model.

    The function retrieves a machine learning model from Google Cloud Storage (GCS), ensures necessary attributes are available in the model,
    and processes historical stock data to generate future price predictions. It accommodates forecasting for a user-defined horizon in trading days
    while providing contextual metadata such as historical values and dates.

    Arguments:
        ticket (str): Stock ticker symbol to predict (e.g., "AAPL").
        forecast_horizon (int): Number of business days for the prediction period.
        target_col (str): Column in the stock data that serves as the prediction target (e.g., "Close").
        history_days (int): Number of historical days to include in the returned historical data for context.

    Returns:
        dict: Contains forecast results, historical data, and model information. Structure includes:
              - status (str): Status of the operation ('success').
              - ticker (str): Ticker for which the prediction was made.
              - target_column (str): Target column used in the prediction.
              - forecast_horizon (int): Number of days forecasted.
              - historical_dates (List[str]): List of historical dates used for context.
              - historical_values (List[float]): Historical values corresponding to `target_col`.
              - predictions (List[dict]): List of forecasted dates and corresponding predicted values.
              - last_actual_date (str): Date of the last available historical data.
              - last_actual_value (float): Value of the target column on the last available historical date.
              - model_used (str): Name of the model file used for forecasting.
              - model_info (str): Information about whether the model was specific to the ticker or a generic one.

    Raises:
        HTTPException: Various cases including:
            - When no trained model is found.
            - When the model lacks required attributes.
            - When loading historical stock data fails or is incomplete.
            - When the target column is missing from the data.
            - When the historical data is insufficient for predictions.
            - When errors occur during forecast generation.
        Exception: For unexpected errors during execution.
    """
    try:
        # Encontrar el modelo en GCS
        gcs_model_path, gcs_metadata_path = find_model_paths_in_gcs(ticket, model_type="rf")
        if not gcs_model_path:
            raise HTTPException(status_code=404, detail=f"No se encontró modelo entrenado (específico o genérico) para {ticket} en GCS.")

        try:
            model = load_model_from_cache_or_gcs(gcs_model_path, gcs_metadata_path)
            print(f"Model loaded successfully from {gcs_model_path}")
            print(f"Model type: {type(model)}")

            # Verificar que el modelo tiene los atributos necesarios
            if not hasattr(model, 'n_lags'):
                raise ValueError("Model missing n_lags attribute")
            if not hasattr(model, 'best_pipeline_'):
                raise ValueError("Model missing best_pipeline_ attribute")

        except Exception as model_error:
            # Imprime el error real para depuración en el backend
            print(f"ERROR loading model: {model_error}")
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Error loading model: {str(model_error)}"
            )

        training_end_date = None
        if hasattr(model, 'metadata') and model.metadata:
            training_end_date = model.metadata.get("training_end_date")
            print(f"Using training end date from metadata: {training_end_date}")
        else:
            print("Warning: Model metadata not found. Attempting to load from GCS.")

        end_date = datetime.now()
        if training_end_date:
            try:
                end_date = datetime.strptime(training_end_date, "%Y-%m-%d")
                print(f"Using training end date from metadata: {end_date.strftime('%Y-%m-%d')}")
            except (ValueError, TypeError) as date_error:
                print(
                    f"Warning: Invalid date format in metadata ('{training_end_date}'). Falling back to current date. Error: {date_error}")
            else:
                print(f"Warning: training_end_date not found in metadata. Falling back to current date.")

        # Cargar datos históricos

        try:
            start_date = end_date - timedelta(days=365 * 3)  # Tres años de datos históricos
            print(f"Loading data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            data = load_stock_data(ticket, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

            print(f"Data loaded: {len(data)} rows, columns: {data.columns.tolist()}")

            # Verificar que tenemos datos suficientes
            if data.empty:
                raise HTTPException(
                    status_code=404,
                    detail=f"No data available for ticker {ticket}"
                )

            if target_col not in data.columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Target column '{target_col}' not found in data. Available columns: {data.columns.tolist()}"
                )

            # Verificar que hay suficientes filas para los rezagos
            if len(data) <= model.n_lags * 2:
                raise HTTPException(
                    status_code=400,
                    detail=f"Not enough historical data for prediction. Need at least {model.n_lags * 2} rows, but got {len(data)}."
                )

        except HTTPException:
            raise
        except Exception as data_error:
            raise HTTPException(
                status_code=500,
                detail=f"Error loading stock data: {str(data_error)}"
            )

        try:
            if not hasattr(model, 'feature_scaler') or not hasattr(model, 'target_scaler'):
                print("Warning: Model missing feature_scaler or target_scaler attributes. Attempting to load from GCS.")
                raise ValueError("Model missing feature_scaler or target_scaler attributes")

            forecast = forecast_future_prices(
                model=model,
                data=data.copy(),
                forecast_horizon=forecast_horizon,
                target_col=target_col
            )

        except Exception as forecast_error:
            import traceback
            error_details = traceback.format_exc()
            print(f"Forecast error details: {error_details}")

            if "Found array with 0 sample(s)" in str(forecast_error):
                raise HTTPException(
                    status_code=500,
                    detail="Error during data scaling: Empty array encountered. This typically happens when data preparation removes all rows. Check if your historical data matches the format expected by the model."
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error making prediction: {str(forecast_error)}"
                )

        last_date = data.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=forecast_horizon,
            freq='B'  # 'B' para días hábiles del mercado
        ).strftime('%Y-%m-%d').tolist()

        predictions = [{"date": date, "prediction": float(pred)}
                       for date, pred in zip(forecast_dates, forecast)]

        historical_data_to_return = data.iloc[-history_days:]
        historical_dates = historical_data_to_return.index.strftime('%Y-%m-%d').tolist()
        historical_values = historical_data_to_return[target_col].tolist()

        model_basename = os.path.basename(gcs_model_path)
        model_info = "Modelo específico para el ticker" if ticket in gcs_model_path else "Modelo genérico"

        return {
            "status": "success",
            "ticker": ticket,
            "target_column": target_col,
            "forecast_horizon": forecast_horizon,
            "historical_dates": historical_dates,
            "historical_values": historical_values,
            "predictions": predictions,
            "last_actual_date": last_date.strftime("%Y-%m-%d"),
            "last_actual_value": float(data[target_col].iloc[-1]),
            "model_used": model_basename,
            "model_info": model_info
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("models")
async def list_models():
    """
    Lists all models available in the specified model directory. The function searches
    for model files with a specific naming pattern in the MODEL_DIR directory, gathers
    metadata if available, and calculates the size of each model file.

    Returns:
        dict: A dictionary containing the total number of models and a list of model
        details including name, path, metadata, and size in megabytes.

    Raises:
        HTTPException: If an error occurs while retrieving or processing the models,
        an HTTPException with status code 500 is raised.
    """
    if not storage_client or not MODEL_BUCKET_NAME:
        raise HTTPException(status_code=503, detail="Servicio de almacenamiento no configurado.")

    try:
        bucket = storage_client.bucket(MODEL_BUCKET_NAME)
        prefix = "rf_models/"
        blobs = bucket.list_blobs(prefix=prefix)
        model_info = []

        processed_models = set()  # Para evitar duplicados

        for blob in blobs:
            if blob.name.endswith(".joblib"):
                model_base_path = blob.name.replace(".joblib", "")
                if model_base_path in processed_models:
                    continue

                model_name = os.path.basename(blob.name)
                metadata_path = model_base_path + "_metadata.json"
                metadata_blob = bucket.blob(metadata_path)
                metadata = None

                if metadata_blob.exists():
                    try:
                        metadata = json.loads(metadata_blob.download_as_text())
                    except Exception as e:
                        print(f"Error loading metadata for {model_name}: {e}")
                else:
                    metadata = {"info: No metadata found"}

                model_info.append({
                    "name": model_name,
                    "gcs_path": f"gs://{MODEL_BUCKET_NAME}/{blob.name}",
                    "metadata": metadata,
                    "size_mb": blob.size / (1024 * 1024),  # Convertir a MB
                    "last_modified": blob.updated.isoformat() if blob.updated else None
                })
                processed_models.add(model_base_path)

        return {
            "total_models": len(model_info),
            "models": model_info
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")


@app.get("/health")
async def health_check():
    """
    Handles health check endpoint of the application.

    This function serves as a simple health check mechanism to verify that the
    application is running and reachable. It provides a status message indicating
    the health status of the application.

    Returns:
        dict: A dictionary containing the status message with an "Ok" value.
    """
    return {"status": "Ok"}
