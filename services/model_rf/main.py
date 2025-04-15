from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import os
import glob
from datetime import datetime, timedelta

from services.model_rf.rf_model2 import TimeSeriesRandomForestModel
from services.model_rf.train import train_ts_model
from services.model_rf.forecast import forecast_future_prices
from utils.import_data import load_data

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
    end_date: str = datetime.now().strftime("%Y-%m-%d")
    n_lags: int = 10
    target_col: str = "Close"
    train_size: float = 0.8
    save_model_path: str = None


# Rutas para el almacenamiento de modelos
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Diccionario global para almacenar los modelos entrenados
loaded_models = {}


def get_default_model_path(ticket):
    """
    Generates the default model file path based on a given ticket identifier.

    The function constructs a path to the Random Forest model file using a
    global model directory and a specific ticket identifier. The resulting
    file path includes the ticket identifier as part of the filename.

    Args:
        ticket (str): A string identifier used to build the model file path.

    Returns:
        str: The complete file path to the default model file for the given
        ticket.
    """
    return os.path.join(MODEL_DIR, f"rf_model_{ticket}.joblib")


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
    return {"message": "Random Forest Time Series Model Service is running!"}


@app.post("/train")
async def train_model(request: TrainRequest):
    """
    Handles the training of a time-series forecasting model based on the provided
    historical stock data and training configurations.

    Args:
        request (TrainRequest): A request object containing the necessary
            parameters for training the model, including stock ticket, date range,
            desired lags, target column, training size, and save path for the model.

    Returns:
        dict: A response dictionary containing the training status, success message,
            computed metrics, best hyperparameters, and the path where the trained
            model was saved.

    Raises:
        HTTPException: If any error occurs during the data loading, training, or
            model saving process.
    """
    try:
        # Cargar datos históricos
        data = load_stock_data(request.ticket, request.start_date, request.end_date)

        # Establecer ruta para guardar el modelo
        save_path = request.save_model_path
        if save_path is None or save_path == "":
            # Si no se proporciona una ruta, usar la ruta predeterminada
            save_path = get_default_model_path(request.ticket)

        # Entrenar modelo
        model = train_ts_model(
            data=data,
            n_lags=request.n_lags,
            target_col=request.target_col,
            train_size=request.train_size,
            save_model_path=save_path
        )

        # Agregar el modelo a la memoria caché
        loaded_models[save_path] = model

        # Devolver métricas y mejores parámetros
        return {
            "status": "success",
            "message": f"Model trained successfully for {request.ticket}",
            "metrics": model.metrics,
            "best_params": model.best_params_,
            "model_path": save_path
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")


@app.get("/predict")
async def predict(
        ticket: str = Query("NU", description="Ticker of the stock to predict"),
        forecast_horizon: int = Query(10, description="Forecast horizon in days"),
        target_col: str = Query("Close", description="Target column for prediction")
):
    """
    Handles HTTP GET requests to provide stock price predictions for a specified ticker,
    forecast horizon, and target column. It leverages a pre-trained time series random forest model
    to predict future stock prices.

    Args:
        ticket (str): Ticker of the stock to predict.
        forecast_horizon (int): Forecast horizon in days.
        target_col (str): Target column for prediction.

    Returns:
        dict: A dictionary containing the prediction results, including:
            - status: Status of the prediction.
            - ticker: The input stock ticker.
            - target_column: The input target column.
            - forecast_horizon: The input forecast horizon.
            - predictions: List of dictionaries containing predicted dates and their corresponding values.
            - last_actual_date: Date of the last actual value.
            - last_actual_value: Last actual value of the target column.
            - model_used: Name of the model file used for prediction.
            - model_info: Description of the model's usage, specific or generic for the ticker.

    Raises:
        HTTPException: If an error occurs during model retrieval, loading, data fetching/preparation,
            or forecast computation. Specific error details are included in the HTTP error response.
    """
    try:
        # Verificar que el ticket no esté vacío
        model_path = find_model_for_ticket(ticket)
        if model_path is None:
            raise HTTPException(
                status_code=404,
                detail=f"No trained model found for {ticket}. Train a model first.")

        try:
            model = TimeSeriesRandomForestModel.load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
            print(f"Model type: {type(model)}")

            # Verificar que el modelo tiene los atributos necesarios
            if not hasattr(model, 'n_lags'):
                raise ValueError("Model missing n_lags attribute")
            if not hasattr(model, 'best_pipeline_'):
                raise ValueError("Model missing best_pipeline_ attribute")

        except Exception as model_error:
            raise HTTPException(
                status_code=500,
                detail=f"Error loading model: {str(model_error)}"
            )

        try:
            end_date = datetime.now()
            # Usar un período más largo para asegurar suficientes datos
            start_date = end_date - timedelta(days=365 * 3)  # Tres años de datos históricos
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
            forecast = forecast_future_prices(
                model=model,
                data=data,
                forecast_horizon=forecast_horizon,
                target_col=target_col
            )

            last_date = data.index[-1]
            forecast_dates = [(last_date + timedelta(days=i + 1)).strftime("%Y-%m-%d")
                              for i in range(forecast_horizon)]

            predictions = [{"date": date, "prediction": float(pred)}
                           for date, pred in zip(forecast_dates, forecast)]

            model_info = "Modelo específico para el ticker" if ticket in model_path else "Modelo genérico"

            return {
                "status": "success",
                "ticker": ticket,
                "target_column": target_col,
                "forecast_horizon": forecast_horizon,
                "predictions": predictions,
                "last_actual_date": last_date.strftime("%Y-%m-%d"),
                "last_actual_value": float(data[target_col].iloc[-1]),
                "model_used": os.path.basename(model_path),
                "model_info": model_info
            }

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

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/models")
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
    try:
        models = glob.glob(os.path.join(MODEL_DIR, "rf_model_*.joblib"))
        model_info = []

        for model_path in models:
            model_name = os.path.basename(model_path)
            metadata = model_path.replace(',joblib', "_metadata.json")
            metadata = None

            if os.path.exists(metadata):
                try:
                    import json
                    with open(metadata, 'r') as f:
                        metadata = json.load(f)
                except:
                    metadata = {"error": "Cloud not load metadata"}

            model_info.append({
                "name": model_name,
                "path": model_path,
                "metadata": metadata,
                "size_mb": round(os.path.getsize(model_path) / (1024 * 1024), 2),  # Tamaño en MB
            })
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)  # El puerto 8001 coincide con el microservicio de RF en el API Gateway