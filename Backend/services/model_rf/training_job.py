import os
import json
import joblib
import tempfile # Para descargar/subir archivos temporalmente
import traceback
from datetime import datetime

from google.cloud import storage

from Backend.utils.import_data import load_data
from Backend.utils.preprocessing import scale_data
from .rf_model2 import TimeSeriesRandomForestModel

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID") # Cuando se cree el proyecto se reemplazará por el ID del proyecto
MODEL_BUCKET_NAME = os.getenv("MODEL_BUCKET_NAME") # Cuando se cree el proyecto se reemplazará por el ID del bucket

if not GCP_PROJECT_ID or not MODEL_BUCKET_NAME:
    print("ERROR: Faltan variables de entorno GCP_PROJECT_ID o MODEL_BUCKET_NAME.")
    exit(1)

storage_client = storage.Client(project=GCP_PROJECT_ID)
bucket = storage_client.bucket(MODEL_BUCKET_NAME)

def upload_to_gcs(source_file_name, destination_blob_name):
    """
    Uploads a file to Google Cloud Storage (GCS) at the specified destination. The
    function takes a local file and uploads it to a GCS bucket specified by the
    global variables defined in the environment.

    Parameters:
        source_file_name (str): The path to the local file to be uploaded.
        destination_blob_name (str): The desired path or object name in the GCS
        bucket where the file will be stored.

    Raises:
        Exception: If an error occurs during the upload process, it raises the
        exception after printing the error message for debugging purposes.
    """
    try:
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        print(f"Archivo {source_file_name} subido a gs://{MODEL_BUCKET_NAME}/{destination_blob_name}")
    except Exception as e:
        print(f"ERROR al subir {source_file_name} a GCS: {e}")
        raise

def run_training(params):
    """
    Performs the training process of a time series machine learning model using specified
    parameters and configurations. This includes loading data, preparing it for modeling,
    training the model, optimizing hyperparameters, evaluating the results, and saving the
    trained model along with its metadata to Google Cloud Storage (GCS). Additionally,
    handles potential errors and logs progress at each step. The model type and related
    configurations are passed as a dictionary (`params`).

    Parameters:
        params (dict): Dictionary containing training parameters.
            - ticket (str, optional): Ticker symbol for fetching data. Defaults to "NU".
            - start_date (str, optional): Start date for the data. Defaults to "2020-12-10".
            - end_date (str, optional): End date for the data. Defaults to "2023-10-01".
            - n_lags (int, optional): Number of lag features for time series modeling.
              Defaults to 10.
            - target_col (str, optional): Target column to be predicted. Defaults to "Close".
            - train_size (float, optional): Train/test split ratio. Defaults to 0.8.
            - model_type (str, optional): Type of model to train. Defaults to "rf".
    """
    print(f"Iniciando entrenamiento con parámetros: {params} ...")

    # Extraer los parámetros
    ticker = params.get("ticket", "NU")
    start_date = params.get("start_date", "2020-12-10")
    end_date = params.get("end_date", "2023-10-01")
    n_lags = int(params.get("n_lags", 10))
    target_col = params.get("target_col", "Close")
    train_size = float(params.get("train_size", 0.8))
    model_type = params.get("model_type", "rf")

    # Logica de train_ts_model
    try:
        print(f"Cargando datos para {ticker} desde {start_date} hasta {end_date}...")
        data = load_data(ticker=ticker, start_date=start_date, end_date=end_date)
        if data.empty:
             print(f"No se encontraron datos para {ticker}. Abortando.")
             return
        print(f"Datos cargados: {len(data)} filas.")

        print("Inicializando y preparando el modelo...")
        model = TimeSeriesRandomForestModel(n_lags=n_lags)
        processed_data = model.prepare_data(data, target_col=target_col)

        if processed_data.empty:
            print("Datos procesados vacíos. Abortando.")
            return

        train_split_index = int(len(processed_data) * train_size)
        train_data = processed_data.iloc[:train_split_index]
        test_data = processed_data.iloc[train_split_index:]

        X_train = train_data.drop(columns=[target_col])
        y_train = train_data[target_col].values.reshape(-1, 1)
        X_test = test_data.drop(columns=[target_col])
        y_test = test_data[target_col].values.reshape(-1, 1)

        feature_names = X_train.columns.tolist()
        print(f"Features: {feature_names}")

        print("Escalando datos...")
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, feature_scaler, target_scaler = scale_data(
            X_train, X_test, y_train, y_test
        )

        model.feature_scaler = feature_scaler
        model.target_scaler = target_scaler

        print("Optimizando hiperparámetros...")
        model.optimize_hyperparameters(
            X_train_scaled,
            y_train_scaled.ravel(),
            feature_names=feature_names
        )
        print(f"Mejores parámetros encontrados: {model.best_params_}")

        print("Evaluando modelo...")
        model.evaluate(X_test_scaled, y_test)
        print(f"Métricas del modelo: {model.metrics}")

        # Guardar el modelo
        print("Guardando el modelo...")
        model_filename = f"{model_type}_model_{ticker}.joblib"
        metadata_filename = f"{model_type}_model_{ticker}_metadata.json"

        # Ruta del modelo en el bucket de GCS
        gcs_model_path = f"{model_type}_models/{model_filename}"
        gcs_metadata_path = f"{model_type}_models/{metadata_filename}"

        # Se guarda temporalmente
        with tempfile.TemporaryDirectory() as tmpdir:
            local_model_path = os.path.join(tmpdir, model_filename)
            local_metadata_path = os.path.join(tmpdir, metadata_filename)

            print(f"Guardando modelo localmente en {local_model_path}...")
            model.save_model_local(local_model_path, local_metadata_path)  # Crear este metodo

            print(f"Subiendo {model_filename} a GCS...")
            upload_to_gcs(local_model_path, gcs_model_path)
            print(f"Subiendo {metadata_filename} a GCS...")
            upload_to_gcs(local_metadata_path, gcs_metadata_path)

        print("Entrenamiento finalizado y modelo guardado en GCS.")

    except Exception as e:
        print(f"ERROR durante el entrenamiento para {ticker}: {e}")
        traceback.print_exc()

# Punto de entrada para el entrenamiento
if __name__ == "__main__":
    print("Iniciando Cloud Run Job para entrenamiento...")

    training_params = {
        "model_type": os.getenv("MODEL_TYPE", "rf"),
        "ticket": os.getenv("TICKER", "NU"),
        "start_date": os.getenv("START_DATE", "2020-12-10"),
        "end_date": os.getenv("END_DATE", datetime.now().strftime("%Y-%m-%d")),
        "n_lags": os.getenv("N_LAGS", 10),
        "target_col": os.getenv("TARGET_COL", "Close"),
        "train_size": os.getenv("TRAIN_SIZE", 0.8),
    }

    run_training(training_params)
    print("Cloud Run Job finalizado.")


