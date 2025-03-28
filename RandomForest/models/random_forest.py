# Librerias
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error

# Directorio de trabajo
MODEL_PATH = "models/model.joblib"
N_LAGS = 10  # Número de etiquetas a predecir

def create_lag_features(data, n_lags=N_LAGS):
    """Crea características de retraso para el conjunto de datos"""
    lagged_data = data.copy()
    for i in range(1, n_lags + 1):
        lagged_data[f'lag_{i}'] = lagged_data['Close'].shift(i)
    lagged_data.dropna(inplace=True)
    return lagged_data

# Función para entrenar el modelo
def train_model(data_path: str):
    # Cargar los datos
    data = pd.read_csv(data_path, parse_dates=["Date"])
    data = data.sort_values("Date")

    # Crear características de rezagos
    df_supervised = create_lag_features(data, n_lags=N_LAGS)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    train_size = int(len(df_supervised) * 0.8)
    X_train = df_supervised.iloc[:train_size][[
        f'lag_{i}' for i in range(1, N_LAGS + 1)]] 
    y_train = df_supervised.iloc[:train_size]["Close"]
    X_test = df_supervised.iloc[train_size:][[
        f'lag_{i}' for i in range(1, N_LAGS + 1)]]
    y_test = df_supervised.iloc[train_size:]["Close"]

    # Validación cruzada temporal
    tscv = TimeSeriesSplit(n_splits=3) # Número de divisiones para la validación cruzada

    # Grid para ajuste de hiperparámetros
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [10, 20], # se prueba con valores más altos para evitar el sobreajuste
        'max_features': ['sqrt', 'log2', 0.5]
    }

    # GridSearch para encontrar los mejores hiperparámetros
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv=tscv, 
        scoring='neg_mean_squared_error', # Usar MSE como métrica de evaluación
        n_jobs=-1, # Usar todos los núcleos disponibles
    )

    grid_search.fit(X_train, y_train)

    # Se evaluan los mejores modelos
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Hacer predicciones
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    # Guardar el modelo
    joblib.dump(best_model, MODEL_PATH)

    # Guardar las métricas en un archivo JSON
    metadata = {
        'best_params': grid_search.best_params_,
        'mse_test': mse,
        'mae_test': mae,
        'rmse_test': rmse,
        'mse_train': mape,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    import json
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH.replace('.joblib', '_metadata.json'), 'w') as f:
        json.dump({k: float(v) if isinstance(v, np.float64) else v for k, v in metadata.items()}, f, indent=4)

    print(f"Modelo guardado en {MODEL_PATH}")
    print(f"Metadatos guardados en {MODEL_PATH.replace('.joblib', '_metadata.json')}")

    return {
        "model": "Modelo entrenado y guardado correctamente",
        "best_params": grid_search.best_params_,
        "mse_test": mse,
        "mae_test": mae,
        "rmse_test": rmse,
        "mape": mape,
    }

# Función para predecir
def predict_future(data_path: str, n_steps: int = 1):
    """Predice el futuro usando el modelo entrenado"""
    # Cargar el modelo
    model = joblib.load(MODEL_PATH)

    # Cargar los datos
    data = pd.read_csv(data_path, parse_dates=["Date"])

    # Crear características de rezagos
    df_supervised = create_lag_features(data, n_lags=N_LAGS)

    # Obtener las últimas filas para predecir
    last_rows = df_supervised.tail(1)[[f'lag_{i}' for i in range(1, N_LAGS + 1)]]
    
    # Predecir el futuro
    predictions = []
    input_array = last_rows.values
    for _ in range(n_steps):
        pred = model.predict(input_array)[0]
        predictions.append(pred)
        
        # Actualizar el array de entrada deslizando las características
        input_array = np.roll(input_array, -1)
        input_array[0, -1] = pred

    return {
        "predictions": predictions
    }

# Funcion de prueba
def evaluate_model(data_path: str, days_out: int = 10):
    """
    Reserva los últimos 'days_out' del dataset para evaluación.
    Usa el modelo entrenado para predecir esos días recursivamente y 
    calcula métricas de error comparando las predicciones con los valores reales.
    """
    # Cargar y ordenar los datos
    data = pd.read_csv(data_path, parse_dates=["Date"])
    data.sort_values("Date", inplace=True)
    
    # Separar el conjunto de entrenamiento y el hold-out
    train_df = data.iloc[:-days_out]
    holdout_df = data.iloc[-days_out:]
    
    # Preparar las características del conjunto de entrenamiento
    df_supervised_train = create_lag_features(train_df, n_lags=N_LAGS)
    # Usar la última fila de train_df para iniciar la predicción
    last_row = df_supervised_train.iloc[-1][[f'lag_{i}' for i in range(1, N_LAGS + 1)]]
    current_input = np.array(last_row)  # Vector de tamaño N_LAGS
    
    # Cargar el modelo entrenado
    model = joblib.load(MODEL_PATH)
    
    # Predecir de forma recursiva para 'days_out' días
    predictions = []
    for _ in range(days_out):
        pred = model.predict(current_input.reshape(1, -1))[0]
        predictions.append(pred)
        # Actualizar la entrada: desplazar el vector y añadir la predicción
        current_input = np.roll(current_input, -1)
        current_input[-1] = pred
    
    # Los valores reales a comparar serán los 'Close' del holdout
    actual = holdout_df["Close"].values
    
    # Calcular métricas
    mse = mean_squared_error(actual, predictions)
    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mse)
    
    return {
        "predictions": predictions,
        "actual": actual.tolist(),
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "days_out": days_out
    }