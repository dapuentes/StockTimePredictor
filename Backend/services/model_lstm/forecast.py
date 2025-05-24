import pandas as pd
from typing import Tuple
import numpy as np
from .lstm_model import TimeSeriesLSTMModel


def forecast_future_prices_lstm(
        model: TimeSeriesLSTMModel,
        data: pd.DataFrame,
        forecast_horizon: int = 10,
        target_col: str = 'Close'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Orquesta la predicción de precios futuros utilizando un modelo LSTM entrenado.

    Esta función actúa como un intermediario, llamando al método `predict_future` del
    modelo, que contiene la lógica compleja de predicción recursiva y el cálculo
    de intervalos de confianza mediante Monte Carlo Dropout.

    Args:
        model (TimeSeriesLSTMModel): La instancia del modelo LSTM ya entrenado y cargado.
        data (pd.DataFrame): El DataFrame con los datos históricos necesarios para iniciar el pronóstico.
        forecast_horizon (int): El número de días/pasos hacia el futuro a predecir.
        target_col (str): El nombre de la columna objetivo.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Una tupla conteniendo tres arrays de NumPy:
        - El pronóstico de los puntos principales (predicciones).
        - Los límites inferiores del intervalo de predicción.
        - Los límites superiores del intervalo de predicción.
    """
    print(f"\n--- Iniciando la llamada al pronóstico para los próximos {forecast_horizon} días ---")

    # 1. Delegar toda la lógica de pronóstico al método del modelo
    # Este método ya está diseñado para manejar la recursividad y los intervalos.
    forecast, lower_bounds, upper_bounds = model.predict_future(
        historical_data_df=data,
        forecast_horizon=forecast_horizon,
        target_col=target_col
    )

    # 2. Imprimir los resultados en la consola para una fácil verificación
    print(f"\n--- Resultados del Pronóstico (Valores Desescalados) ---")
    for i in range(forecast_horizon):
        print(
            f"Día {i + 1}: "
            f"Predicción = {forecast[i]:.4f} "
            f"(Intervalo 95%: [{lower_bounds[i]:.4f} - {upper_bounds[i]:.4f}])"
        )

    # 3. Devolver los resultados para que sean procesados por la API
    return forecast, lower_bounds, upper_bounds