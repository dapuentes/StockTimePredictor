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
    Forecast future prices using an LSTM model for a given forecast horizon while handling
    prediction intervals and providing console-based output.

    Args:
        model (TimeSeriesLSTMModel): The LSTM model used for forecasting. Must implement a
            `predict_future` method designed to handle recursive and interval-based predictions.
        data (pd.DataFrame): A DataFrame containing the historical data required for prediction.
        forecast_horizon (int): The number of future periods (e.g., days) to forecast. Default is 10.
        target_col (str): The name of the column in the data representing the target variable
            to forecast. Default is 'Close'.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing three numpy arrays:
            - The first array represents the forecast values.
            - The second array represents the lower bounds of the prediction intervals.
            - The third array represents the upper bounds of the prediction intervals.
    """
    print(f"\n--- Iniciando la llamada al pronóstico para los próximos {forecast_horizon} días ---")


    # Este método ya está diseñado para manejar la recursividad y los intervalos.
    forecast, lower_bounds, upper_bounds = model.predict_future(
        historical_data_df=data,
        forecast_horizon=forecast_horizon,
        target_col=target_col
    )

    # Imprimir los resultados en la consola para una fácil verificación
    print(f"\n--- Resultados del Pronóstico (Valores Desescalados) ---")
    for i in range(forecast_horizon):
        print(
            f"Día {i + 1}: "
            f"Predicción = {forecast[i]:.4f} "
            f"(Intervalo 95%: [{lower_bounds[i]:.4f} - {upper_bounds[i]:.4f}])"
        )

    return forecast, lower_bounds, upper_bounds