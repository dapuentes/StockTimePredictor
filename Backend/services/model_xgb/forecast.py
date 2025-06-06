# Backend/services/model_xgb/forecast.py

import pandas as pd
import numpy as np
from typing import Tuple

# Assuming TimeSeriesXGBoostModel is in the same directory
from .xgb_model import TimeSeriesXGBoostModel


def forecast_future_prices_xgb(
    model: TimeSeriesXGBoostModel,
    data: pd.DataFrame,
    forecast_horizon: int = 10,
    target_col: str = 'Close'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Forecast future prices using a trained TimeSeriesXGBoostModel.

    This function utilizes the `predict_future` method of the provided XGBoost model
    to compute forecasted values. For XGBoost, true confidence intervals are
    typically not generated by the base model in the same way as some other models;
    therefore, the returned lower and upper bounds might be placeholders (e.g.,
    repeating the point forecast or NaN values) as defined in the model's
    `predict_future` method.

    Args:
        model (TimeSeriesXGBoostModel): A trained instance of TimeSeriesXGBoostModel.
        data (pd.DataFrame): Historical data to base the forecasts on. This DataFrame
                             should contain all necessary columns for the model's
                             preprocessor to generate features.
        forecast_horizon (int): Number of future time steps to forecast. Defaults to 10.
        target_col (str): Name of the target column in the `data` DataFrame that
                          the model was trained to predict. Defaults to 'Close'.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing three numpy arrays:
            - The first array represents the point forecast values (unscaled).
            - The second array represents the lower bounds of the forecast (unscaled, placeholder).
            - The third array represents the upper bounds of the forecast (unscaled, placeholder).
    """
    print(f"\n--- Iniciando la llamada al pronóstico XGBoost para los próximos {forecast_horizon} días ---")

    if not isinstance(model, TimeSeriesXGBoostModel):
        raise TypeError(f"El modelo proporcionado no es una instancia de TimeSeriesXGBoostModel. Recibido: {type(model)}")

    # The model's predict_future method handles the recursive forecasting logic,
    # including data preparation using its internal preprocessor.
    forecast, lower_bounds, upper_bounds = model.predict_future(
        historical_data_df=data.copy(),  # Use a copy to prevent modifying the original DataFrame
        forecast_horizon=forecast_horizon,
        target_col=target_col
    )

    print(f"\n--- Resultados del Pronóstico XGBoost (Valores Desescalados) ---")
    for i in range(forecast_horizon):
        # Note: lower_bounds and upper_bounds are placeholders as per TimeSeriesXGBoostModel.predict_future
        print(
            f"Día {i + 1}: "
            f"Predicción = {forecast[i]:.4f} "
            f"(Intervalo Placeholder: [{lower_bounds[i]:.4f} - {upper_bounds[i]:.4f}])"
        )

    return forecast, lower_bounds, upper_bounds