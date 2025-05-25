import pandas as pd


def forecast_future_prices(model, data: pd.DataFrame, forecast_horizon: int = 10, target_col: str = 'Close'):
    """
    Forecast future prices for a given dataset using a specified model. This function
    utilizes the `predict_future` method of the provided model to compute forecasted
    values, along with their lower and upper confidence bounds. All scaling and
    descaling operations are assumed to be handled internally by the model's
    `predict_future` method.

    Args:
        model: A trained model instance capable of forecasting future values.
        data (pd.DataFrame): Historical data to base the forecasts on. The data
            should include the feature columns expected by the model and the target
            column containing the variable to forecast.
        forecast_horizon (int): Number of future time steps to forecast. Defaults to 10.
        target_col (str): Name of the target column in the dataset that contains the
            values to predict. Defaults to 'Close'.

    Returns:
        tuple: A tuple of three elements:
            forecast (np.ndarray): Array of forecasted values for the next
                `forecast_horizon` time steps.
            lower_bounds (np.ndarray): Array of lower confidence bounds for the
                forecasted values.
            upper_bounds (np.ndarray): Array of upper confidence bounds for the
                forecasted values.
    """

    forecast, lower_bounds, upper_bounds = model.predict_future(
        historical_data_df=data.copy(),  # Usar data.copy() para seguridad
        forecast_horizon=forecast_horizon,
        target_col=target_col
    )

    # Imprimir los resultados
    print(f"Forecasted prices for the next {forecast_horizon} days:")
    for i in range(forecast_horizon):
        print(f"Day {i + 1}: {forecast[i]:.4f} (Interval: [{lower_bounds[i]:.4f} - {upper_bounds[i]:.4f}])")

    return forecast, lower_bounds, upper_bounds