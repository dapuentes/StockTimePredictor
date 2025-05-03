def forecast_future_prices(model, data, forecast_horizon=10, target_col='Close'):
    """
    Forecast future prices based on the given model and data.

    This function performs a future price prediction for a specified number of
    days using the input model. It processes the data, scales features if
    necessary, utilizes the model's prediction capabilities, and optionally
    scales results back to their original scale. Additionally, it prints the
    forecasted prices and generates a plot for visualization.

    Args:
        model: A predictive model object. This object should have the methods
            `prepare_data`, `predict_future`, and `plot_forecast`. It may also
            include optional scalers, `feature_scaler` and `target_scaler`, used
            for preprocessing input features and postprocessing predicted values.
        data: A DataFrame that contains historical data used for training and
            generating future predictions.
        forecast_horizon: int, optional. The number of days to forecast into
            the future. Defaults to 10.
        target_col: str, optional. The name of the column in the data which
            serves as the target variable for prediction. Defaults to 'Close'.

    Returns:
        list: A list of forecasted future prices corresponding to the next
            `forecast_horizon` days.
    """

    # Preparar los datos más recientes
    processed_data = model.prepare_data(data, target_col=target_col)

    # Obtener la última fila de datos para la predicción
    last_data = processed_data.iloc[-1:]
    X_last = last_data.drop(columns=[target_col])

    if model.feature_scaler:
        X_last_scaled = model.feature_scaler.transform(X_last)
    else:
        X_last_scaled = X_last.values

    # Pronosticar precios futuros
    forecast_scaled = model.predict_future(X_last_scaled, forecast_horizon)

    if model.target_scaler:
        forecast = model.target_scaler.inverse_transform(
            forecast_scaled.reshape(-1, 1)).ravel()
    else:
        forecast = forecast_scaled

    # Imprimir los resultados
    print(f"Forecasted prices for the next {forecast_horizon} days:")
    for i in range(forecast_horizon):
        print(f"Day {i + 1}: {forecast[i]}")

    model.plot_forecast(
        data,
        forecast,
        target_col=target_col
    )

    return forecast