import pandas as pd


def forecast_future_prices(model, data: pd.DataFrame, forecast_horizon: int = 10, target_col: str = 'Close'):
    """
    Forecast future prices based on the given model and data.
    ... (la descripción de la función sigue igual) ...
    """

    # --- INICIO DE CAMBIOS ---

    # Las siguientes líneas para preparar X_last_scaled ya NO son necesarias:
    # # Preparar los datos más recientes
    # processed_data = model.prepare_data(data, target_col=target_col) # Esto se hará dentro de predict_future

    # # Obtener la última fila de datos para la predicción
    # last_data = processed_data.iloc[-1:]
    # X_last = last_data.drop(columns=[target_col])

    # if model.feature_scaler:
    #     X_last_scaled = model.feature_scaler.transform(X_last)
    # else:
    #     X_last_scaled = X_last.values

    # Nueva llamada a model.predict_future:
    # model.predict_future ahora espera el DataFrame histórico completo y el target_col,
    # y devuelve directamente los valores desescalados.
    forecast, lower_bounds, upper_bounds = model.predict_future(
        historical_data_df=data.copy(),  # Usar data.copy() para seguridad
        forecast_horizon=forecast_horizon,
        target_col=target_col
    )

    # El bloque para desescalar los resultados ya NO es necesario,
    # porque predict_future ya los devuelve desescalados.
    # if model.target_scaler:
    #     forecast = model.target_scaler.inverse_transform(
    #         forecast_scaled.reshape(-1, 1)).ravel()
    #     lower_bounds = model.target_scaler.inverse_transform(
    #         lower_bounds_scaled.reshape(-1, 1)).ravel()
    #     upper_bounds = model.target_scaler.inverse_transform(
    #         upper_bounds_scaled.reshape(-1, 1)).ravel()
    # else:
    #     forecast = forecast_scaled # Ya no se reciben forecast_scaled, etc.
    #     lower_bounds = lower_bounds_scaled
    #     upper_bounds = upper_bounds_scaled

    # --- FIN DE CAMBIOS ---

    # Imprimir los resultados (esta parte sigue igual)
    print(f"Forecasted prices for the next {forecast_horizon} days:")
    for i in range(forecast_horizon):
        print(f"Day {i + 1}: {forecast[i]:.4f} (Interval: [{lower_bounds[i]:.4f} - {upper_bounds[i]:.4f}])")

    return forecast, lower_bounds, upper_bounds