import numpy as np

def forecast_future_prices(model, data, forecast_horizon=10, target_col='Close'):
    """
    Forecast future prices based on the given model and data.

    Args:
        model: A predictive model object.
        data: A DataFrame that contains historical data.
        forecast_horizon: int, optional. The number of days to forecast.
        target_col: str, optional. The name of the column in the data.

    Returns:
        list: A list of forecasted future prices.
    """
    # Preparar los datos más recientes
    processed_data = model.prepare_data(data, target_col=target_col)

    # Obtener la última fila de datos para la predicción
    last_data = processed_data.iloc[-1:]
    X_last = last_data.drop(columns=[target_col])

    # Verificar si hay un escalador de características y aplicarlo
    if hasattr(model, 'feature_scaler') and model.feature_scaler is not None:
        X_last_scaled = model.feature_scaler.transform(X_last)
        print("Aplicando StandardScaler a las características de entrada")
    else:
        X_last_scaled = X_last.values

    # CORRECCIÓN: Crear secuencia para la predicción
    # Necesitamos suficientes datos históricos para crear una secuencia
    # Tomar las últimas 'time_steps' filas para formar una secuencia
    last_n_rows = min(model.time_steps, len(processed_data))
    sequence_data = processed_data.iloc[-last_n_rows:].drop(columns=[target_col])

    if hasattr(model, 'feature_scaler') and model.feature_scaler is not None:
        sequence_data_scaled = model.feature_scaler.transform(sequence_data)
    else:
        sequence_data_scaled = sequence_data.values

    # Crear una única secuencia 3D para la predicción: [1, time_steps, n_features]
    X_sequence = np.expand_dims(sequence_data_scaled, axis=0)

    print(f"Forma de la secuencia de entrada: {X_sequence.shape}")

    try:
        # Pronosticar precios futuros con la secuencia correcta
        forecast_scaled = model.predict_future(X_sequence, forecast_horizon)
        print(f"Valor de forecast_scaled: {forecast_scaled}")

        if hasattr(model, 'target_scaler') and model.target_scaler is not None:
            forecast = model.target_scaler.inverse_transform(
                forecast_scaled.reshape(-1, 1)).ravel()
            print(f"Valor de forecast: {forecast}")
        else:
            forecast = forecast_scaled

        # Imprimir los resultados
        print(f"Forecasted prices for the next {forecast_horizon} days:")
        for i in range(forecast_horizon):
            print(f"Day {i + 1}: {forecast[i]}")

        model.plot_forecast(data, forecast, target_col=target_col)

        return forecast
    except Exception as e:
        print(f"Error en la predicción: {e}")
        print(f"Forma de X_sequence: {X_sequence.shape}")
        print(f"Tipo de X_sequence: {type(X_sequence)}")
        raise