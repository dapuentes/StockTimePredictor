def forecast_future_prices(model, data, forecast_horizon=10, target_col='Close'):
    """
    Pronosticar precios futuros utilizando el modelo entrenado

    Parámetros:
    - model: Modelo entrenado
    - data: DataFrame con los datos más recientes
    - forecast_horizon: Horizonte de pronóstico en días (el valor predeterminado es 10)
    - target_col: Nombre de la columna objetivo para la predicción (el valor predeterminado es 'Close')

    Devuelve:
    - Array de precios futuros pronosticados
    """
    
    # Print model properties for debugging
    print("Model properties:", dir(model))
    
    # Preparar los datos más recientes
    processed_data = model.prepare_data(data, target_col=target_col)

    # Obtener la última fila de datos para la predicción
    last_data = processed_data.iloc[-1:]
    X_last = last_data.drop(columns=[target_col])
    
    if model.feature_scaler:
        # Get the column names the scaler was trained on
        if not hasattr(model, 'feature_names'):
            # If feature_names not stored, try to get them from the scaler
            try:
                feature_names = model.feature_scaler.feature_names_in_
                # Only use available columns
                common_cols = [col for col in feature_names if col in X_last.columns]
                X_last = X_last[common_cols]
            except AttributeError:
                print("Warning: No feature names found in model. Using all available features.")
        else:
            # Use stored feature names
            common_cols = [col for col in model.feature_names if col in X_last.columns]
            X_last = X_last[common_cols]
            
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