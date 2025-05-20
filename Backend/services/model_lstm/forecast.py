import pandas as pd

def forecast_future_prices(
    model,  # Instancia de TimeSeriesLSTMModel entrenada
    historical_data_df: pd.DataFrame,
    forecast_horizon: int = 10,
    target_col: str = 'Close'
):
    """
    Pronostica precios futuros utilizando un modelo LSTM entrenado.

    Args:
        model (TimeSeriesLSTMModel): Instancia del modelo LSTM entrenado.
        historical_data_df (pd.DataFrame): DataFrame con los datos históricos necesarios
                                           para generar la última secuencia de entrada para el modelo.
        forecast_horizon (int): Número de períodos futuros a pronosticar.
        target_col (str): Nombre de la columna objetivo.

    Returns:
        tuple: (
            np.ndarray: Array con los valores pronosticados.
            None: Placeholder para límites inferiores (no implementado para LSTM aquí).
            None: Placeholder para límites superiores (no implementado para LSTM aquí).
        )
    """
    print(f"--- Iniciando pronóstico LSTM para los próximos {forecast_horizon} períodos de '{target_col}' ---")

    if not hasattr(model, 'predict_future'):
        raise AttributeError("El objeto 'model' no tiene un método 'predict_future'. Asegúrate de pasar una instancia entrenada de TimeSeriesLSTMModel.")
    if not isinstance(historical_data_df, pd.DataFrame):
        raise TypeError("historical_data_df debe ser un DataFrame de pandas.")
    if historical_data_df.empty:
        raise ValueError("historical_data_df no puede estar vacío.")

    # El método predict_future del modelo LSTM se encarga de la lógica de predicción recursiva.
    # Devuelve solo las predicciones puntuales.
    try:
        forecast_values = model.predict_future(
            historical_data_df=historical_data_df.copy(), # Usar una copia para seguridad
            forecast_horizon=forecast_horizon,
            target_col=target_col
        )
        print(f"Pronóstico LSTM para '{target_col}' para los próximos {forecast_horizon} períodos:")
        if forecast_values is not None and len(forecast_values) > 0:
            for i, value in enumerate(forecast_values):
                print(f"Período {i + 1}: {value:.4f}")
        else:
            print("No se generaron valores de pronóstico.")

        # Por ahora, no se implementan límites inferiores y superiores para LSTM.
        lower_bounds = None
        upper_bounds = None

        print("--- Pronóstico LSTM finalizado ---")
        return forecast_values, lower_bounds, upper_bounds

    except Exception as e:
        print(f"Error durante el proceso de pronóstico LSTM: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None



