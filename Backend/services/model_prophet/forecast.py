import pandas as pd
import numpy as np
from typing import Tuple

# Se asume que ProphetModel está en el mismo directorio o en una ruta accesible.
from .prophet_model import ProphetModel

def forecast_future_prices_prophet(
    model: ProphetModel,
    data: pd.DataFrame,
    forecast_horizon: int = 10,
    target_col: str = 'Close'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pronostica precios futuros utilizando un modelo ProphetModel entrenado.
    Esta versión utiliza un método robusto para construir manualmente los regresores futuros.
    """
    print(f"\n--- Iniciando la llamada al pronóstico Prophet para los próximos {forecast_horizon} días ---")

    if not isinstance(model, ProphetModel):
        raise TypeError(f"El modelo proporcionado no es una instancia de ProphetModel. Recibido: {type(model)}")
    
    if not model.has_fitted:
        raise ValueError("El modelo debe estar entrenado antes de poder usarlo para pronosticar.")

    if model.preprocessor is None:
        raise ValueError("El modelo no contiene un preprocesador, que es necesario para preparar los datos futuros.")

    # 1. Crear el DataFrame futuro con el historial y el horizonte.
    future_df = model.model.make_future_dataframe(periods=forecast_horizon)
    
    # 2. Preparar datos históricos y limpiar zona horaria.
    historical_data = data.copy()
    if historical_data.index.tz is not None:
        historical_data.index = historical_data.index.tz_localize(None)

    # 3. Calcular regresores SOLO en los datos históricos.
    historical_with_regressors = model.preprocessor.add_prophet_regressors(historical_data)
    
    # 4. Limpieza exhaustiva de los regresores históricos.
    historical_with_regressors.bfill(inplace=True)
    historical_with_regressors.ffill(inplace=True)
    
    if historical_with_regressors.isnull().values.any():
        raise ValueError("Los regresores históricos todavía contienen NaNs después del relleno.")

    # 5. Construir manualmente los regresores para las fechas futuras.
    regressor_cols = [col for col in historical_with_regressors.columns if col not in historical_data.columns]
    historical_regressors_clean = historical_with_regressors[regressor_cols]

    last_known_regressors = historical_regressors_clean.iloc[-1:]

    last_historical_date = historical_data.index.max()
    future_only_dates = future_df[future_df['ds'] > last_historical_date]['ds']
    
    future_regressors_df = pd.DataFrame(index=future_only_dates)

    for col in last_known_regressors.columns:
        future_regressors_df[col] = last_known_regressors[col].values[0]
    
    full_regressors = pd.concat([historical_regressors_clean, future_regressors_df])
    
    future_df_with_regressors = future_df.merge(full_regressors, left_on='ds', right_index=True, how='left')
    
    if future_df_with_regressors.isnull().values.any():
        nan_cols = future_df_with_regressors.columns[future_df_with_regressors.isnull().any()].tolist()
        raise ValueError(f"Se encontraron NaNs en las columnas de regresores después de la unión y el relleno final: {nan_cols}.")

    # 6. Realizar la predicción.
    print(f"Realizando predicción para {len(future_df_with_regressors)} periodos...")
    forecast = model.predict(future_df_with_regressors)
    
    # 7. Extraer solo los valores del horizonte de pronóstico.
    future_predictions = forecast.iloc[-forecast_horizon:]
    
    point_forecast = future_predictions['yhat'].values
    lower_bounds = future_predictions['yhat_lower'].values
    upper_bounds = future_predictions['yhat_upper'].values

    print(f"\n--- Resultados del Pronóstico Prophet ---")
    for i in range(len(point_forecast)):
        print(
            f"Día {i + 1}: "
            f"Predicción = {point_forecast[i]:.4f} "
            f"(Intervalo de Confianza: [{lower_bounds[i]:.4f} - {upper_bounds[i]:.4f}])"
        )

    return point_forecast, lower_bounds, upper_bounds
