# Backend/services/model_xgb/train.py

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf, pacf

# Imports para el modelo y utilidades
try:
    # Import relativo para cuando se ejecuta como parte del paquete del servicio
    from .xgb_model import TimeSeriesXGBoostModel
except ImportError:
    # Fallback para poder ejecutar el script directamente en pruebas
    from xgb_model import TimeSeriesXGBoostModel

from utils.evaluation import evaluate_regression
from utils.preprocessing import scale_data_universal


def train_xgb_model(
    data: pd.DataFrame,
    n_lags: int = 10,
    target_col: str = 'Close',
    train_size_ratio: float = 0.7,
    save_model_path_prefix: str = None
):
    """
    Entrena y optimiza un modelo XGBoost para pronóstico de series temporales.
    """
    print("--- Iniciando el Pipeline de Entrenamiento del Modelo XGBoost ---")

    # 1. Crear la instancia del modelo
    print(f"1. Creando preprocesador XGBoost con n_lags={n_lags}...")
    model = TimeSeriesXGBoostModel(n_lags=n_lags)

    # 2. Ingeniería de características
    print(f"2. Aplicando ingeniería de características al DataFrame (target: '{target_col}')...")
    engineered_df = model.prepare_data(data.copy(), target_col_name=target_col)
    print(f'   -> Datos Originales: {len(data)} filas')
    print(f'   -> Datos Ingenierizados: {len(engineered_df)} filas')

    if engineered_df.isna().any().any():
        raise ValueError("Se encontraron valores NaN después de la ingeniería de características.")

    # 3. Dividir los datos
    print("3. Dividiendo los datos en conjuntos de entrenamiento y prueba...")
    train_end_idx = int(len(engineered_df) * train_size_ratio)
    train_df = engineered_df.iloc[:train_end_idx]
    test_df = engineered_df.iloc[train_end_idx:]
    print(f"   -> Tamaño Entrenamiento: {len(train_df)}, Prueba: {len(test_df)}")

    # 4. Separar características (X) y objetivo (y)
    print(f'4. Separando características y objetivo...')
    feature_names_for_split = model.feature_names
    X_train = train_df[feature_names_for_split]
    y_train = train_df[target_col]
    X_test = test_df[feature_names_for_split]
    y_test = test_df[target_col]
    train_indices = train_df.index
    actual_feature_names_in_X_train = X_train.columns.tolist()

    # 5. Ajustar escaladores y transformar datos
    print("5. Ajustando y aplicando escaladores...")
    # Obtener los objetos escaladores (aún no ajustados)
    feature_scaler, target_scaler = model.preprocessor.get_scalers()

    # La función ajusta los escaladores 'in-place' y devuelve los datos transformados
    X_train_scaled_np, X_test_scaled_np, y_train_scaled_np, y_test_scaled_np, _, _ = scale_data_universal(
        X_train, X_test, y_train, y_test, feature_scaler, target_scaler
    )
    # En este punto, las variables 'feature_scaler' y 'target_scaler' ya contienen los escaladores ajustados.

    print("   -> Datos escalados correctamente.")

    # Asignar los escaladores (ya ajustados) a la instancia del modelo para que puedan guardarse
    model.feature_scaler = feature_scaler
    model.target_scaler = target_scaler
    if hasattr(model.preprocessor, 'feature_scaler'):
        model.preprocessor.feature_scaler = feature_scaler
    if hasattr(model.preprocessor, 'target_scaler'):
        model.preprocessor.target_scaler = target_scaler

    # 6. Optimización de Hiperparámetros
    print('6. Iniciando optimización de hiperparámetros...')
    model.optimize_hyperparameters(
        X_train_scaled_np,
        y_train_scaled_np.ravel(),  # Asegurar que sea 1D
        feature_names=actual_feature_names_in_X_train
    )
    print(f"   -> Mejores hiperparámetros encontrados: {model.best_params_}")

    # 7. Realizar predicciones sobre los datos de entrenamiento para obtener métricas
    print("\n7. Realizando y evaluando predicciones sobre el conjunto de entrenamiento...")
    y_train_pred_scaled = model.predict(X_train_scaled_np)

    # Usar el 'target_scaler' (ya ajustado) para desescalar las predicciones
    if target_scaler:
        y_train_pred_unscaled = target_scaler.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
        print("   -> Predicciones de entrenamiento desescaladas correctamente.")
        
        # 8. Calcular las métricas de entrenamiento con los datos correctos
        train_metrics = evaluate_regression(y_train.values, y_train_pred_unscaled)
        print(f"   -> Métricas de entrenamiento: {train_metrics}")
    else:
        y_train_pred_unscaled = y_train_pred_scaled.ravel() # Fallback si no hay scaler
        print("   -> ADVERTENCIA: No se pudo desescalar predicciones de entrenamiento (target_scaler no disponible).")

    # 9. Calcular Residuales
    print("\n9. Calculando residuales del entrenamiento...")
    residuals = y_train.values - y_train_pred_unscaled
    residuals_dates = train_indices.tolist()
    print(f"   -> Media de residuales: {np.mean(residuals):.6f}")

    # 10. Calcular ACF/PACF
    print("\n10. Calculando funciones de autocorrelación (ACF y PACF)...")
    nlags = min(40, len(residuals) // 2 - 1) if len(residuals) > 2 else 0
    if nlags > 0:
        acf_values, confint_acf = acf(residuals, nlags=nlags, alpha=0.05, fft=False)
        pacf_values, confint_pacf = pacf(residuals, nlags=nlags, alpha=0.05, method='ywm')
        print("   -> ACF y PACF calculados.")
    else:
        acf_values, confint_acf, pacf_values, confint_pacf = (None, None, None, None)
        print("   -> No hay suficientes residuales para calcular ACF/PACF.")

    # 11. Evaluación Final sobre el conjunto de prueba
    print("\n11. Evaluando el modelo final en el conjunto de prueba...")
    # Pasar el 'target_scaler' (ya ajustado) al método de evaluación
    model.evaluate(X_test_scaled_np, y_test, target_scaler)
    print(f"   -> Métricas finales del modelo: {model.metrics}")

    # 12. Guardar el Modelo
    if save_model_path_prefix is not None:
        print(f"\n12. Guardando modelo en prefijo: {save_model_path_prefix}")
        training_end_date_str = data.index[-1].strftime("%Y-%m-%d") if isinstance(data.index, pd.DatetimeIndex) else None
        model.save_model(save_model_path_prefix, training_end_date_str)
        print("   -> Modelo guardado exitosamente.")
    else:
        print("\n12. No se guardó el modelo (save_model_path_prefix es None).")

    print("\n--- Pipeline de Entrenamiento XGBoost Completado Exitosamente ---")

    return model, model.feature_names, residuals, residuals_dates, acf_values, pacf_values, confint_acf, confint_pacf