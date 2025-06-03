from .rf_model import TimeSeriesRandomForestModel
from utils.preprocessing import scale_data_universal
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
import numpy as np


def train_ts_model(data, n_lags=10, target_col='Close', train_size=0.8, save_model_path=None):
    """
    Trains a time series forecasting model using a Random Forest algorithm.

    This function prepares and processes the time series data, splits it into training
    and testing sets, scales the features and target variable, optimizes hyperparameters,
    evaluates the model on the test data, and optionally saves the trained model if a
    file path is provided.

    Args:
        data: pandas.DataFrame
            The input time series data containing the features and target column.
        n_lags: int, optional
            The number of previous time steps to use as features for the target variable.
            Defaults to 10.
        target_col: str, optional
            The name of the column in the dataset representing the target variable.
            Defaults to 'Close'.
        train_size: float, optional
            The proportion of the data to use for training the model. The remaining data
            will be used for testing. Must be a value between 0 and 1. Defaults to 0.8.
        save_model_path: str, optional
            The file path to save the trained model. If None, the model will not be saved.
            Defaults to None.

    Returns:
        TimeSeriesRandomForestModel
            The trained time series Random Forest model.
    """
    
    print("--- Iniciando el Pipeline de Entrenamiento del Modelo Random Forest ---")
    
    # 1. Crear e inicializar el modelo
    print(f"1. Instanciando modelo Random Forest con n_lags={n_lags}")
    model = TimeSeriesRandomForestModel(n_lags=n_lags)

    # 2. Preparar los datos
    print("2. Realizando ingeniería de características...")
    processed_data = model.prepare_data(data, target_col=target_col)
    print(f"   -> Datos originales: {len(data)} filas")
    print(f"   -> Datos procesados: {len(processed_data)} filas")
    print(f"   -> Forma de datos procesados: {processed_data.shape}")
    print(f"   -> Columnas después del procesamiento: {list(processed_data.columns)}")

    # 2.5. Inspección de valores inválidos
    print("\n2.5. Inspeccionando el DataFrame 'processed_data' en busca de valores inválidos...")
    invalid_cols = []
    for col in processed_data.columns:
        if processed_data[col].isnull().any() or np.isinf(processed_data[col]).any():
            invalid_cols.append(col)

    if invalid_cols:
        print(f"   -> ¡ERROR! Se encontraron valores NaN o Inf en las siguientes columnas: {invalid_cols}")
        print("   -> Mostrando algunas de las filas problemáticas:")
        print(processed_data[processed_data.isin([np.nan, np.inf, -np.inf]).any(axis=1)])
        raise ValueError(f"Procesamiento fallido. Columnas con valores inválidos: {invalid_cols}")
    else:
        print("   -> ¡Inspección completada! El DataFrame 'processed_data' está limpio.")

    if processed_data.empty:
        raise ValueError("Processed data is empty. Check the input data and parameters.")

    # 3. Dividir los datos
    print(f"\n3. Dividiendo los datos en {train_size * 100}% para entrenamiento y {(1-train_size) * 100}% para prueba...")
    train_size_idx = int(len(processed_data) * train_size)
    train_data = processed_data.iloc[:train_size_idx]
    test_data = processed_data.iloc[train_size_idx:]
    print(f"   -> Tamaño del conjunto de entrenamiento: {len(train_data)} filas")
    print(f"   -> Tamaño del conjunto de prueba: {len(test_data)} filas")

    # 4. Separar características y objetivo
    print(f"\n4. Separando características y variable objetivo ('{target_col}')...")
    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col].values.reshape(-1, 1)
    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col].values.reshape(-1, 1)

    # Guardar los nombres de las características antes de escalar
    feature_names = X_train.columns.tolist()
    print(f"   -> Número de características: {len(feature_names)}")
    print(f"   -> Nombres de características: {feature_names[:10]}{'...' if len(feature_names) > 10 else ''}")

    # 5. Escalar los datos
    print("\n5. Escalando características y variable objetivo...")
    feature_scaler, target_scaler = model.preprocessor.get_scalers()
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, _, _ = scale_data_universal(
        X_train, X_test, y_train, y_test, feature_scaler, target_scaler
    )
    print(f"   -> Forma de características de entrenamiento escaladas: {X_train_scaled.shape}")
    print(f"   -> Forma de características de prueba escaladas: {X_test_scaled.shape}")
    print(f"   -> Forma de objetivo de entrenamiento escalado: {y_train_scaled.shape}")

    model.feature_scaler = feature_scaler
    model.target_scaler = target_scaler

    # 6. Optimizar hiperparámetros
    print("\n6. Iniciando optimización de hiperparámetros...")
    model.optimize_hyperparameters(
        X_train_scaled,
        y_train_scaled.ravel(),
        feature_names=feature_names
    )
    print(f"   -> Mejores hiperparámetros encontrados: {model.best_params_}")

    # 7. Entrenamiento y predicciones en conjunto de entrenamiento
    print("\n7. Realizando predicciones sobre el conjunto de entrenamiento...")
    y_train_pred_scaled = model.best_pipeline_.predict(X_train_scaled)

    if model.target_scaler:
        y_train_pred = model.target_scaler.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
        print("   -> Predicciones desescaladas aplicando target_scaler")
    else:
        y_train_pred = y_train_pred_scaled
        print("   -> No se aplicó desescalado (target_scaler no disponible)")
    
    # 8. Calcular métricas de entrenamiento
    print("\n8. Calculando métricas de rendimiento en entrenamiento...")
    from utils.evaluation import evaluate_regression
    train_metrics = evaluate_regression(y_train.flatten(), y_train_pred)
    print(f"   -> Métricas de entrenamiento: {train_metrics}")

    # 9. Calcular residuales
    print("\n9. Calculando residuales del entrenamiento...")
    residuals = y_train.flatten() - y_train_pred
    residuals_dates = train_data.index.tolist()
    print(f"   -> Residuales calculados. Forma: {residuals.shape}")
    print(f"   -> Media de residuales: {np.mean(residuals):.6f}")
    print(f"   -> Desviación estándar de residuales: {np.std(residuals):.6f}")

    # 10. Análisis de autocorrelación
    print("\n10. Calculando funciones de autocorrelación (ACF y PACF)...")
    nlags = 40
    acf_values, confint_acf = acf(residuals, nlags=nlags, alpha=0.05)
    pacf_values, confint_pacf = pacf(residuals, nlags=nlags, alpha=0.05)
    print(f"   -> ACF y PACF calculados para {nlags} lags")

    # 11. Evaluar en conjunto de prueba
    print("\n11. Evaluando el modelo final en el conjunto de prueba...")
    model.evaluate(X_test_scaled, y_test)
    print(f"   -> Métricas finales del modelo: {model.metrics}")

    # 12. Guardar el modelo
    if save_model_path is not None:
        print(f"\n12. Guardando modelo en: {save_model_path}")
        training_end_date = data.index[-1].strftime("%Y-%m-%d")  # Ultima fecha de entrenamiento
        model.save_model(save_model_path, training_end_date)
        print(f"   -> Modelo guardado exitosamente")
    else:
        print("\n12. No se guardó el modelo (save_model_path es None)")

    print("\n--- Pipeline de Entrenamiento Random Forest Completado Exitosamente ---")

    return model, feature_names, residuals, residuals_dates, acf_values, pacf_values, confint_acf, confint_pacf