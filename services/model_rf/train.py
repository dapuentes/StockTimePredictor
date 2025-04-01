from .rf_model2 import TimeSeriesRandomForestModel

def train_ts_model(data, n_lags=10, target_col='Close', train_size=0.8, save_model_path=None):
    """
    Entrenar un modelo de Random Forest para datos de series temporales

    Parámetros:
    - data: DataFrame con los datos
    - n_lags: Número de características de rezago a crear
    - target_col: Nombre de la columna objetivo para la predicción (el valor predeterminado es 'Close')
    - train_size: Proporción del conjunto de datos a usar para el entrenamiento (el valor predeterminado es 0.8)
    - save_model_path: Ruta para guardar el modelo entrenado (el valor predeterminado es None, no se guarda)

    Devuelve:
    - Modelo entrenado con sus métricas de rendimiento
    """
    
    from utils.preprocessing import scale_data

    model = TimeSeriesRandomForestModel(n_lags=n_lags)

    # Preparar los datos
    processed_data = model.prepare_data(data, target_col=target_col)
    print(f"Processed data shape: {processed_data.shape}")
    print(processed_data.head())

    train_size = int(len(processed_data) * train_size)
    train_data = processed_data.iloc[:train_size]
    test_data = processed_data.iloc[train_size:]
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    # Separar características y objetivo
    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col].values.reshape(-1, 1)
    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col].values.reshape(-1, 1)

    # Guardar los nombres de las características antes de escalar
    feature_names = X_train.columns.tolist()
    print(f"Feature names: {feature_names}")
    
    # Escalar los datos
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, feature_scaler, target_scaler = scale_data(
        X_train, X_test, y_train, y_test
    )
    print(f"Scaled train data shape: {X_train_scaled.shape}")
    print(f"Scaled test data shape: {X_test_scaled.shape}")

    model.feature_scaler = feature_scaler
    model.target_scaler = target_scaler

    # Modelo optimizado - pasar explícitamente los nombres de características
    model.optimize_hyperparameters(
        X_train_scaled,
        y_train_scaled.ravel(),
        feature_names=feature_names
    )
    print(f"Best parameters: {model.best_params_}")

    # Evaluar el modelo
    model.evaluate(X_test_scaled, y_test)
    print(f"Model metrics: {model.metrics}")

    if save_model_path is not None:
        model.save_model(save_model_path)
        print(f"Model saved to {save_model_path}")
    
    return model