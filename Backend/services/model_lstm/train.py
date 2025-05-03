from .lstm_model import TimeSeriesLSTMModel

def train_lstm_model(
        data, n_lags=10, target_col='Close', train_size=0.8, validation_size=0.2,
        batch_size=32, epochs=100, save_model_path=None
):

    from Backend.utils import scale_data

    # Inicializar el modelo LSTM
    model = TimeSeriesLSTMModel(n_lags=n_lags)

    # Preparar los datos
    processed_data = model.prepare_data(data, target_col=target_col)
    print(f"Processed data shape: {processed_data.shape}")
    print(processed_data.head())

    # Dividir los datos en conjuntos de entrenamiento y prueba
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

    # Dividir los datos de entrenamiento en conjuntos de entrenamiento y validación
    validation_size = int(len(X_train_scaled) * (1 - validation_size)) # Es 1 - validation_size porque validation_size es el porcentaje de datos de validación
    X_val = X_train_scaled[validation_size:]
    y_val = y_train_scaled[validation_size:]
    X_train_final = X_train_scaled[:validation_size]
    y_train_final = y_train_scaled[:validation_size]

    print(f"Final train data shape: {X_train_final.shape}")
    print(f"Validation data shape: {X_val.shape}")

    # Optimizar los hiperparámetros con el conjunto de datos
    model.optimize_hyperparameters(
        X_train_final,
        y_train_final,
        X_val,
        y_val,
        feature_names=feature_names
    )

    print(f"Best hyperparameters: units={model.units}, layers={model.layers}, dropout={model.dropout}, learning_rate={model.learning_rate}")

    # Entrenando con el conjunto de datos usando los mejores hiperparámetros
    model.fit(
        X_train_scaled, y_train_scaled,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_size,
        verbose=1
    )

    # Evaluar el modelo
    metrics = model.evaluate(X_test_scaled, y_test)
    print(f"Model metrics: {model.metrics}")

    if save_model_path is not None:
        model.save_model(save_model_path)
        print(f"Model saved to {save_model_path}")

    return model




