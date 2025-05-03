from .rf_model2 import TimeSeriesRandomForestModel


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

    from Backend.utils import scale_data

    model = TimeSeriesRandomForestModel(n_lags=n_lags)

    # Preparar los datos
    processed_data = model.prepare_data(data, target_col=target_col)
    print(len(data), len(processed_data))
    print(f"Processed data shape: {processed_data.shape}")
    print(processed_data.head())

    if processed_data.empty:
        raise ValueError("Processed data is empty. Check the input data and parameters.")

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

    # Modelo de entrenamiento
    y_train_pred_scaled = model.best_pipeline_.predict(X_train_scaled)

    if model.target_scaler:
        y_train_pred = model.target_scaler.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
    else:
        y_train_pred = y_train_pred_scaled

    from Backend.utils import evaluate_regression
    train_metrics = evaluate_regression(y_train.flatten(), y_train_pred)
    print(f"Train metrics: {train_metrics}")

    # Evaluar el modelo
    model.evaluate(X_test_scaled, y_test)
    print(f"Model metrics: {model.metrics}")

    if save_model_path is not None:
        training_end_date = data.index[-1].strftime("%Y-%m-%d")  # Ultima fecha de entrenamiento
        model.save_model(save_model_path, training_end_date)
        print(f"Model saved to {save_model_path}")

    return model, feature_names