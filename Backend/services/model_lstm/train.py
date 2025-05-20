import pandas as pd
import numpy as np
import os

from services.model_lstm.lstm_model import TimeSeriesLSTMModel


def train_lstm_model(
        data: pd.DataFrame,
        target_col: str = 'Close',
        n_lags: int = 10,
        train_size_ratio: float = 0.8,
        save_model_path_prefix: str = None,
        epochs: int = 100,
        batch_size: int = 32,
        use_hyperparameter_optimization: bool = False,
        hp_strategy: str = "bayesian",  # "random", "bayesian", "hyperband"
        hp_max_trials: int = 10,
        hp_epochs_per_trial: int = 50,
        hp_project_name: str = "lstm_tuning_via_train_script",
        initial_lstm_units: int = 50,
        initial_lstm_layers: int = 1,
        initial_dropout_rate: float = 0.2,
        initial_learning_rate: float = 0.001
):
    """
    Entrena un modelo LSTM para series temporales.

    Args:
        data (pd.DataFrame): DataFrame de entrada con datos históricos.
        target_col (str): Nombre de la columna objetivo.
        n_lags (int): Número de lags a generar como características.
        train_size_ratio (float): Proporción del dataset para entrenamiento.
        save_model_path_prefix (str, optional): Prefijo para la ruta donde guardar el modelo.
                                                Si es None, el modelo no se guarda.
        epochs (int): Número de épocas para el entrenamiento (si no se usa optimización de HP).
        batch_size (int): Tamaño del lote para el entrenamiento.
        use_hyperparameter_optimization (bool): Si es True, realiza optimización de hiperparámetros.
        hp_strategy (str): Estrategia para Keras Tuner ('random', 'bayesian', 'hyperband').
        hp_max_trials (int): Máximo de intentos para el tuner.
        hp_epochs_per_trial (int): Épocas por cada intento del tuner.
        hp_project_name (str): Nombre del proyecto para Keras Tuner.
        initial_lstm_units (int): Unidades LSTM iniciales (usadas si no hay optimización o como default en tuner).
        initial_lstm_layers (int): Capas LSTM iniciales.
        initial_dropout_rate (float): Tasa de dropout inicial.
        initial_learning_rate (float): Tasa de aprendizaje inicial.

    Returns:
        tuple: (
            TimeSeriesLSTMModel: El modelo LSTM entrenado.
            list: Nombres de las características utilizadas.
            np.ndarray: Residuales del conjunto de entrenamiento (y_true - y_pred).
            list: Fechas correspondientes a los residuales.
        )
    """
    print(f"--- Iniciando entrenamiento del modelo LSTM para {target_col} ---")
    print(
        f"Configuración: n_lags={n_lags}, train_size_ratio={train_size_ratio}, epochs={epochs}, batch_size={batch_size}")
    print(f"Optimización de Hiperparámetros: {use_hyperparameter_optimization}")
    if use_hyperparameter_optimization:
        print(f"Estrategia HP: {hp_strategy}, Max Trials: {hp_max_trials}, Epochs/Trial: {hp_epochs_per_trial}")

    # 1. Instanciar el modelo
    #    Los hiperparámetros pasados aquí pueden ser sobrescritos por Keras Tuner si se usa.
    model = TimeSeriesLSTMModel(
        n_lags=n_lags,
        units=initial_lstm_units,
        layers=initial_lstm_layers,
        dropout_rate=initial_dropout_rate,
        learning_rate=initial_learning_rate
    )
    print("TimeSeriesLSTMModel instanciado.")

    # 2. Preparar datos (ingeniería de características)
    #    Esto incluye añadir lags y otras características definidas en model.prepare_data()
    print("Preparando datos (ingeniería de características)...")
    processed_data_df = model.prepare_data(data.copy(), target_col=target_col)
    print(f"Datos procesados: {processed_data_df.shape}")
    # print(processed_data_df.head())

    if processed_data_df.empty:
        raise ValueError(
            "Los datos procesados están vacíos. Revisa los datos de entrada y los parámetros de preparación.")
    if len(processed_data_df) < model.n_lags * 2:  # Necesita suficientes datos para secuencias y división
        raise ValueError(
            f"Datos procesados insuficientes ({len(processed_data_df)} filas) para n_lags ({model.n_lags}) y división train/test.")

    # 3. Dividir datos en entrenamiento y prueba
    #    La validación se puede manejar dentro de fit o optimize_hyperparameters
    train_split_idx = int(len(processed_data_df) * train_size_ratio)
    train_df = processed_data_df.iloc[:train_split_idx]
    test_df = processed_data_df.iloc[train_split_idx:]

    print(f"Tamaño del DataFrame de entrenamiento: {train_df.shape}")
    print(f"Tamaño del DataFrame de prueba: {test_df.shape}")

    if train_df.empty or len(train_df) <= model.n_lags:
        raise ValueError(
            f"El conjunto de entrenamiento está vacío o es demasiado pequeño ({len(train_df)} filas) después de la división para n_lags ({model.n_lags}).")
    if test_df.empty or len(test_df) <= model.n_lags:
        print(
            f"Advertencia: El conjunto de prueba está vacío o es demasiado pequeño ({len(test_df)} filas) después de la división para n_lags ({model.n_lags}). La evaluación podría no ser significativa.")

    # 4. Entrenamiento del modelo
    if use_hyperparameter_optimization:
        print("Iniciando optimización de hiperparámetros...")
        # Keras Tuner necesita un conjunto de validación. Podemos dividir train_df o pasar test_df.
        # Por simplicidad, Keras Tuner puede usar una fracción de X_train_seq para validación si validation_data no se pasa.
        # Opcionalmente, podríamos crear un validation_df explícito aquí.
        # Para este ejemplo, dejaremos que optimize_hyperparameters maneje la división de validación si es necesario.
        model.optimize_hyperparameters(
            train_df=train_df,
            target_col=target_col,
            validation_df=test_df if not test_df.empty and len(test_df) > model.n_lags else None,
            # Usar test_df para validación en tuning si es adecuado
            strategy=hp_strategy,
            max_trials=hp_max_trials,
            epochs_per_trial=hp_epochs_per_trial,
            project_name=hp_project_name
        )
        print(f"Optimización de hiperparámetros completada. Mejores HPs: {model.best_hyperparameters}")
        # El modelo ya está reentrenado con los mejores HPs dentro de optimize_hyperparameters
    else:
        print("Iniciando entrenamiento del modelo (sin optimización de HP)...")
        model.fit(
            train_df=train_df,
            target_col=target_col,
            validation_df=test_df if not test_df.empty and len(test_df) > model.n_lags else None,
            # Usar test_df para validación si es adecuado
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        print("Entrenamiento del modelo completado.")

    # (Opcional) Calcular métricas en el conjunto de entrenamiento para referencia
    print("Calculando métricas en el conjunto de entrenamiento...")
    # Necesitamos predecir sobre los datos de entrenamiento para obtener residuales
    # model.predict() espera un DataFrame que pasará por model.preprocess_data y luego sequence_generator

    # Para obtener y_train_true_original y y_train_pred_original para residuales:
    X_train_scaled_for_residuals, y_train_scaled_for_residuals = model.preprocess_data(train_df, target_col,
                                                                                       is_training=False)  # is_training=False para no re-ajustar scalers
    X_train_seq_for_residuals, y_train_seq_true_scaled = model.sequence_generator.create_sequences(
        X_train_scaled_for_residuals, y_train_scaled_for_residuals)

    residuals = np.array([])
    residual_dates = []

    from utils.evaluation import evaluate_regression
    if X_train_seq_for_residuals.shape[0] > 0:
        y_train_pred_scaled = model.model.predict(X_train_seq_for_residuals, verbose=0)

        y_train_pred_original = model.target_scaler.inverse_transform(y_train_pred_scaled).flatten()
        y_train_true_original = model.target_scaler.inverse_transform(y_train_seq_true_scaled).flatten()

        train_metrics = evaluate_regression(y_train_true_original, y_train_pred_original)
        print(f"Métricas en datos de entrenamiento (después de crear secuencias): {train_metrics}")

        residuals = y_train_true_original - y_train_pred_original
        # Las fechas de los residuales corresponden a las fechas de y_train_seq_true_scaled
        # El DataFrame original para y_train_seq_true_scaled es train_df.
        # Las secuencias empiezan después de n_lags.
        # El target y_data[i + self.n_lags] corresponde a la fila i + n_lags del DataFrame original (después de procesado y escalado)
        # train_df_for_residual_dates = train_df.iloc[model.n_lags : model.n_lags + len(y_train_true_original)]
        # O de forma más robusta, tomar las últimas N fechas de train_df donde N es len(residuals)
        if len(residuals) > 0:
            residual_dates_pd_index = train_df.index[model.n_lags: model.n_lags + len(residuals)]
            residual_dates = residual_dates_pd_index.strftime('%Y-%m-%d').tolist() if isinstance(
                residual_dates_pd_index, pd.DatetimeIndex) else residual_dates_pd_index.tolist()

        print(f"Calculados {len(residuals)} residuales del entrenamiento.")
    else:
        print("No se pudieron generar secuencias del conjunto de entrenamiento para calcular residuales.")

    # 5. Evaluar el modelo en el conjunto de prueba
    if not test_df.empty and len(test_df) > model.n_lags:
        print("Evaluando el modelo en el conjunto de prueba...")
        test_metrics = model.evaluate(test_df=test_df, target_col=target_col)
        print(f"Métricas en datos de prueba: {model.metrics}")  # model.metrics se actualiza en model.evaluate()
    else:
        print("Conjunto de prueba vacío o demasiado pequeño, omitiendo evaluación.")
        model.metrics = {metric: np.nan for metric in ['MSE', 'RMSE', 'MAE', 'MAPE']}

    # 6. Guardar el modelo
    if save_model_path_prefix:
        print(f"Guardando el modelo en el prefijo: {save_model_path_prefix}...")
        training_end_date_str = None
        if isinstance(data.index, pd.DatetimeIndex) and not data.empty:
            training_end_date_str = data.index[-1].strftime("%Y-%m-%d")
        elif not data.empty:  # Si no es DatetimeIndex pero hay datos, usa un placeholder o el último índice
            training_end_date_str = str(data.index[-1])

        model.save_model(model_path_prefix=save_model_path_prefix, training_end_date=training_end_date_str)
        print("Modelo guardado.")
    else:
        print("No se proporcionó save_model_path_prefix, el modelo no se guardará.")

    # 7. Retornar resultados
    #    feature_names se establece en model.preprocess_data durante el primer escalado (is_training=True)
    feature_names_from_model = model.feature_names if model.feature_names else []

    print("--- Entrenamiento del modelo LSTM finalizado ---")
    return model, feature_names_from_model, residuals, residual_dates