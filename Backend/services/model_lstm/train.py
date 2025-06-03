import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf

# Importar las clases y la fábrica que hemos construido
from utils.preprocessing import PreprocessorFactory
from .lstm_model import TimeSeriesLSTMModel
from utils.preprocessing import split_data_universal, scale_data_universal

import tensorflow as tf

# Verificar GPU al inicio
print("=== Verificación de GPU ===")
print(f"Versión de TensorFlow: {tf.__version__}")
print(f"GPUs disponibles: {tf.config.list_physical_devices('GPU')}")
print(f"Construido con soporte CUDA: {tf.test.is_built_with_cuda()}")

# Configurar uso de memoria GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Permitir crecimiento gradual de memoria
        tf.config.experimental.set_memory_growth(gpus[0], True)
        
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0], 
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)]  # 3GB límite
        )
    except RuntimeError as e:
        print(f"Error configurando GPU: {e}")
else:
    print("No se detectaron GPUs. Usando CPU.")
print("===========================\n")


def train_lstm_model(
        data: pd.DataFrame,
        target_col: str = 'Close',
        # --- Parámetros del Preprocesador ---
        sequence_length: int = 60,
        n_lags: int = 5,
        # --- Parámetros del Modelo ---
        lstm_units: int = 50,
        dropout_rate: float = 0.2,
        # --- Parámetros de Entrenamiento ---
        train_size: float = 0.8,
        validation_size: float = 0.1,
        epochs: int = 50,
        batch_size: int = 32,
        optimize_params: bool = True,
        save_model_path: str = None,
        # Paciencia
        patience: int = 10
        
):
    print("--- Iniciando el Pipeline de Entrenamiento del Modelo LSTM ---")

    # 1. Crear el preprocesador LSTM específico usando la fábrica
    print(f"1. Creando preprocesador con sequence_length={sequence_length} y n_lags={n_lags}")
    lstm_preprocessor = PreprocessorFactory.create_preprocessor(
        'lstm', sequence_length=sequence_length, n_lags=n_lags
    )

    # 2. Inyectar el preprocesador en el modelo al crearlo
    print(f"2. Instanciando modelo LSTM con units={lstm_units} y dropout={dropout_rate}")
    model = TimeSeriesLSTMModel(
        preprocessor=lstm_preprocessor,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate
    )

    # 3. Preparar las características (features)
    print("3. Realizando ingeniería de características...")
    processed_data = model.preprocessor.prepare_data(data, target_col=target_col)

    print("\n3.5. Inspeccionando el DataFrame 'processed_data' en busca de valores inválidos...")
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

    # 4. Dividir datos en conjuntos de entrenamiento y prueba
    print(
        f"4. Dividiendo los datos en {train_size * 100}% para entrenamiento/validación y {(1 - train_size) * 100}% para prueba final.")
    train_val_data, test_data = np.split(processed_data, [int(len(processed_data) * train_size)])

    # Dividir el primer bloque de nuevo en Entrenamiento y Validación
    val_split_index = int(len(train_val_data) * (1 - validation_size))
    train_data = train_val_data[:val_split_index]
    validation_data = train_val_data[val_split_index:]

    print(f"   -> Tamaño del conjunto de entrenamiento: {len(train_data)}")
    print(f"   -> Tamaño del conjunto de validación: {len(validation_data)}")
    print(f"   -> Tamaño del conjunto de prueba: {len(test_data)}")

    previous_day_prices_test = data[target_col].shift(1).loc[test_data.index] 

    # Separar características y objetivo para los tres conjuntos
    TARGET_NAME = 'target' 
    X_train, y_train = train_data.drop(columns=[TARGET_NAME]), train_data[TARGET_NAME]
    X_val, y_val = validation_data.drop(columns=[TARGET_NAME]), validation_data[TARGET_NAME]
    X_test, y_test = test_data.drop(columns=[TARGET_NAME]), test_data[TARGET_NAME]
    train_index = X_train.index

    # 5. Escalar los datos
    print("5. Escalando características y variable objetivo...")
    feature_scaler, target_scaler = model.preprocessor.get_scalers()

    # Ajustar con X_train y transformar los tres
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_val_scaled = feature_scaler.transform(X_val)
    X_test_scaled = feature_scaler.transform(X_test)

    # Ajustar con y_train y transformar los tres
    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_val_scaled = target_scaler.transform(y_val.values.reshape(-1, 1))
    y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))

    model.feature_scaler = feature_scaler
    model.target_scaler = target_scaler

    # 6. Crear secuencias
    print(f"6. Creando secuencias de datos con longitud {sequence_length}...")
    X_train_seq, y_train_seq = model.preprocessor.create_sequences(X_train_scaled, y_train_scaled)
    X_val_seq, y_val_seq = model.preprocessor.create_sequences(X_val_scaled, y_val_scaled)
    X_test_seq, y_test_seq = model.preprocessor.create_sequences(X_test_scaled, y_test_scaled)
    print(f"  -> Forma de secuencias de entrenamiento: X={X_train_seq.shape}, y={y_train_seq.shape}")
    print(f"  -> Forma de secuencias de validación: X={X_val_seq.shape}, y={y_val_seq.shape}")
    print(f"  -> Forma de secuencias de prueba: X={X_test_seq.shape}, y={y_test_seq.shape}")

    y_train_actual_for_residuals = y_train.values[sequence_length:]
    residual_dates_train = train_index[sequence_length:]

    # PASO DE DIAGNÓSTICO
    print("\n6.5. Verificando la integridad de los datos antes del entrenamiento...")
    if np.isnan(X_train_seq).any() or np.isinf(X_train_seq).any():
        raise ValueError("Se encontraron valores NaN o Inf en X_train_seq. El preprocesamiento falló.")
    if np.isnan(y_train_seq).any() or np.isinf(y_train_seq).any():
        raise ValueError("Se encontraron valores NaN o Inf en y_train_seq. El preprocesamiento falló.")
    print("   -> ¡Datos de entrenamiento verificados! No contienen NaN ni Inf.")

    # 7. Entrenar el modelo
    if optimize_params:
        print("\n7. Iniciando optimización de hiperparámetros (esto puede tardar)...")
        model.optimize_hyperparameters(
            X_train_seq, y_train_seq,
            X_val_seq=X_val_seq,  # Usar el conjunto de prueba real para el tuner
            y_val_seq=y_val_seq,
            max_trials=20,  # Aumentar trials
            search_epochs=15,  # Aumentar épocas de búsqueda
            final_epochs=epochs,  # Usar las épocas definidas para el entrenamiento final
            patience=patience,  # Añadir paciencia
        )
    else:
        print("\n7. Iniciando entrenamiento del modelo con parámetros fijos...")
        earlty_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1)
        model.fit(  # Este fit ahora usa X_test_seq como validation_data
            X_train_seq, y_train_seq,
            epochs=epochs, batch_size=batch_size,
            validation_data=(X_val_seq, y_val_seq),  # Usar el conjunto de prueba aquí también
            callbacks=[earlty_stopping],  # Añadir EarlyStopping
        )

    if hasattr(model, 'best_params_') and model.best_params_:
        print("\n--- Resultados de la Optimización ---")
        print(f"Mejores hiperparámetros encontrados: {model.best_params_}")
        print("------------------------------------")

    print("\nCalculando residuales del conjunto de entrenamiento...")
    # 1. Hacer predicciones sobre el conjunto de entrenamiento (secuenciado y escalado)
    y_train_pred_scaled_seq = model.predict(X_train_seq)

    # 2. Desescalar las predicciones
    # y_train_pred_scaled_seq ya tiene la forma (samples, 1) si la última capa Dense es (units=1)
    y_train_pred_unscaled = model.target_scaler.inverse_transform(y_train_pred_scaled_seq).flatten()

    # 3. Los valores reales y_train_actual_for_residuals ya están desescalados y tienen la forma correcta

    # 4. Calcular residuales
    residuals_train = y_train_actual_for_residuals - y_train_pred_unscaled

    print(f"  -> Residuales del entrenamiento calculados. Forma: {residuals_train.shape}")

    # 8. Evaluar el modelo en el conjunto de prueba
    print("\n8. Evaluando el modelo final en el conjunto de prueba...")
    y_test_actual_for_eval = y_test.values[sequence_length:]
    previous_prices_for_eval = previous_day_prices_test.values[sequence_length:]

    model.evaluate(X_test_seq, y_test_seq, previous_prices_for_eval, y_test_actual_for_eval, target_col)
    print(f"   -> Métricas finales del modelo: {model.metrics}")

    # Lags del acf y pacf
    nlags = 40

    acf_values, confint_acf = acf(residuals_train, nlags=nlags, alpha=0.05)

    pacf_values, confint_pacf = pacf(residuals_train, nlags=nlags, alpha=0.05)
    
    # 9. Guardar el modelo si se proporciona una ruta
    if save_model_path:
        print(f"\n9. Guardando modelo en: {save_model_path}")
        model.save_model(save_model_path)

    print("\n--- Pipeline de Entrenamiento LSTM Completado Exitosamente ---")

    residuals_train = np.array(residuals_train)
    return model, residuals_train, residual_dates_train, acf_values, pacf_values, confint_acf, confint_pacf

