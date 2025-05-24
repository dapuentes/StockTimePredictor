import numpy as np
import pandas as pd
import joblib
import os
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import kerastuner as kt
from pandas.tseries.offsets import BDay
import tensorflow as tf

from utils.preprocessing import LSTMPreprocessor


class TimeSeriesLSTMModel:
    """
    Un modelo de ML para series de tiempo basado en LSTM.
    Utiliza un preprocesador inyectado y maneja el ciclo de vida de un modelo Keras.
    """

    def __init__(self, preprocessor: LSTMPreprocessor, lstm_units=50, dropout_rate=0.2):
        self.preprocessor = preprocessor
        self.model = None  # El modelo se construirá después, cuando se conozca el input_shape
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.feature_scaler = None
        self.target_scaler = None
        self.history = None  # Para guardar el historial de entrenamiento
        self.metrics = None

    def build_model(self, input_shape):
        """
        Construye y compila el modelo LSTM.
        Parámetros:
        - input_shape: Tupla que indica la forma de los datos de entrada (sequence_length, n_features)
        """
        model = Sequential()
        model.add(LSTM(
            units=self.lstm_units,
            return_sequences=True,  # Verdadero porque podríamos apilar otra capa LSTM
        ))
        model.add(Dropout(self.dropout_rate))

        # Segunda capa LSTM
        model.add(LSTM(units=self.lstm_units, return_sequences=False))
        model.add(Dropout(self.dropout_rate))

        # Capa de salida
        model.add(Dense(units=25))
        model.add(Dense(units=1))

        optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        self.model = model
        print("Modelo LSTM construido y compilado exitosamente.")
        self.model.summary()

    def optimize_hyperparameters(self, X_train, y_train, X_val_seq, y_val_seq,
                                 max_trials=20,  # Aumentar un poco las pruebas
                                 search_epochs=15,  # Épocas para cada prueba del tuner
                                 final_epochs=50):  # Épocas para el modelo final con los mejores HPs
        """
        Optimiza los hiperparámetros del modelo LSTM utilizando KerasTuner.
        """

        input_shape = (X_train.shape[1], X_train.shape[2])

        def build_hypermodel(hp):
            # Define los rangos de búsqueda para los hiperparámetros
            lstm_units_1 = hp.Int('lstm_units_1', min_value=32, max_value=128, step=32)
            lstm_units_2 = hp.Int('lstm_units_2', min_value=32, max_value=128, step=32)
            dropout_rate_hp = hp.Float('dropout_rate', min_value=0.1, max_value=0.4, step=0.1)
            learning_rate_hp = hp.Choice('learning_rate', values=[0.001, 0.0005, 0.0001])

            # Construye el modelo con los hiperparámetros
            model = Sequential()
            model.add(LSTM(units=lstm_units_1, return_sequences=True, input_shape=input_shape))
            model.add(Dropout(dropout_rate_hp))
            model.add(LSTM(units=lstm_units_2, return_sequences=False))
            model.add(Dropout(dropout_rate_hp))
            model.add(Dense(units=25))
            model.add(Dense(units=1))

            optimizer = Adam(learning_rate=learning_rate_hp, clipnorm=1.0)  # Mantener clipnorm
            model.compile(optimizer=optimizer, loss='mean_squared_error')
            return model

        tuner = kt.RandomSearch(
            build_hypermodel,
            objective='val_loss',
            max_trials=max_trials,
            executions_per_trial=1,
            directory='keras_tuner_dir',
            project_name=f'lstm_tuning_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}'
        )

        print(f"Iniciando la búsqueda de hiperparámetros (max_trials={max_trials}, search_epochs={search_epochs})...")
        tuner.search(X_train, y_train, epochs=search_epochs, validation_data=(X_val_seq, y_val_seq))

        print("Búsqueda de hiperparámetros completada.")

        self.best_params_ = tuner.get_best_hyperparameters(num_trials=1)[0].values
        print(f"Mejores hiperparámetros encontrados por KerasTuner: {self.best_params_}")

        # Construir y entrenar el modelo final con los mejores HPs
        print(f"\nReconstruyendo y entrenando el modelo final con los mejores HPs por {final_epochs} épocas...")

        # Usar los HPs encontrados para configurar el modelo actual
        self.lstm_units = self.best_params_[
            'lstm_units_1']  # Asumiendo que quieres usar el primero para la primera capa
        # O podrías tener self.lstm_units_1, self.lstm_units_2 en la clase
        self.dropout_rate = self.best_params_['dropout_rate']
        # Es crucial que el optimizador se cree con la mejor learning_rate

        final_model = build_hypermodel(tuner.get_best_hyperparameters(num_trials=1)[0])  # Construye con los mejores HPs

        # Almacenamos el modelo final en la instancia
        self.model = final_model

        # Entrenamos este modelo final
        self.history = self.model.fit(
            X_train, y_train,
            epochs=final_epochs,
            batch_size=32,  # Puedes hacerlo un HP también si quieres
            validation_data=(X_val_seq, y_val_seq),
            verbose=1
        )
        print("Modelo final entrenado con los mejores hiperparámetros.")
        self.model.summary()

        return self.best_params_

    def fit(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1):
        """
        Entrena el modelo LSTM.
        Primero se asegura de que el modelo esté construido.
        """

        if len(X_train.shape) != 3:
            raise ValueError(
                f"X_train debe tener 3 dimensiones (samples, timesteps, features), recibido: {X_train.shape}")

        if len(y_train.shape) != 2:
            raise ValueError(f"y_train debe tener 2 dimensiones (samples, 1), recibido: {y_train.shape}")

        if self.model is None:
            # Construye el modelo si no se ha hecho explícitamente
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.build_model(input_shape)

        print("Iniciando entrenamiento del modelo LSTM...")
        self.history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        print("Entrenamiento completado.")
        return self.history

    def predict(self, X):
        """Realiza predicciones usando el modelo LSTM entrenado."""
        return self.model.predict(X)

    # El decorador compila el bucle en un grafo de alto rendimiento
    @tf.function
    def _run_mc_dropout(self, input_sequence, n_iter):
        """Función interna optimizada para ejecutar el bucle de MC Dropout."""
        # Replicar el tensor de entrada para procesarlo en un solo lote grande
        replicated_input = tf.tile(input_sequence, [n_iter, 1, 1])

        # Realizar todas las predicciones en una sola llamada al modelo
        predictions = self.model(replicated_input, training=True)
        return predictions

    def predict_with_uncertainty(self, input_sequence, n_iter=100):

        """
        Realiza predicciones múltiples usando Monte Carlo Dropout para estimar la incertidumbre.

        Parámetros:
        - input_sequence: La secuencia de entrada con la forma (1, sequence_length, n_features).
        - n_iter: El número de pasadas hacia adelante (predicciones) a realizar.

        Devuelve:
        - point_prediction (float): La media de las predicciones.
        - lower_bound (float): El percentil 2.5.
        - upper_bound (float): El percentil 97.5.
        """
        # Llama a la función compilada por @tf.function
        predictions = self._run_mc_dropout(input_sequence, tf.constant(n_iter))

        # Ahora calculamos los estadísticos con NumPy
        predictions_np = predictions.numpy().flatten()

        point_prediction = np.mean(predictions_np)
        lower_bound = np.percentile(predictions_np, 2.5)
        upper_bound = np.percentile(predictions_np, 97.5)

        return point_prediction, lower_bound, upper_bound

    def predict_future(self, historical_data_df, forecast_horizon, target_col='Close', n_iter_mc=100):
        print("\n--- Iniciando pronóstico futuro recursivo con LSTM (Versión Robusta) ---")

        if self.model is None: raise ValueError("El modelo no ha sido entrenado.")
        if self.feature_scaler is None or self.target_scaler is None: raise ValueError(
            "Los escaladores no están disponibles.")

        sequence_length = self.preprocessor.sequence_length
        recursive_data_df = historical_data_df.copy()

        final_predictions = []
        final_lower_bounds = []
        final_upper_bounds = []

        # [MEJORA 3] Calcular volatilidad histórica para un rango dinámico
        historical_volatility = recursive_data_df[target_col].pct_change().std()

        for i in range(forecast_horizon):
            print(f"\n--- Pronosticando paso {i + 1}/{forecast_horizon} ---")

            # 1. Preparar y escalar datos
            processed_df = self.preprocessor.prepare_data(recursive_data_df, target_col)
            features_df = processed_df.drop(columns=[target_col])
            scaled_features = self.feature_scaler.transform(features_df)

            # [MEJORA 5] Validación de la longitud de la secuencia
            if len(scaled_features) < sequence_length:
                raise ValueError(
                    f"No hay suficientes datos ({len(scaled_features)}) para crear la secuencia de entrada (se necesitan {sequence_length}).")

            # 2. Obtener secuencia y predecir con incertidumbre
            last_sequence = scaled_features[-sequence_length:]
            input_sequence = np.reshape(last_sequence, (1, sequence_length, features_df.shape[1]))
            point_pred_scaled, lower_scaled, upper_scaled = self.predict_with_uncertainty(input_sequence,
                                                                                          n_iter=n_iter_mc)

            # [MEJORA 1] Desescalado robusto e individual
            point_pred_unscaled = self.target_scaler.inverse_transform([[point_pred_scaled]])[0, 0]
            lower_unscaled = self.target_scaler.inverse_transform([[lower_scaled]])[0, 0]
            upper_unscaled = self.target_scaler.inverse_transform([[upper_scaled]])[0, 0]

            final_predictions.append(point_pred_unscaled)
            final_lower_bounds.append(lower_unscaled)
            final_upper_bounds.append(upper_unscaled)

            print(f"Predicción: {point_pred_unscaled:.4f} (Intervalo: [{lower_unscaled:.4f} - {upper_unscaled:.4f}])")

            # 3. Crear la nueva fila con datos más realistas
            last_real_row = recursive_data_df.iloc[-1]

            # [MEJORA 7] Usar BDay para el siguiente día hábil
            next_index = last_real_row.name + BDay(1)

            # [MEJORA 3] Rango dinámico basado en volatilidad
            daily_range = max(0.005, historical_volatility * 0.5)  # Asegura un rango mínimo del 0.5%

            new_row_dict = {
                'Open': last_real_row[target_col],
                'High': point_pred_unscaled * (1 + daily_range),
                'Low': point_pred_unscaled * (1 - daily_range),
                'Close': point_pred_unscaled,
            }

            # [MEJORA 4] Propagación inteligente de otras columnas
            for col in recursive_data_df.columns:
                if col not in new_row_dict:
                    if 'Volume' in col:
                        # Usar promedio móvil simple de los últimos 5 periodos para el volumen
                        new_row_dict[col] = recursive_data_df[col].iloc[-5:].mean()
                    else:
                        # Para otras columnas, propagar el último valor conocido
                        new_row_dict[col] = last_real_row[col]

            new_row_df = pd.DataFrame(new_row_dict, index=[next_index])

            # 4. Añadir la nueva fila para la siguiente iteración
            recursive_data_df = pd.concat([recursive_data_df, new_row_df])

        print("\n--- Pronóstico futuro completado ---")
        return np.array(final_predictions), np.array(final_lower_bounds), np.array(final_upper_bounds)

    def evaluate(self, X_test, y_test):
        """Evalúa el modelo usando métricas de rendimiento."""
        from utils.evaluation import evaluate_regression

        y_pred_scaled = self.predict(X_test)

        # Desescalar tanto 'y_test' como 'y_pred' para la evaluación
        y_test_orig = self.target_scaler.inverse_transform(y_test)
        y_pred_orig = self.target_scaler.inverse_transform(y_pred_scaled)

        self.metrics = evaluate_regression(y_test_orig.flatten(), y_pred_orig.flatten())
        return self.metrics

    def save_model(self, dir_path):
        """
        Guarda el modelo LSTM y los componentes asociados (preprocesador y scalers).
        """
        os.makedirs(dir_path, exist_ok=True)

        # Keras recomienda el formato .keras
        self.model.save(os.path.join(dir_path, 'lstm_model.keras'))

        # Guardar los otros componentes con joblib
        components = {
            'preprocessor': self.preprocessor,
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler
        }
        joblib.dump(components, os.path.join(dir_path, 'lstm_components.joblib'))
        print(f"Modelo y componentes guardados en el directorio: {dir_path}")

    @classmethod
    def load_model(cls, dir_path):
        """
        Carga un modelo LSTM y sus componentes.
        """
        # Cargar el modelo Keras
        keras_model = load_model(os.path.join(dir_path, 'lstm_model.keras'))

        # Cargar los componentes
        components = joblib.load(os.path.join(dir_path, 'lstm_components.joblib'))

        # Crear una nueva instancia de la clase y poblarla
        instance = cls(preprocessor=components['preprocessor'])
        instance.model = keras_model
        instance.feature_scaler = components['feature_scaler']
        instance.target_scaler = components['target_scaler']

        print(f"Modelo y componentes cargados desde el directorio: {dir_path}")
        return instance