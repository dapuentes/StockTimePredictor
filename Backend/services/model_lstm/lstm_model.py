import numpy as np
import pandas as pd
import joblib
import os
from io import BytesIO # Para manejo de bytes en memoria
import tempfile # Para directorios temporales
import shutil # Para operaciones de archivos y directorios
import json
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
import kerastuner as kt
from pandas.tseries.offsets import BDay
import tensorflow as tf
from typing import Optional

from utils.preprocessing import LSTMPreprocessor

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    print("Advertencia: La librería google-cloud-storage no está instalada. Guardado/Carga en GCS no funcionará.")



class TimeSeriesLSTMModel:
    """A machine learning model for time series prediction using LSTM architecture.

    This class encapsulates an LSTM-based time series model, providing functionality for building, training, optimizing,
    and making predictions with uncertainty. It allows hyperparameter tuning via KerasTuner, includes dropout layers
    for regularization, and supports Monte Carlo Dropout for uncertainty estimation.

    Attributes:
        preprocessor (LSTMPreprocessor): Preprocessor instance for data preparation.
        model (Sequential | None): The built LSTM model. None until the model is built.
        lstm_units (int): Number of units for each LSTM layer.
        dropout_rate (float): Dropout rate for regularization.
        feature_scaler (Any): Scaler for input features, initialized as None.
        target_scaler (Any): Scaler for target variable, initialized as None.
        history (Any | None): Training history of the model, stored after training.
        metrics (Any | None): Metrics of the model, initialized as None.
    """

    def __init__(self, preprocessor: LSTMPreprocessor, lstm_units=50, dropout_rate=0.2, bucket_name: str = None):
        self.preprocessor = preprocessor
        self.model: Optional[Sequential] = None # El modelo Keras
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.feature_scaler = None
        self.target_scaler = None
        self.history = None  # Para guardar el historial de entrenamiento
        self.metrics = None
        self.best_params_ = None
        self.bucket_name = None
        self._KERAS_MODEL_SUBDIR = "keras_model_files"
        self._COMPONENTS_FILENAME = "lstm_components.joblib"
        self._METADATA_FILENAME = "lstm_metadata.json"

    def build_model(self, input_shape):
        """
        Builds and compiles an LSTM-based model for time series or sequential data tasks.

        This function constructs a sequential model using two LSTM layers, batch normalization,
        dropout for regularization, and dense layers for output. The model is compiled with
        mean squared error as the loss function and the Adam optimizer with gradient clipping.

        Args:
            input_shape: tuple
                Shape of the input data. It generally includes the number of time steps and
                the number of features per step.
        """
        model = Sequential()
        model.add(LSTM(
            units=self.lstm_units,
            return_sequences=True,  # Verdadero porque podríamos apilar otra capa LSTM
            kernel_regularizer=regularizers.l2(0.001),  # Regularización L2 para evitar el sobreajuste
        ))
        model.add(BatchNormalization())  # Normalización por lotes para estabilizar el aprendizaje
        model.add(Dropout(self.dropout_rate))

        # Segunda capa LSTM
        model.add(LSTM(units=self.lstm_units, return_sequences=False))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))

        # Capa de salida
        model.add(Dense(units=25, activation='relu'))  # relu para la capa oculta
        model.add(Dense(units=1))

        optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        self.model = model
        print("Modelo LSTM construido y compilado exitosamente.")
        self.model.summary()

    def optimize_hyperparameters(self, X_train, y_train, X_val_seq, y_val_seq,
                                 max_trials=20,  # Aumentar un poco las pruebas
                                 search_epochs=15,  # Épocas para cada prueba del tuner
                                 final_epochs=50,  # Épocas para el modelo final con los mejores HPs
                                 patience=10):  # Paciencia para EarlyStopping
        """
        Optimizes hyperparameters for an LSTM-based neural network model, performs hyperparameter tuning using
        Keras Tuner, and trains a final model with the best identified hyperparameters. The optimization leverages
        random search to explore a defined range of potential hyperparameter values, and early stopping is used
        to prevent overfitting and enhance computational efficiency.

        The function internally defines a model-building function for Keras Tuner, specifies the hyperparameter
        search range, and conducts a search process across multiple trials. After identifying the best hyperparameters,
        a final LSTM model is built and trained using these optimal values.

        During the hyperparameter tuning and final model training processes, patience counts are applied to monitor
        performance on the validation dataset and decide when to stop training early.

        Args:
            X_train: Training input data in the shape (samples, timesteps, features) used to train and find optimal
                hyperparameters.
            y_train: Target values corresponding to X_train for supervised learning.
            X_val_seq: Validation input data in the shape (samples, timesteps, features) used to validate performance
                during the tuning and final training processes.
            y_val_seq: Target values corresponding to X_val_seq for validation monitoring.
            max_trials: Specifies the maximum number of hyperparameter tuning trials to execute during the search
                process.
            search_epochs: Number of epochs to train each model during hyperparameter tuning trials.
            final_epochs: Number of epochs to train the final model built with identified best hyperparameters.
            patience: Number of epochs to wait for improvement in validation loss before applying early stopping.
                This value is used to configure early stopping in both hyperparameter tuning and final model training.

        Returns:
            A dictionary containing the best hyperparameters as identified during the hyperparameter tuning process.
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
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate_hp))

            model.add(LSTM(units=lstm_units_2, return_sequences=False))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate_hp))

            model.add(Dense(units=25, activation='relu'))
            model.add(Dense(units=1))

            optimizer = Adam(learning_rate=learning_rate_hp, clipnorm=1.0)  # Mantener clipnorm
            model.compile(optimizer=optimizer, loss='mean_squared_error')
            return model
        
        tuner_project_name = f'lstm_tuning_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs('keras_tuner_dir', exist_ok=True)

        tuner = kt.RandomSearch(
            build_hypermodel,
            objective='val_loss',
            max_trials=max_trials,
            executions_per_trial=1,
            directory='keras_tuner_dir',
            project_name=f'lstm_tuning_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}'
        )

        print(f"Iniciando la búsqueda de hiperparámetros (max_trials={max_trials}, search_epochs={search_epochs})...")

        # Usar EarlyStopping para evitar el sobreajuste
        search_early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience // 2,
            verbose=1
        )

        tuner.search(X_train, y_train, epochs=search_epochs, validation_data=(X_val_seq, y_val_seq),
                     callbacks=[search_early_stopping])

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

        final_early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )

        # Entrenamos este modelo final
        self.history = self.model.fit(
            X_train, y_train,
            epochs=final_epochs,
            batch_size=32,
            validation_data=(X_val_seq, y_val_seq),
            verbose=1,
            callbacks=[final_early_stopping]
        )
        print("Modelo final entrenado con los mejores hiperparámetros.")
        self.model.summary()

        return self.best_params_

    def fit(self, X_train, y_train, epochs=50, batch_size=32, validation_data=None, callbacks=None):
        """
        Trains the LSTM model using the provided training data, training parameters,
        and optionally, validation data and callbacks.

        Args:
            X_train: A 3-dimensional array representing the training input features
                with shape (samples, timesteps, features).
            y_train: A 2-dimensional array representing the corresponding target
                outputs with shape (samples, 1).
            epochs: An integer representing the number of training iterations
                over the entire dataset. Default is 50.
            batch_size: An integer specifying the number of samples processed before
                updating the model. Default is 32.
            validation_data: Optional. A tuple (X_val, y_val) representing the data
                used for validation during training. Default is None.
            callbacks: Optional. A list of callback instances that are invoked
                during training at specific points. Default is None.

        Returns:
            A `History` object that contains details of the training process, such
            as loss and accuracy values for each epoch, as logged during fitting.

        Raises:
            ValueError: If `X_train` does not have exactly 3 dimensions
                (samples, timesteps, features).
            ValueError: If `y_train` does not have exactly 2 dimensions
                (samples, 1).
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
            validation_split=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        print("Entrenamiento completado.")
        return self.history

    def predict(self, X):
        """
        Uses the model's prediction method to make predictions on the given input.

        Args:
            X: Input data for which predictions are to be made. The type and format
                of X should match the requirements of the underlying model's
                predict method.

        Returns:
            The output of the underlying model's predict method, which contains
            the predictions for the input data.
        """
        return self.model.predict(X)

    # El decorador compila el bucle en un grafo de alto rendimiento
    @tf.function
    def _run_mc_dropout(self, input_sequence, n_iter):
        """
        Applies Monte Carlo (MC) Dropout by running multiple forward passes through the model
        in training mode using the same input sequence.

        Args:
            input_sequence: A 3D tensor containing the input data. The shape is usually
                (batch_size, timesteps, features), where batch_size is the number of
                input sequences, timesteps is the length of the sequence, and features
                is the number of features per timestep.
            n_iter: An integer representing the number of stochastic forward passes to
                perform. This determines how many times the input_sequence is replicated,
                and thus how many MC Dropout predictions will be generated.

        Returns:
            A tensor containing the predictions from the model with MC Dropout applied.
            The shape will typically be (batch_size * n_iter, ...), where the rest of the
            shape depends on the model's output dimensions.
        """
        # Replicar el tensor de entrada para procesarlo en un solo lote grande
        replicated_input = tf.tile(input_sequence, [n_iter, 1, 1])

        # Realizar todas las predicciones en una sola llamada al modelo
        predictions = self.model(replicated_input, training=True)
        return predictions

    def predict_with_uncertainty(self, input_sequence, n_iter=100):

        """
        Predicts the output with uncertainty estimation using Monte Carlo Dropout. The function
        performs n_iter stochastic passes (forward passes with dropout activated) through the
        trained neural network for the given input sequence. The uncertainty is calculated based
        on the distribution of predictions obtained.

        Args:
            input_sequence: Input sequence for the trained model. Format and dimensions are
                dependent on the model implementation.
            n_iter: Number of Monte Carlo iterations for performing stochastic forward passes.

        Returns:
            Tuple containing:
                point_prediction: The mean of the predictions obtained from n_iter forward passes,
                    representing the most likely value.
                lower_bound: The lower bound of the confidence interval at 2.5 percentile.
                upper_bound: The upper bound of the confidence interval at 97.5 percentile.
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
        """
        Predicts future values for a specified target column over a defined forecast horizon using
        a trained model. This method utilizes a recursive forecasting approach with log-return
        predictions and reconstructs the forecasted price for each future timestep. Uncertainty
        bounds for the predictions are also calculated using Monte Carlo simulations.

        This method assumes that the model has been properly trained, along with corresponding
        feature and target scalers, and that the preprocessor contains the necessary feature
        configuration.

        Args:
            historical_data_df (pd.DataFrame): The historical data containing the features and
                target column used for initializing and continuing the forecasting process.
            forecast_horizon (int): The number of future timesteps to forecast iteratively.
            target_col (str, optional): The column name in `historical_data_df` representing the
                target variable. Defaults to 'Close'.
            n_iter_mc (int, optional): The number of Monte Carlo iterations for estimating
                uncertainty in predictions. Defaults to 100.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing three numpy arrays:
                - The predicted prices for the forecast horizon.
                - The corresponding lower bounds of uncertainty for each forecast.
                - The corresponding upper bounds of uncertainty for each forecast.

        Raises:
            ValueError: If the model has not been trained or necessary scalers/preprocessor
                components are missing.
            ValueError: If insufficient data is available to form the input sequence for
                forecasting.
        """
        print(f"\n--- Iniciando pronóstico futuro para '{target_col}' basado en retornos logarítmicos ---")

        if self.model is None: raise ValueError("El modelo no ha sido entrenado.")
        if self.feature_scaler is None or self.target_scaler is None: raise ValueError(
            "Los escaladores no están disponibles.")
        if not hasattr(self.preprocessor, 'feature_names') or not self.preprocessor.feature_names:
            raise ValueError("Los `feature_names` no están disponibles en el preprocesador. Re-entrena el modelo.")

        sequence_length = self.preprocessor.sequence_length
        recursive_data_df = historical_data_df.copy()

        final_predictions = []
        final_lower_bounds = []
        final_upper_bounds = []

        # Calcular volatilidad histórica para un rango dinámico
        historical_volatility = recursive_data_df[target_col].pct_change().std()

        for i in range(forecast_horizon):
            print(f"\n--- Pronosticando paso {i + 1}/{forecast_horizon} ---")

            # 1. Preparar datos usando el preprocesador.
            #    OJO: `prepare_data` ahora crea una columna 'target' (log_returns) y elimina 'Close'.
            #    Le pasamos el `target_col` original ('Close') para que sepa sobre qué columna calcular los retornos.
            processed_df = self.preprocessor.prepare_data(recursive_data_df, target_col=target_col)

            # 2. Asegurar el orden de las columnas y escalar
            features_df = processed_df[self.preprocessor.feature_names]
            scaled_features = self.feature_scaler.transform(features_df)

            # 3. Obtener la última secuencia para la predicción
            if len(scaled_features) < sequence_length:
                raise ValueError(
                    f"No hay suficientes datos ({len(scaled_features)}) para crear la secuencia de entrada (se necesitan {sequence_length}).")

            last_sequence = scaled_features[-sequence_length:]
            input_sequence = np.reshape(last_sequence, (1, sequence_length, features_df.shape[1]))

            # 4. Predecir el RETORNO LOGARÍTMICO ESCALADO con incertidumbre
            scaled_log_return_pred, lower_scaled, upper_scaled = self.predict_with_uncertainty(input_sequence,
                                                                                               n_iter=n_iter_mc)

            # 5. Desescalar la predicción para obtener el RETORNO LOGARÍTMICO REAL
            unscaled_log_return_pred = self.target_scaler.inverse_transform([[scaled_log_return_pred]])[0, 0]
            lower_log_return_unscaled = self.target_scaler.inverse_transform([[lower_scaled]])[0, 0]
            upper_log_return_unscaled = self.target_scaler.inverse_transform([[upper_scaled]])[0, 0]

            # 6. Reconstruir el PRECIO
            last_real_price = recursive_data_df[target_col].iloc[-1]

            # Fórmula de reconstrucción: Precio_t = Precio_{t-1} * exp(retorno_log_t)
            next_price_pred = last_real_price * np.exp(unscaled_log_return_pred)
            final_predictions.append(next_price_pred)

            lower_bound_pred = last_real_price * np.exp(lower_log_return_unscaled)
            final_lower_bounds.append(lower_bound_pred)

            upper_bound_pred = last_real_price * np.exp(upper_log_return_unscaled)
            final_upper_bounds.append(upper_bound_pred)

            print(
                f"Predicción (Precio): {next_price_pred:.4f} (Intervalo: [{lower_bound_pred:.4f} - {upper_bound_pred:.4f}])")

            # 3. Crear la nueva fila con datos más realistas
            last_real_row = recursive_data_df.iloc[-1]

            # [MEJORA 7] Usar BDay para el siguiente día hábil
            next_index = last_real_row.name + BDay(1)

            # [MEJORA 3] Rango dinámico basado en volatilidad
            daily_range = max(0.005, historical_volatility * 0.5)  # Asegura un rango mínimo del 0.5%

            new_row_dict = {
                'Open': last_real_row[target_col],
                'High': next_price_pred * (1 + daily_range),
                'Low': next_price_pred * (1 - daily_range),
                'Close': next_price_pred,
            }

            # Propagación inteligente de otras columnas
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

    def evaluate(self, X_test_seq, y_test_seq, previous_prices_for_eval, y_test_actual_for_eval, price_col_name):
        """
        Evaluates the performance of the LSTM regression model on test data by predicting log returns,
        rescaling them to the original scale, reconstructing predicted and true prices, and computing
        evaluation metrics.
        
            - X_test_seq (np.ndarray): Test dataset features for sequence prediction.
            - y_test_seq (np.ndarray): True target values (scaled) for the test features.
            previous_prices_for_eval (np.ndarray): Array of previous prices used to reconstruct predicted and true prices.
            - y_test_actual_for_eval (np.ndarray): Actual log returns (unscaled) for evaluation.
            - price_col_name (str): Name of the price column (for reference or logging).
            - dict: Dictionary containing evaluation metrics (e.g., MAE, RMSE, R2) for the regression model based on reconstructed prices.
        Raises:
            ValueError: If the length of 'previous_prices_for_eval' and 'y_pred_log_returns' do not match.
        """
        from utils.evaluation import evaluate_regression

        print(" -> Realizando predicciones sobre el conjunto de prueba...")

        # Predecir los retornos logarítmicos escalados
        y_pred_scaled = self.predict(X_test_seq)
        
        # Desescalar las predicciones para obtener los retornos logarítmicos
        y_pred_log_returns = self.target_scaler.inverse_transform(y_pred_scaled).flatten()
        
        # Los `y_test_actual_for_eval` ya son los retornos logarítmicos reales sin escalar
        y_true_log_returns = y_test_actual_for_eval
        
        # Reconstruir los precios predichos y los precios reales
        if len(previous_prices_for_eval) != len(y_pred_log_returns):
            raise ValueError("La longitud de 'previous_prices_for_eval' y 'y_pred_log_returns' no coincide.")

        y_pred_price = previous_prices_for_eval * np.exp(y_pred_log_returns)
        y_true_price = previous_prices_for_eval * np.exp(y_true_log_returns)
        
        print(" -> Calculando métricas sobre los precios reconstruidos...")
        # Calcular métricas sobre los precios, que son mucho más interpretables
        self.metrics = evaluate_regression(y_true_price, y_pred_price)
        return self.metrics

    def _get_metadata(self, training_end_date=None):
        """Helper para generar el diccionario de metadatos."""
        serializable_best_params = {}
        if self.best_params_:
            for k, v in self.best_params_.items():
                if isinstance(v, (np.integer, np.int_)): serializable_best_params[k] = int(v)
                elif isinstance(v, (np.floating, np.float_)): serializable_best_params[k] = float(v)
                elif isinstance(v, np.ndarray): serializable_best_params[k] = v.tolist()
                else: serializable_best_params[k] = v
        
        return {
            'model_type': 'TimeSeriesLSTMModel',
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'sequence_length': self.preprocessor.sequence_length if self.preprocessor else None,
            'n_lags_preprocessor': self.preprocessor.n_lags if self.preprocessor else None,
            'best_params_tuner': serializable_best_params,
            'metrics': self.metrics,
            'timestamp': pd.Timestamp.now().isoformat(),
            'training_end_date': training_end_date,
            'feature_names_preprocessor': self.preprocessor.feature_names if self.preprocessor else None
        }

    def save_model(self, dir_path: str, training_end_date: Optional[str] = None, bucket_name_override: Optional[str] = None):
        """
        Guarda el modelo LSTM (Keras) y sus componentes (preprocesador, scalers)
        en GCS o localmente. `dir_path` es la ruta base para el "directorio" del modelo.
        """
        if self.model is None:
            raise ValueError("No hay modelo Keras para guardar.")
        if self.preprocessor is None or self.feature_scaler is None or self.target_scaler is None:
            print("Advertencia: Preprocesador o scalers no están definidos. Se guardará solo el modelo Keras.")

        active_bucket_name = bucket_name_override if bucket_name_override is not None else self.bucket_name
        
        components_to_save = {
            'preprocessor_config': { # Guardar config del preprocesador en lugar del objeto entero si es complejo
                'class': self.preprocessor.__class__.__name__,
                'sequence_length': self.preprocessor.sequence_length,
                'n_lags': self.preprocessor.n_lags,
                'feature_names': self.preprocessor.feature_names
            },
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler
        }

        with tempfile.TemporaryDirectory() as temp_dir_local:
            # 1. Guardar modelo Keras en el directorio temporal local
            keras_model_local_path = os.path.join(temp_dir_local, self._KERAS_MODEL_SUBDIR)
            self.model.save(keras_model_local_path) # Guarda en formato SavedModel (directorio)
            print(f"Modelo Keras guardado temporalmente en: {keras_model_local_path}")

            # 2. Guardar componentes (preprocesador, scalers) en el directorio temporal local
            components_local_path = os.path.join(temp_dir_local, self._COMPONENTS_FILENAME)
            joblib.dump(components_to_save, components_local_path)
            print(f"Componentes guardados temporalmente en: {components_local_path}")
            
            # 3. Guardar metadatos en el directorio temporal local
            metadata = self._get_metadata(training_end_date)
            metadata_local_path = os.path.join(temp_dir_local, self._METADATA_FILENAME)
            with open(metadata_local_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            print(f"Metadatos guardados temporalmente en: {metadata_local_path}")


            # 4. Subir a GCS o copiar a destino local
            if active_bucket_name and GCS_AVAILABLE:
                try:
                    client = storage.Client()
                    bucket = client.bucket(active_bucket_name)
                    
                    print(f"Subiendo artefactos del modelo a GCS: gs://{active_bucket_name}/{dir_path}/")
                    # Subir cada archivo del directorio temporal al "directorio" GCS
                    for item_name in os.listdir(temp_dir_local):
                        item_local_path = os.path.join(temp_dir_local, item_name)
                        item_gcs_path = os.path.join(dir_path, item_name) # Ruta dentro del bucket

                        if os.path.isfile(item_local_path):
                            blob = bucket.blob(item_gcs_path)
                            blob.upload_from_filename(item_local_path)
                            print(f"  Subido: {item_name} a {item_gcs_path}")
                        elif os.path.isdir(item_local_path): # Para el directorio del modelo Keras
                            for root, _, files in os.walk(item_local_path):
                                for file_name in files:
                                    file_local_path = os.path.join(root, file_name)
                                    # Crear la ruta relativa dentro del directorio Keras
                                    relative_path = os.path.relpath(file_local_path, item_local_path)
                                    file_gcs_path = os.path.join(dir_path, item_name, relative_path) # item_name es _KERAS_MODEL_SUBDIR
                                    
                                    blob = bucket.blob(file_gcs_path)
                                    blob.upload_from_filename(file_local_path)
                                    print(f"  Subido (Keras): {file_name} a {file_gcs_path}")
                    print(f"Modelo y componentes guardados en GCS: gs://{active_bucket_name}/{dir_path}")
                except Exception as e:
                    print(f"Error al guardar en GCS: {e}. Intentando guardar localmente como fallback.")
                    self._save_locally_from_temp(temp_dir_local, dir_path) # dir_path es la ruta local ahora
            else:
                if active_bucket_name and not GCS_AVAILABLE:
                    print("Advertencia: bucket_name especificado pero google-cloud-storage no disponible. Guardando localmente.")
                self._save_locally_from_temp(temp_dir_local, dir_path)

    def _save_locally_from_temp(self, temp_dir_local: str, dest_dir_local: str):
        """Copia el contenido de temp_dir_local a dest_dir_local."""
        if os.path.exists(dest_dir_local):
            shutil.rmtree(dest_dir_local) # Eliminar destino si existe para evitar errores con copytree
        shutil.copytree(temp_dir_local, dest_dir_local)
        print(f"Modelo y componentes guardados localmente en: {dest_dir_local}")

    @classmethod
    def load_model(cls, dir_path: str, bucket_name: Optional[str] = None):
        """
        Carga el modelo LSTM (Keras) y sus componentes desde GCS o localmente.
        `dir_path` es la ruta base para el "directorio" del modelo.
        """
        instance = None # Para asegurar que se retorna algo o se lanza error

        with tempfile.TemporaryDirectory() as temp_dir_local:
            keras_model_local_path = os.path.join(temp_dir_local, cls._KERAS_MODEL_SUBDIR) # Usar cls para acceder a la constante
            components_local_path = os.path.join(temp_dir_local, cls._COMPONENTS_FILENAME)
            metadata_local_path = os.path.join(temp_dir_local, cls._METADATA_FILENAME) # Opcional cargar metadatos aquí

            if bucket_name and GCS_AVAILABLE:
                try:
                    client = storage.Client()
                    bucket = client.bucket(bucket_name)
                    print(f"Descargando artefactos del modelo desde GCS: gs://{bucket_name}/{dir_path}/ a {temp_dir_local}")

                    # Descargar directorio del modelo Keras
                    keras_model_gcs_prefix = os.path.join(dir_path, cls._KERAS_MODEL_SUBDIR) + "/" # Asegurar trailing slash
                    blobs_keras = bucket.list_blobs(prefix=keras_model_gcs_prefix)
                    os.makedirs(keras_model_local_path, exist_ok=True)
                    for blob in blobs_keras:
                        if not blob.name.endswith('/'): # No es un "directorio" placeholder
                            file_rel_path = os.path.relpath(blob.name, keras_model_gcs_prefix)
                            local_file_path = os.path.join(keras_model_local_path, file_rel_path)
                            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                            blob.download_to_filename(local_file_path)
                            print(f"  Descargado (Keras): {blob.name} a {local_file_path}")
                    
                    # Descargar componentes
                    components_gcs_path = os.path.join(dir_path, cls._COMPONENTS_FILENAME)
                    blob_components = bucket.blob(components_gcs_path)
                    if not blob_components.exists():
                        raise FileNotFoundError(f"Archivo de componentes no encontrado en GCS: gs://{bucket_name}/{components_gcs_path}")
                    blob_components.download_to_filename(components_local_path)
                    print(f"  Descargado (Componentes): {components_gcs_path} a {components_local_path}")

                except Exception as e:
                    print(f"Error al cargar desde GCS: {e}. Intentando cargar localmente como fallback.")
                    return cls._load_locally_to_instance(dir_path) # dir_path es la ruta local
            else:
                if bucket_name and not GCS_AVAILABLE:
                    print("Advertencia: bucket_name especificado pero google-cloud-storage no disponible. Cargando localmente.")
                return cls._load_locally_to_instance(dir_path) # dir_path es la ruta local

            # Cargar desde el directorio temporal local
            loaded_keras_model = load_model(keras_model_local_path)
            loaded_components = joblib.load(components_local_path)
            
      
            preprocessor_config = loaded_components['preprocessor_config']
            rehydrated_preprocessor = LSTMPreprocessor(
                sequence_length=preprocessor_config['sequence_length'],
                n_lags=preprocessor_config['n_lags']
            )
            rehydrated_preprocessor.feature_names = preprocessor_config['feature_names']

            instance = cls(preprocessor=rehydrated_preprocessor, bucket_name=bucket_name) # Pasar el bucket_name original
            instance.model = loaded_keras_model
            instance.feature_scaler = loaded_components['feature_scaler']
            instance.target_scaler = loaded_components['target_scaler']
    

            print(f"Modelo LSTM y componentes cargados (desde GCS o fallback local a través de temp).")
            return instance
            
    @classmethod
    def _load_locally_to_instance(cls, dir_path: str):
        """Carga el modelo y componentes localmente y devuelve una instancia."""
        keras_model_local_path = os.path.join(dir_path, cls._KERAS_MODEL_SUBDIR)
        components_local_path = os.path.join(dir_path, cls._COMPONENTS_FILENAME)

        if not os.path.exists(keras_model_local_path) or not os.path.exists(components_local_path):
            raise FileNotFoundError(f"Directorio del modelo Keras ({keras_model_local_path}) o archivo de componentes ({components_local_path}) no encontrado localmente.")

        loaded_keras_model = load_model(keras_model_local_path)
        loaded_components = joblib.load(components_local_path)

        preprocessor_config = loaded_components['preprocessor_config']
        rehydrated_preprocessor = LSTMPreprocessor(
            sequence_length=preprocessor_config['sequence_length'],
            n_lags=preprocessor_config['n_lags']
        )
        rehydrated_preprocessor.feature_names = preprocessor_config['feature_names']
        
        instance = cls(preprocessor=rehydrated_preprocessor) # No se pasa bucket_name aquí
        instance.model = loaded_keras_model
        instance.feature_scaler = loaded_components['feature_scaler']
        instance.target_scaler = loaded_components['target_scaler']
        
        print(f"Modelo LSTM y componentes cargados localmente desde: {dir_path}")
        return instance