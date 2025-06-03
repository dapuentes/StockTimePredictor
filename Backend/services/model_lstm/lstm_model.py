import numpy as np
import pandas as pd
import joblib
import os
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
import kerastuner as kt
from pandas.tseries.offsets import BDay
import tensorflow as tf

from utils.preprocessing import LSTMPreprocessor


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

    def save_model(self, dir_path):
        """
        Saves the model and its components to the specified directory.

        This method saves the trained machine learning model using Keras and its
        associated components, including preprocessor, feature scaler, and target
        scaler, to the specified directory. The model is saved in the `.keras` format,
        and the components are stored using the `joblib` library. If the directory does
        not already exist, it is created.

        Args:
            dir_path: The path to the directory where the model and components will be
                saved, as a string.
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
        
        # Guardar metadatos del modelo
        metadata = {
            'best_params': self.best_params_ if hasattr(self, 'best_params_') else None,
            'metrics': self.metrics if hasattr(self, 'metrics') else None,
            'model_type': 'LSTM',
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Guardar metadatos en archivo JSON
        metadata_file = os.path.join(dir_path, 'lstm_metadata.json')
        import json
        with open(metadata_file, 'w') as f:
            # Convertir valores numpy a tipos nativos de Python para JSON serialization
            metadata_serializable = {}
            for k, v in metadata.items():
                if isinstance(v, dict):
                    metadata_serializable[k] = {sub_k: float(sub_v) if isinstance(sub_v, np.float64) else sub_v 
                                               for sub_k, sub_v in v.items()}
                elif isinstance(v, np.ndarray):
                    metadata_serializable[k] = v.tolist()
                elif isinstance(v, np.float64):
                    metadata_serializable[k] = float(v)
                else:
                    metadata_serializable[k] = v
            json.dump(metadata_serializable, f, indent=4)
        
        print(f"Modelo, componentes y metadatos guardados en el directorio: {dir_path}")

    @classmethod
    def load_model(cls, dir_path):
        """
        Loads a pre-trained Keras LSTM model and its associated components from the given
        directory path, recreates an instance of the class, and populates it with the loaded
        data. This method facilitates restoring a previously saved model setup for inference
        or further usage.

        Args:
            dir_path (str): The directory path from which the model and its components
                should be loaded.

        Returns:
            cls: An instance of the class populated with the loaded model, preprocessor,
                and scalers.
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