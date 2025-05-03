from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import RandomSearch, BayesianOptimization, Hyperband
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    A transformer for selecting specific features from a dataset.

    FeatureSelector is a custom transformer that enables feature selection
    by specifying their indices. It implements the scikit-learn TransformerMixin
    and can be used in machine learning pipelines to preprocess data
    before feeding it into a model. The user has to provide a list of indices
    representing the features to select, or it will default to returning
    all features.

    Attributes:
        features_index (Optional[List[int]]): A list of feature indices to select.
            If None, all features will be returned.
    """

    def __init__(self, features_index=None):
        self.features_index = features_index

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.features_index is None:
            return X
        return X[:, self.features_index]


class SequenceGenerator:
    """
    Utility class to generate sequences for LSTM model training.

    This class transforms time series data into sequences suitable for training
    LSTM models by creating sliding windows of input-output pairs.

    Attributes:
        n_lags (int): Length of input sequences
        target_steps (int): Number of future steps to predict
    """

    def __init__(self, n_lags=10, target_steps=1):
        """
        Initialize the sequence generator.

        Args:
            n_lags (int): Length of the input sequences
            target_steps (int): Number of future steps to predict
        """
        self.n_lags = n_lags
        self.target_steps = target_steps

    def create_sequences(self, data, target_col=None):
        """
        Create sequence data from time series.

        Args:
            data (array-like): Input time series data
            target_col (int, optional): Index of target column. If None, uses last column.

        Returns:
            tuple: (X, y) where X contains input sequences and y contains target values
        """
        if isinstance(data, pd.DataFrame):
            if target_col is None:
                # Usa la última columna como objetivo
                target_col = data.columns[-1]

            # Convertir DataFrame to numpy array
            y_data = data[target_col].values
            if len(data.columns) > 1:
                X_data = data.drop(columns=[target_col]).values
            else:
                X_data = data.values
        else:
            X_data = data
            y_data = data[:, -1] if target_col is None else data[:, target_col]

        X, y = [], []

        for i in range(len(X_data) - self.n_lags - self.target_steps + 1):
            # Secuencia de entrada
            X.append(X_data[i:(i + self.n_lags)])

            # Valor objetivo
            if self.target_steps == 1:
                y.append(y_data[i + self.n_lags])
            else:
                y.append(y_data[(i + self.n_lags):(i + self.n_lags + self.target_steps)])

        return np.array(X), np.array(y)


class TimeSeriesLSTMModel:
    """
    A deep learning model for time series forecasting based on LSTM.

    This class implements an LSTM-based model configured for handling
    time series data. It includes methods for preparing data with feature
    engineering, optimizing hyperparameters, fitting the model, making predictions,
    and evaluating performance. It supports advanced functionality such as multi-step
    future prediction and allows customization of network architecture.

    Attributes:
        model (Sequential): The Keras LSTM model
        n_lags (int): The length of input sequences
        forecast_horizon (int): Number of steps to forecast
        feature_scaler (StandardScaler): Scaler for features
        target_scaler (StandardScaler): Scaler for target variable
        feature_names (Optional[list]): List of feature names used in training
        sequence_generator (SequenceGenerator): Utility for creating LSTM sequences
        metrics (dict): Performance metrics after evaluation
    """

    def __init__(self,
                 units=50,
                 layers=1,
                 dropout=0.2,
                 learning_rate=0.001,
                 n_lags=10,
                 forecast_horizon=1
                 ):
        """
        Initialize the LSTM model with configurable parameters.

        Args:
            units (int): Number of LSTM units per layer
            layers (int): Number of LSTM layers
            dropout (float): Dropout rate for regularization
            learning_rate (float): Learning rate for optimizer
            n_lags (int): Length of input sequences
            forecast_horizon (int): Number of future steps to predict
        """
        self.units = units
        self.layers = layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.n_lags = n_lags
        self.forecast_horizon = forecast_horizon

        # Otros atributos
        self.model = None
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.feature_names = None
        self.sequence_generator = SequenceGenerator(n_lags=n_lags,
                                                    target_steps=forecast_horizon)
        self.metrics = {}

        # Arquitectura del modelo
        self._build_model()

    def _build_model(self, n_features=1):
        """
        Build the LSTM model architecture.

        Args:
            n_features (int): Number of input features
        """
        model = Sequential()

        # Add LSTM layers
        for i in range(self.layers):
            return_sequences = i < self.layers - 1  # Retorna secuencias solo si no es la última capa

            if i == 0:
                model.add(LSTM(units=self.units,
                               return_sequences=return_sequences,
                               input_shape=(self.n_lags, n_features)))
            else:
                model.add(LSTM(units=self.units, return_sequences=return_sequences))

            # La regularización se utiliza para evitar el sobreajuste
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout))

        # Capa de salida
        if self.forecast_horizon == 1:
            model.add(Dense(1))
        else:
            model.add(Dense(self.forecast_horizon))

        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse')

        self.model = model
        return model

    def prepare_data(self, data, target_col='Close'):
        """
        Prepare time series data with feature engineering.

        Args:
            data (DataFrame): Input data containing the time series
            target_col (str): Name of the target column for prediction

        Returns:
            DataFrame: Processed DataFrame with features
        """
        from Backend.utils import feature_engineering, add_lags

        # Si los datos no estan ordenados, ordenarlos por índice
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.sort_index()

        # Crear características de retraso
        data_with_lags = add_lags(data, target_col=target_col, n_lags=self.n_lags)

        # Caracteristicas adicionales
        required_cols = ['Close']
        if all(col in data.columns for col in required_cols):
            try:
                return feature_engineering(data_with_lags)
            except Exception as e:
                print(f"Warning: Could not apply full feature engineering: {e}")
                print("Using only lag features instead.")
                return data_with_lags
        else:
            print(f"Warning: Missing required columns {required_cols}. Using only lag features.")
            return data_with_lags

    def preprocess_data(self, X, y=None, is_training=True):
        """
        Scale features and target data.

        Args:
            X (array-like): Feature data
            y (array-like, optional): Target data
            is_training (bool): Whether this is training data

        Returns:
            tuple: Scaled X and y data
        """
        # Scale features
        if is_training:
            X_scaled = self.feature_scaler.fit_transform(X)
        else:
            X_scaled = self.feature_scaler.transform(X)

        # Scale target if provided
        if y is not None:
            if is_training:
                y_reshaped = y.reshape(-1, 1) if len(y.shape) == 1 else y
                y_scaled = self.target_scaler.fit_transform(y_reshaped)
            else:
                y_reshaped = y.reshape(-1, 1) if len(y.shape) == 1 else y
                y_scaled = self.target_scaler.transform(y_reshaped)

            # Reshape back if it was 1D
            if len(y.shape) == 1:
                y_scaled = y_scaled.flatten()

            return X_scaled, y_scaled

        return X_scaled

    def fit(self, X_train, y_train, validation_data=None, epochs=100, batch_size=32, verbose=1):
        """
        Train the LSTM model.

        Args:
            X_train (array-like): Training features
            y_train (array-like): Training targets
            validation_data (tuple, optional): Validation data (X_val, y_val)
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            verbose (int): Verbosity level (0, 1, or 2)

        Returns:
            self: The fitted model instance
        """
        # Escalar los datos
        X_scaled, y_scaled = self.preprocess_data(X_train, y_train, is_training=True)

        X_seq, y_seq = self.sequence_generator.create_sequences(X_scaled)

        # Se tiene que reconstruir el modelo si no existe o si la forma de entrada ha cambiado
        if self.model is None or self.model.input_shape[2] != X_seq.shape[2]:
            self._build_model(n_features=X_seq.shape[2])

        # Validación de datos
        val_data = None
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_scaled, y_val_scaled = self.preprocess_data(X_val, y_val, is_training=False)
            X_val_seq, y_val_seq = self.sequence_generator.create_sequences(X_val_scaled)
            val_data = (X_val_seq, y_val_seq)

        # Callbacks para detener el entrenamiento temprano y guardar el mejor modelo
        callbacks = [
            EarlyStopping(monitor='val_loss' if val_data else 'loss',
                          patience=10, restore_best_weights=True),
            ModelCheckpoint('best_model.h5', save_best_only=True,
                            monitor='val_loss' if val_data else 'loss')
        ]

        # Entrenar el modelo
        history = self.model.fit(
            X_seq, y_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=verbose
        )

        return self

    def predict(self, X):
        """
        Make predictions using the trained model.

        Args:
            X (array-like): Input features for prediction

        Returns:
            array: Model predictions
        """
        # Scale the features
        X_scaled = self.preprocess_data(X, is_training=False)

        # Create sequences
        X_seq, _ = self.sequence_generator.create_sequences(X_scaled)

        # Hacer predicciones
        y_pred_scaled = self.model.predict(X_seq)

        # Transformación inversa de las predicciones
        if y_pred_scaled.ndim == 2:
            y_pred = self.target_scaler.inverse_transform(y_pred_scaled)
        else:
            original_shape = y_pred_scaled.shape
            y_pred_scaled_reshaped = y_pred_scaled.reshape(-1, 1)
            y_pred_reshaped = self.target_scaler.inverse_transform(y_pred_scaled_reshaped)
            y_pred = y_pred_reshaped.reshape(original_shape)

        return y_pred

    def optimize_hyperparameters(self, X_train, y_train, X_val=None, y_val=None, feature_names=None,
                                 strategy="adaptive"):
        """
        Optimize the hyperparameters for an LSTM model using Keras Tuner and train the final model 
        with the best-found parameters. This function includes preprocessing the training and 
        validation data, creating sequences for training, hyperparameter optimization via various 
        strategies, and final model training with the optimized hyperparameters.

        Parameters:
            X_train: numpy.ndarray
                Training feature dataset, expected to be 2-dimensional, which will be scaled and 
                converted into sequences for LSTM training.

            y_train: numpy.ndarray
                Training target dataset, expected to be a single-dimensional array or compatible with 
                the LSTM sequence generator.

            X_val: numpy.ndarray, optional, default=None
                Validation feature dataset used for hyperparameter tuning. Will be processed if provided; 
                otherwise, a validation split is used during the training phase.

            y_val: numpy.ndarray, optional, default=None
                Validation target dataset consistent with `X_val`. Ignored if `X_val` is not provided.

            feature_names: list of str, optional, default=None
                List of feature names corresponding to the columns in `X_train` and other datasets.

            strategy: str, optional, default="adaptive"
                The optimization strategy to use for hyperparameter tuning. Supports "random", 
                "bayesian", "hyperband", or "adaptive" selection based on dataset characteristics.

        Returns:
            Object
                Returns the object instance with the optimized hyperparameters applied and the 
                final model trained.
        """
        self.feature_names = feature_names

        # Preprocesar los datos
        X_scaled, y_scaled = self.preprocess_data(X_train, y_train, is_training=True)

        # Crear secuencias
        X_seq, y_seq = self.sequence_generator.create_sequences(X_scaled)

        # Preparar datos de validación si se proporcionan
        if X_val is not None and y_val is not None:
            X_val_scaled, y_val_scaled = self.preprocess_data(X_val, y_val, is_training=False)
            X_val_seq, y_val_seq = self.sequence_generator.create_sequences(X_val_scaled)
            validation_data = (X_val_seq, y_val_seq)
            validation_split = None
        else:
            validation_data = None
            validation_split = 0.2

        # Definir el modelo hiperparametrizado
        from kerastuner import HyperModel
        class MyHyperModel(HyperModel):
            def __init__(self, input_shape):
                self.input_shape = input_shape

            def build(self, hp):
                model = Sequential()

                # Determinar número de capas
                n_layers = hp.Int('n_layers', min_value=1, max_value=3, default=1)

                # Primera capa LSTM
                units = hp.Int('units', min_value=32, max_value=256, step=32, default=64)
                return_sequences = True if n_layers > 1 else False

                model.add(LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    input_shape=self.input_shape
                ))

                model.add(Dropout(
                    hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1, default=0.2)
                ))

                # Capas adicionales
                for i in range(1, n_layers):
                    return_sequences = i < n_layers - 1
                    model.add(LSTM(
                        units=hp.Int(f'units_{i + 1}', min_value=16, max_value=128, step=16, default=32),
                        return_sequences=return_sequences
                    ))
                    model.add(Dropout(
                        hp.Float(f'dropout_{i + 1}', min_value=0.0, max_value=0.5, step=0.1, default=0.2)
                    ))

                # Capa de salida
                model.add(Dense(1))

                # Compilación
                model.compile(
                    optimizer=Adam(
                        learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2,
                                               sampling='log', default=1e-3)
                    ),
                    loss='mean_squared_error'
                )

                return model

        # Crear el modelo hiperparametrizado
        hypermodel = MyHyperModel(input_shape=(X_seq.shape[1], X_seq.shape[2]))

        # Función para seleccionar la estrategia de optimización
        def get_tuner(strategy, hypermodel):
            if strategy == "random":
                return RandomSearch(
                    hypermodel=hypermodel,
                    objective='val_loss',
                    max_trials=20,
                    executions_per_trial=2,
                    directory='tuner_dir',
                    project_name='lstm_random_search'
                )
            elif strategy == "bayesian":
                return BayesianOptimization(
                    hypermodel=hypermodel,
                    objective='val_loss',
                    max_trials=20,
                    executions_per_trial=2,
                    directory='tuner_dir',
                    project_name='lstm_bayesian_opt'
                )
            elif strategy == "hyperband":
                return Hyperband(
                    hypermodel=hypermodel,
                    objective='val_loss',
                    max_epochs=50,
                    factor=3,
                    directory='tuner_dir',
                    project_name='lstm_hyperband'
                )
            elif strategy == "adaptive":
                # La lógica para selección adaptativa
                if X_seq.shape[0] > 10000:  # Para datasets grandes
                    return Hyperband(
                        hypermodel=hypermodel,
                        objective='val_loss',
                        max_epochs=50,
                        factor=3,
                        directory='tuner_dir',
                        project_name='lstm_adaptive'
                    )
                else:
                    return BayesianOptimization(
                        hypermodel=hypermodel,
                        objective='val_loss',
                        max_trials=20,
                        executions_per_trial=2,
                        directory='tuner_dir',
                        project_name='lstm_adaptive'
                    )

        # Obtener el tuner según la estrategia seleccionada
        tuner = get_tuner(strategy, hypermodel)

        # Ejecutar la búsqueda
        tuner.search(
            X_seq, y_seq,
            epochs=50,
            batch_size=32,
            callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
            validation_data=validation_data,
            validation_split=validation_split,
            verbose=1
        )

        # Obtener los mejores hiperparámetros
        best_hp = tuner.get_best_hyperparameters(1)[0] # Estan en el índice 0
        print("Mejores hiperparámetros encontrados:")
        print(f"- n_layers: {best_hp.get('n_layers')}")
        print(f"- units: {best_hp.get('units')}")
        print(f"- dropout: {best_hp.get('dropout')}")
        print(f"- learning_rate: {best_hp.get('learning_rate')}")

        # Actualizar los hiperparámetros del modelo
        self.layers = best_hp.get('n_layers')
        self.units = best_hp.get('units')
        self.dropout = best_hp.get('dropout')
        self.learning_rate = best_hp.get('learning_rate')

        # Construir y entrenar el modelo final con los mejores hiperparámetros
        self._build_model(n_features=X_seq.shape[2])

        # Reentrenar el modelo con todas las épocas
        self.model.fit(
            X_seq, y_seq,
            epochs=100,  # Más épocas para entrenamiento final
            batch_size=32,
            validation_split=0.2,
            callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
            verbose=1
        )

        return self

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model using performance metrics.

        Args:
            X_test (array-like): Test features
            y_test (array-like): Test targets

        Returns:
            dict: Dictionary with evaluation metrics
        """
        from Backend.utils import evaluate_regression

        # Scale the data
        X_test_scaled, y_test_scaled = self.preprocess_data(X_test, y_test, is_training=False)

        # Crear secuencias
        X_seq, y_seq = self.sequence_generator.create_sequences(X_test_scaled)

        # Hacer predicciones
        y_pred_scaled = self.model.predict(X_seq)

        # Transformación inversa de las predicciones
        if y_pred_scaled.ndim == 2:
            y_pred = self.target_scaler.inverse_transform(y_pred_scaled)
        else:
            original_shape = y_pred_scaled.shape
            y_pred_scaled_reshaped = y_pred_scaled.reshape(-1, 1)
            y_pred_reshaped = self.target_scaler.inverse_transform(y_pred_scaled_reshaped)
            y_pred = y_pred_reshaped.reshape(original_shape)

        if y_seq.ndim == 2:
            y_true = self.target_scaler.inverse_transform(y_seq)
        else:
            original_shape = y_seq.shape
            y_seq_reshaped = y_seq.reshape(-1, 1)
            y_true_reshaped = self.target_scaler.inverse_transform(y_seq_reshaped)
            y_true = y_true_reshaped.reshape(original_shape)

        # Calcula las métricas de evaluación
        self.metrics = evaluate_regression(y_true.flatten(), y_pred.flatten())
        return self.metrics

    def predict_future(self, last_sequence, forecast_horizon=None):
        """
        Recursive prediction of future values.

        Args:
            last_sequence (array-like): The most recent sequence of data
            forecast_horizon (int, optional): Number of steps to forecast

        Returns:
            array: Future predictions
        """
        if forecast_horizon is None:
            forecast_horizon = self.forecast_horizon

        # Extraer la última secuencia
        if isinstance(last_sequence, pd.DataFrame):
            last_sequence = last_sequence.values

        if len(last_sequence.shape) == 1:
            last_sequence = last_sequence.reshape(1, -1)

        # Se escala la última secuencia
        last_sequence_scaled = self.feature_scaler.transform(last_sequence)

        # Reescalar the last sequence
        input_seq = last_sequence_scaled[-self.n_lags:].reshape(1, self.n_lags, -1)

        # Predicción recursiva
        predictions = []
        current_input = input_seq.copy()

        for _ in range(forecast_horizon):
            next_pred = self.model.predict(current_input, verbose=0)[0]

            if isinstance(next_pred, np.ndarray) and next_pred.size > 1:
                # Si la predicción es un array, tomar el primer valor
                next_pred = next_pred[0]

            predictions.append(next_pred)

            # Actualizar la entrada para la siguiente predicción
            current_input = np.roll(current_input, -1, axis=1)
            current_input[0, -1, 0] = next_pred  # Assuming target is the first feature

        # Hay que deshacer el escalado de las predicciones
        predictions_array = np.array(predictions).reshape(-1, 1)
        predictions_rescaled = self.target_scaler.inverse_transform(predictions_array).flatten()

        return predictions_rescaled

    def plot_training_history(self, history):
        """
        Plots the training and validation loss over epochs.

        Args:
            history (History): The training history object returned by Keras fit method.

        Returns:
            None
        """

        if history is None:
            raise ValueError("Model has not been trained yet.")

        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def plot_result(self, y_true, y_pred, title="LSTM Model Predictions"):
        """
        Plots the true vs predicted values.

        Args:
            y_true (array-like): The true target variable.
            y_pred (array-like): The predicted values.
            title (str): Title of the plot.

        Returns:
            None
        """

        from Backend.utils.visualizations import plot_predictions

        plot_predictions(y_true, y_pred, title)

    def plot_forecast(self, historical_data, forecast_values, target_col="Close", title="LSTM Model Forecast"):
        """
        Plots the historical data and forecasted values.

        Args:
            historical_data (DataFrame): The historical data used for training.
            forecast_values (array-like): The predicted future values.
            target_col (str): The name of the target column in the dataset.
            title (str): Title of the plot.

        Returns:
            None
        """

        from Backend.utils.visualizations import plot_forecast

        plot_forecast(historical_data, forecast_values, target_col, title)

    def save_model(self, model_path="models/lstm_model"):
        """
        Saves the trained model to a file.

        Args:
            model_path (str): The path where the model will be saved. (default: "models/lstm_model.h5")

        Returns:
            None
        """

        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Se guarda el modelo en formato HDF5 (Keras)
        self.model.save(f"{model_path}.h5")

        # Se guardan los estados del modelo y los hiperparámetros
        import pickle
        with open(f"{model_path}_scalers.pkl", 'wb') as f:
            pickle.dump({
                'feature_scaler': self.feature_scaler,
                'target_scaler': self.target_scaler,
                'n_lags': self.n_lags,
                'forecast_horizon': self.forecast_horizon
            }, f)

        # Metadata del modelo
        metadata = {
            'units': self.units,
            'layers': self.layers,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'n_lags': self.n_lags,
            'forecast_horizon': self.forecast_horizon,
            'metrics': self.metrics,
            'timestamp': pd.Timestamp.now().isoformat()
        }

        with open(f"{model_path}_metadata.json", 'w') as f:
            json.dump({k: str(v) if not isinstance(v, (int, float)) else v
                       for k, v in metadata.items()}, f, indent=4)

        print(f"Model saved to {model_path}")


    @classmethod
    def load_model(cls, model_path):
        """
        Loads a trained model from a file.

        Args:
            model_path (str): The path where the model is saved.

        Returns:
            TimeSeriesLSTMModel: An instance of the class with the loaded model.
        """

        # Carga el modelo Keras
        model = load_model(f"{model_path}.h5")

        # Se carga el modelo y los hiperparámetros
        import pickle
        with open(f"{model_path}_scalers.pkl", 'rb') as f:
            saved_data = pickle.load(f)

        # Se carga la metadata
        with open(f"{model_path}_metadata.json", 'r') as f:
            metadata = json.load(f)

        # Crea una nueva instancia de la clase
        instance = cls(
            units=metadata['units'],
            layers=metadata['layers'],
            dropout=metadata['dropout'],
            learning_rate=metadata['learning_rate'],
            n_lags=metadata['n_lags'],
            forecast_horizon=metadata['forecast_horizon']
        )

        # Asigna los atributos de la instancia
        instance.model = model
        instance.feature_scaler = saved_data['feature_scaler']
        instance.target_scaler = saved_data['target_scaler']
        instance.n_lags = saved_data['n_lags']
        instance.forecast_horizon = saved_data['forecast_horizon']

        # Si la metadata contiene nombres de características, los asigna
        if 'metrics' in metadata:
            instance.metrics = metadata['metrics']

        return instance