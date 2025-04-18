from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    FeatureSelector is a custom transformer for feature selection in a scikit-learn pipeline.

    This class is designed to select specific features from a dataset based on provided indices.
    It is suitable for use in preprocessing pipelines where automated feature selection might not
    be required, and specific features are predetermined for model training. The class implements
    the scikit-learn TransformerMixin and BaseEstimator, ensuring compatibility with scikit-learn
    pipelines and estimators. The user can specify the indices of the features they want to retain
    or process, or use all features in the absence of specified indices.

    Attributes:
        features_index (list of int or None): Indices of features to be selected. If None, all
        features are retained.

    Methods Summary:
        fit: Compatible with scikit-learn fit method, does not modify the data.
        transform: Selects the specified features based on the indices provided during initialization.
    """

    def __init__(self, features_index=None):
        self.features_index = features_index

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.features_index is None:
            return X
        return X[:, self.features_index]


class TimeSeriesLSTMModel:
    """
    A machine learning model for time series forecasting based on LSTM.

    This class implements an LSTM-based model configured for handling
    time series data. It includes methods for preparing data with feature
    engineering, creating sequences for LSTM input, optimizing hyperparameters,
    fitting the model, making predictions, and evaluating performance.
    It supports advanced functionality such as recursive future prediction
    and allows customization of feature and sequence configurations.

    Attributes:
        model (Sequential): The Keras Sequential LSTM model.
        best_params_ (Optional[dict]): The best hyperparameters selected during
            optimization.
        n_lags (int): Number of lag features to create for time series forecasting.
        time_steps (int): Number of time steps for sequence creation.
        feature_scaler (Optional[object]): Scaler instance used for feature scaling.
        target_scaler (Optional[object]): Scaler instance used for target scaling.
        feature_names (Optional[list]): List of feature names used in training.
        metrics (Optional[dict]): Dictionary containing model evaluation metrics.
        history (Optional[object]): Training history from model fitting.
    """

    def __init__(self,
                 units=50,
                 dropout_rate=0.2,
                 learning_rate=0.001,
                 n_lags=10,
                 time_steps=5
                 ):
        """
        Inicializa el modelo LSTM con parámetros configurables

        Parámetros:
        - units: Número de unidades en la capa LSTM
        - dropout_rate: Tasa de dropout para evitar sobreajuste
        - learning_rate: Tasa de aprendizaje para el optimizador
        - n_lags: Número de características de retardo a crear
        - time_steps: Número de pasos de tiempo para secuencias
        """
        self.units = units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.n_lags = n_lags
        self.time_steps = time_steps
        self.model = None
        self.best_params_ = None
        self.feature_scaler = None
        self.target_scaler = None
        self.feature_names = None
        self.metrics = None
        self.history = None

    def _build_model(self, input_shape, output_shape=1):
        """
        Builds and compiles a Sequential LSTM-based model with dropout layers and a dense output layer.

        The method creates a model for time-series or sequence prediction tasks. The model consists of
        stacked LSTM layers, each followed by a Dropout layer to prevent overfitting. The output layer
        is a Dense layer to map the input to the desired number of output dimensions. The Adam optimizer
        is used for compiling the model with Mean Squared Error as the loss metric.

        Parameters
        ----------
        input_shape : tuple
            The shape of the input data for the first LSTM layer, excluding the batch size.
        output_shape : int, optional
            The number of output dimensions for the Dense layer (default is 1).

        Returns
        -------
        Sequential
            A compiled LSTM-based model ready for training.
        """
        model = Sequential()
        model.add(LSTM(units=self.units,
                       return_sequences=True,
                       input_shape=input_shape))
        model.add(Dropout(self.dropout_rate))
        model.add(LSTM(units=self.units // 2,
                       return_sequences=False))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(output_shape))

        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        return model

    def prepare_data(self, data, target_col='Close'):
        """
        Prepares the input data for modeling tasks by sorting it (if required),
        adding lag features, and optionally applying additional feature engineering.

        Parameters:
            data (pd.DataFrame): The input data containing time-series and other feature
                columns. It must include the target column specified by `target_col`.
            target_col (str): The name of the target column for which lag features
                will be generated. Default is 'Close'.

        Returns:
            pd.DataFrame: The transformed DataFrame containing the original data with
            lag features and possibly additional engineered features.

        Notes:
            - The input data will be sorted by its index if it is a DatetimeIndex and not already sorted.
            - Lag features are always created.
            - Additional feature engineering will be applied if the input data contains the
              required columns.
            - If required columns for feature engineering are missing, a warning will be displayed,
              and only lag features will be returned without raising an error.
        """

        from utils.preprocessing import feature_engineering, add_lags

        # Ordenar los datos por fecha si no está ordenado
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.sort_index()

        # Crear características de rezagos
        data_with_lags = add_lags(data, target_col=target_col, n_lags=self.n_lags)

        # Crear características adicionales
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

    def create_sequences(self, X, y=None, time_steps=None):
        """
        Creates sequences from input data (X) and optionally corresponding target
        data (y), based on provided time steps or a default time_steps value. This
        is useful for preparing time-series data for training and prediction tasks.

        Parameters:
        X : list or numpy.ndarray
            Input feature data for sequence creation, must support slicing and
            indexing operations.
        y : list, numpy.ndarray, or None
            Optional target data corresponding to the input features. If provided,
            sequences will include features and target pairs. Default is None.
        time_steps : int or None
            Window size for sequence creation. If not provided, a default value of
            the instance's 'time_steps' is used.

        Returns:
        numpy.ndarray
            A 3D array of created sequences when target (y) is None, otherwise a
            tuple containing sequences of input features and corresponding targets

        Raises:
        ValueError
            If X is not compatible for slicing or its length does not allow sequence
            creation with the given number of time steps.
        """
        if time_steps is None:
            time_steps = self.time_steps

        from utils.preprocessing import create_sequences

        if y is not None:
            return create_sequences(X, y, time_steps)
        else:
            # Para predicción cuando no tenemos y
            X_array = np.array(X)
            X_seq = []
            for i in range(len(X_array) - time_steps + 1):
                X_seq.append(X_array[i:i + time_steps])
            return np.array(X_seq)

    def fit(self, X_train, y_train, validation_data=None, epochs=100, batch_size=32, patience=10, checkpoint_path=None):
        """
        Fits the model to the training data using specified configurations and trains the model.

        Parameters
        ----------
        X_train : numpy.ndarray
            The input training data with shape (samples, time_steps, features).
        y_train : numpy.ndarray
            The target values corresponding to the input training data.
        validation_data : tuple or None, optional
            A tuple (X_val, y_val) containing validation data for monitoring model performance.
            Default is None.
        epochs : int, optional
            The number of epochs for training the model. Default is 100.
        batch_size : int, optional
            The number of samples per gradient update. Default is 32.
        patience : int, optional
            The number of epochs with no improvement after which training will be stopped early.
            This is used with the EarlyStopping callback. Default is 10.
        checkpoint_path : str or None, optional
            The file path to save the best model weights during training. If None, the model
            checkpoint callback will not be used. Default is None.

        Returns
        -------
        self
            The instance of the class where this function is used, allowing for chaining
            of methods after execution.

        Raises
        ------
        ValueError
            If the input `X_train` does not have the required shape of (samples, time_steps, features).
        """
        # Asegurarse de que los datos tienen la forma correcta
        if len(X_train.shape) != 3:
            raise ValueError("X_train debe tener forma (muestras, time_steps, características)")

        input_shape = (X_train.shape[1], X_train.shape[2])  # (time_steps, n_features)

        # Construir modelo
        self.model = self._build_model(input_shape)

        # Configurar callbacks
        callbacks = [EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)]

        if checkpoint_path:
            callbacks.append(ModelCheckpoint(
                filepath=checkpoint_path,
                save_best_only=True,
                monitor='val_loss'
            ))

        # Entrenar modelo
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )

        return self

    def predict(self, X):
        """
        Predicts outcomes based on the input data using the trained model.

        The method uses the trained model to generate predictions for the provided input
        data. Ensure that the model is trained before calling this method; otherwise, an
        exception will be raised.

        Args:
            X: The input data on which predictions are to be made.

        Raises:
            ValueError: If the model is not trained prior to making the predictions.

        Returns:
            The predictions generated by the trained model for the input data.
        """
        if self.model is None:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones")

        return self.model.predict(X)

    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, param_grid=None, feature_names=None):
        """
        Optimize hyperparameters of a neural network using a manual grid search over
        the provided parameter space.

        Parameters
        ----------
        X_train : numpy.ndarray
            Training set features for model input.
        y_train : numpy.ndarray
            Corresponding target values for the training set.
        X_val : numpy.ndarray
            Validation set features for evaluating model performance.
        y_val : numpy.ndarray
            Corresponding target values for the validation set.
        param_grid : dict, optional
            A dictionary defining the grid search parameters. Possible keys are:
            'units', 'dropout_rate', 'learning_rate', and 'time_steps'. If not provided,
            a default grid search configuration is used.
        feature_names : list of str, optional
            List of feature names corresponding to input features.

        Returns
        -------
        self : object
            The instance of the class with optimized hyperparameters and the best
            model configuration.

        Raises
        ------
        None

        Notes
        -----
        This method performs a manual grid search over specified hyperparameter ranges
        and identifies the best performing configuration based on the validation loss.
        It reconstructs the model with the optimal parameters and tracks the best
        parameters identified during the process.

        Warnings
        --------
        Training and evaluation within this process may consume significant computational
        resources, especially if a large search grid or dataset is used. To minimize
        overfitting and ensure faster execution, early stopping and a limited number
        of epochs have been incorporated.
        """
        # Guardar los nombres de las características
        self.feature_names = feature_names

        # Configuración predeterminada de búsqueda si no se proporciona
        if param_grid is None:
            param_grid = {
                'units': [50, 100, 200],
                'dropout_rate': [0.1, 0.2, 0.3],
                'learning_rate': [0.001, 0.01],
                'time_steps': [3, 5, 10]
            }

        best_val_loss = float('inf')
        best_params = {}

        # Realizar búsqueda manual
        print("Iniciando optimización de hiperparámetros...")
        for units in param_grid['units']:
            for dropout_rate in param_grid['dropout_rate']:
                for learning_rate in param_grid['learning_rate']:
                    for time_steps in param_grid['time_steps']:
                        print(
                            f"\nProbando: units={units}, dropout={dropout_rate}, lr={learning_rate}, time_steps={time_steps}")

                        # Actualizar hiperparámetros
                        self.units = units
                        self.dropout_rate = dropout_rate
                        self.learning_rate = learning_rate
                        self.time_steps = time_steps

                        # Crear secuencias con los nuevos time_steps
                        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train, time_steps)
                        X_val_seq, y_val_seq = self.create_sequences(X_val, y_val, time_steps)

                        # Construir y entrenar modelo
                        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
                        self.model = self._build_model(input_shape)

                        # Entrenar con early stopping
                        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                        history = self.model.fit(
                            X_train_seq, y_train_seq,
                            epochs=50,  # Pocas épocas para buscar rápido
                            batch_size=32,
                            validation_data=(X_val_seq, y_val_seq),
                            callbacks=[early_stop],
                            verbose=0
                        )

                        # Evaluar el modelo
                        val_loss = self.model.evaluate(X_val_seq, y_val_seq, verbose=0)
                        print(f"Validación loss: {val_loss}")

                        # Actualizar mejor modelo si es necesario
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_params = {
                                'units': units,
                                'dropout_rate': dropout_rate,
                                'learning_rate': learning_rate,
                                'time_steps': time_steps
                            }
                            print(f"Nuevo mejor modelo encontrado: {best_params}")

        # Actualizar modelo con los mejores parámetros
        print(f"\nMejores parámetros: {best_params}")
        self.units = best_params['units']
        self.dropout_rate = best_params['dropout_rate']
        self.learning_rate = best_params['learning_rate']
        self.time_steps = best_params['time_steps']
        self.best_params_ = best_params

        # Reconstruir el modelo con los mejores parámetros
        self.model = None  # Liberar memoria

        return self

    def evaluate(self, X_test, y_test):
        """
        Evaluates a trained model using a specified test dataset.

        This method generates predictions for the given test dataset and evaluates
        the model's performance using the provided target data. If a target scaler
        is used, it ensures predictions and actual values are transformed back
        to the original scale before evaluation.

        Parameters:
        X_test : numpy.ndarray
            Features of the test dataset used for generating predictions.
        y_test : numpy.ndarray
            Actual target values of the test dataset.

        Returns:
        dict
            A dictionary containing evaluation metrics for the model's performance.

        Raises:
        ValueError
            If the model has not been trained prior to calling the evaluate method.
        """
        from utils.evaluation import evaluate_regression

        if self.model is None:
            raise ValueError("El modelo debe ser entrenado antes de evaluarlo")

        y_pred = self.model.predict(X_test)

        # Convertir predicciones de vuelta a la escala original
        if self.target_scaler:
            y_pred = self.target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            y_test_original = self.target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            self.metrics = evaluate_regression(y_test_original, y_pred)
        else:
            self.metrics = evaluate_regression(y_test, y_pred)

        return self.metrics

    def predict_future(self, X_last, forecast_horizon):
        """
        Predicts a future sequence of values based on the last observed input and a trained model.

        This method generates a forecast for a specified horizon by using a recursive approach.
        It repeatedly uses the model to predict the next value and updates the input sequence
        to include the predicted value, ensuring the forecast utilizes previously predicted
        results.

        Attributes:
            model: A trained predictive model that must implement a `predict` method. This model
                is used for generating future predictions.

        Parameters:
            X_last: A NumPy array or DataFrame representing the most recent input values used
                to initiate the predictions. It can either be in its raw state (2D) or already
                formatted as a sequence (3D).
            forecast_horizon: An integer specifying the number of future steps to predict.

        Returns:
            numpy.ndarray: A 1D array containing the future predicted values for the entire
                forecast horizon.

        Raises:
            ValueError: Raised if the model has not been trained prior to calling this method.
        """
        if self.model is None:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones")

        # Convertir X_last en formato de secuencia si todavía no lo está
        if len(X_last.shape) != 3:
            # Asumimos que X_last es un DataFrame o array 2D
            X_sequence = self.create_sequences(X_last)

            # Tomar solo la última secuencia para la predicción recursiva
            current_sequence = X_sequence[-1:].copy()  # [1, time_steps, n_features]
        else:
            # Ya está en formato correcto, sólo copiamos
            current_sequence = X_last.copy()

        predictions = []

        for _ in range(forecast_horizon):
            # Predecir el siguiente valor
            next_pred = self.model.predict(current_sequence)[0]
            predictions.append(next_pred)

            # Actualizar la secuencia para la siguiente predicción
            # Descartamos el primer paso de tiempo y añadimos la nueva predicción al final
            new_sequence = np.zeros_like(current_sequence)
            # Transferir todos los pasos de tiempo excepto el primero
            new_sequence[0, :-1, :] = current_sequence[0, 1:, :]

            # Añadir la nueva predicción como el último valor
            # Esto asume que la última columna del input es el valor a predecir
            new_sequence[0, -1, -1] = next_pred

            # Actualizar la secuencia actual
            current_sequence = new_sequence

        return np.array(predictions)

    def plot_results(self, y_true, y_pred, title="LSTM Model Predictions"):
        """
        Plots the true and predicted results for a given dataset using the LSTM model.

        This method provides a visualization of the comparison between the true values
        and the predicted values from the model. It also allows the user to specify a
        custom title for the plot.

        Args:
            y_true (list or np.ndarray): The actual values from the dataset.
            y_pred (list or np.ndarray): The predicted values generated by the model.
            title (str, optional): The title of the plot. Defaults to "LSTM Model Predictions".

        Returns:
            None
        """
        from utils.visualizations import plot_predictions

        plot_predictions(y_true, y_pred, title=title)

    def plot_forecast(self, historical_data, forecast_values, target_col='Close'):
        """
        Visualize the forecasted values alongside historical data for a specific target column.

        This function plots the historical data and forecasted values, providing a clear visual representation
        of the predicted trends for the designated target column.

        Parameters:
        historical_data : DataFrame
            The historical dataset containing the time series data for the target column.
        forecast_values : DataFrame
            The forecasted dataset containing the predictions for the target column.
        target_col : str, optional
            The name of the column containing the target metric. Default is 'Close'.

        Returns:
        None
        """
        from utils.visualizations import plot_forecast

        plot_forecast(historical_data, forecast_values, target_col=target_col)

    def plot_training_history(self):
        """
        Plots the training history of the model.

        This method visualizes the loss values tracked during the training of the
        model, using matplotlib. It plots the training loss and, if available, the
        validation loss against the number of epochs. The figure produced includes a
        title, axis labels, a legend for clarity, and a grid for easier readability.

        Raises:
            ValueError: If the training history is not available (i.e., the model has
                not been trained yet).
        """
        if self.history is None:
            raise ValueError("El modelo debe ser entrenado antes de mostrar el historial")

        plt.figure(figsize=(12, 5))
        plt.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def save_model(self, model_path="models/lstm_model"):
        """
        Saves the trained model and its associated metadata to disk.

        This method saves a trained Keras model to the specified path and exports its
        metadata, including training parameters, scalers, and metrics, in pickle and
        JSON formats. This ensures the model and relevant information can be
        reproduced or deployed later.

        Parameters:
            model_path (str): The directory path and base name at which to save the
                model and metadata files. Defaults to "models/lstm_model".

        Raises:
            ValueError: If the model has not yet been trained and is therefore `None`.
        """
        if self.model is None:
            raise ValueError("El modelo debe ser entrenado antes de guardarlo")

        # Asegurarse de que el directorio de destino existe
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Guardar el modelo Keras
        keras_path = f"{model_path}.h5"
        self.model.save(keras_path)

        # Guardar los metadatos y scalers
        import pickle
        metadata = {
            'best_params': self.best_params_,
            'units': self.units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'n_lags': self.n_lags,
            'time_steps': self.time_steps,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'timestamp': pd.Timestamp.now().isoformat()
        }

        # Guardar metadatos como pickle
        metadata_path = f"{model_path}_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        # Guardar versión legible en JSON
        json_metadata = {
            'best_params': self.best_params_,
            'units': self.units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'n_lags': self.n_lags,
            'time_steps': self.time_steps,
            'metrics': self.metrics,
            'timestamp': pd.Timestamp.now().isoformat()
        }

        json_path = f"{model_path}_metadata.json"
        with open(json_path, 'w') as f:
            json.dump({k: str(v) if not isinstance(v, (int, float, list, dict)) else v
                       for k, v in json_metadata.items()}, f, indent=4)

        print(f"Modelo guardado en {keras_path}")
        print(f"Metadatos guardados en {metadata_path} y {json_path}")

    @classmethod
    def load_model(cls, model_path="models/lstm_model"):
        """
            Loads a pre-trained Keras model and its associated metadata from disk, creating
            and returning an instance of the class initialized with the stored parameters.

            Args:
                model_path (str): The path to the base file of the Keras model and metadata
                    files (without file extensions). Defaults to "models/lstm_model".

            Returns:
                cls: An instance of the class initialized with the model parameters and
                    metadata stored in the specified path.
        """
        # Cargar el modelo Keras
        keras_path = f"{model_path}.h5"
        keras_model = load_model(keras_path)

        # Cargar metadatos
        import pickle
        metadata_path = f"{model_path}_metadata.pkl"

        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        # Crear nueva instancia
        instance = cls(
            units=metadata.get('units', 50),
            dropout_rate=metadata.get('dropout_rate', 0.2),
            learning_rate=metadata.get('learning_rate', 0.001),
            n_lags=metadata.get('n_lags', 10),
            time_steps=metadata.get('time_steps', 5)
        )

        # Asignar atributos
        instance.model = keras_model
        instance.best_params_ = metadata.get('best_params')
        instance.feature_names = metadata.get('feature_names')
        instance.metrics = metadata.get('metrics')
        instance.feature_scaler = metadata.get('feature_scaler')
        instance.target_scaler = metadata.get('target_scaler')

        return instance