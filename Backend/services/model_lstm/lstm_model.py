from tensorflow.keras.models import Sequential, load_model as keras_load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow as tf  # Importar TensorFlow
# Ensure KerasTuner is installed: pip install keras-tuner
from kerastuner.tuners import RandomSearch, BayesianOptimization, Hyperband
from kerastuner import HyperModel
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os
import json
import pickle
import matplotlib.pyplot as plt

# Import specific functions from your utility modules
from utils.preprocessing import feature_engineering, add_lags
from utils.evaluation import evaluate_regression


class SequenceGenerator:
    def __init__(self, n_lags=10):
        self.n_lags = n_lags
        self.target_steps = 1

    def create_sequences(self, X_data, y_data):
        X_seq, y_seq = [], []
        if len(y_data.shape) == 1:
            y_data = y_data.reshape(-1, 1)
        if len(X_data) <= self.n_lags:
            raise ValueError(f"Not enough data to create sequences. Need > {self.n_lags} samples, got {len(X_data)}.")
        for i in range(len(X_data) - self.n_lags):
            X_seq.append(X_data[i:(i + self.n_lags)])
            y_seq.append(y_data[i + self.n_lags])  # Target is at step i + n_lags
        if not X_seq:
            return np.array([]).reshape(0, self.n_lags, X_data.shape[1] if X_data.ndim > 1 else 1), np.array(
                []).reshape(0, 1)
        return np.array(X_seq), np.array(y_seq)


class TimeSeriesLSTMModel:
    def __init__(self,
                 units=50,
                 layers=1,
                 dropout_rate=0.2,
                 learning_rate=0.001,
                 n_lags=10
                 ):
        self.units = units
        self.layers = layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.n_lags = n_lags
        self.model_output_steps = 1
        self.model = None
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.feature_names = None
        self.sequence_generator = SequenceGenerator(n_lags=self.n_lags)
        self.metrics = {}
        self.history = None
        self.best_hyperparameters = None

    def _build_model(self, n_features, units=None, layers=None, dropout_rate=None, learning_rate=None):
        units = units if units is not None else self.units
        layers = layers if layers is not None else self.layers
        dropout_rate = dropout_rate if dropout_rate is not None else self.dropout_rate
        learning_rate = learning_rate if learning_rate is not None else self.learning_rate

        model = Sequential()
        for i in range(layers):
            return_sequences = i < layers - 1
            if i == 0:
                model.add(LSTM(units=units,
                               return_sequences=return_sequences,
                               input_shape=(self.n_lags, n_features)))
            else:
                model.add(LSTM(units=units, return_sequences=return_sequences))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        model.add(Dense(self.model_output_steps))
        optimizer = Adam(learning_rate=learning_rate)
        # Usar el objeto de función para la pérdida
        model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(),
                      metrics=[tf.keras.metrics.MeanAbsoluteError()])
        self.model = model
        return model

    def prepare_data(self, data, target_col='Close'):
        data_copy = data.copy()
        if not isinstance(data_copy.index, pd.DatetimeIndex):
            try:
                data_copy.index = pd.to_datetime(data_copy.index)
            except Exception as e:
                print(f"Warning: Could not convert DataFrame index to DatetimeIndex: {e}")
        if isinstance(data_copy.index, pd.DatetimeIndex):
            data_copy = data_copy.sort_index()
        data_with_lags = add_lags(data_copy, target_col=target_col, n_lags=self.n_lags)
        try:
            if 'Close' not in data_with_lags.columns:
                print(f"Warning: 'Close' column not found for feature_engineering. Using only lag features.")
                processed_data = data_with_lags
            else:
                processed_data = feature_engineering(data_with_lags)
        except Exception as e:
            print(f"Error during feature_engineering: {e}. Using only lag features.")
            processed_data = data_with_lags
        return processed_data.dropna()

    def preprocess_data(self, data_df, target_col='Close', is_training=True):
        if target_col not in data_df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in DataFrame. Available columns: {data_df.columns.tolist()}")
        y_values = data_df[target_col].values.reshape(-1, 1)
        feature_cols = data_df.drop(columns=[target_col]).columns
        X_df_features_only = data_df[feature_cols].apply(pd.to_numeric, errors='coerce')
        if X_df_features_only.isnull().values.any():
            print("Warning: NaNs found in feature data after numeric conversion. Filling with 0.")
            X_df_features_only = X_df_features_only.fillna(0)
        X_values = X_df_features_only.values

        if is_training:
            self.feature_names = feature_cols.tolist()
            X_scaled = self.feature_scaler.fit_transform(X_values)
            y_scaled = self.target_scaler.fit_transform(y_values)
        else:
            if self.feature_names is None:
                print("Warning: feature_names not set. Assuming prediction data columns match training.")
            if hasattr(self.feature_scaler, 'n_features_in_') and X_values.shape[
                1] != self.feature_scaler.n_features_in_:
                raise ValueError(
                    f"Feature count mismatch: scaler expects {self.feature_scaler.n_features_in_}, got {X_values.shape[1]}.")
            X_scaled = self.feature_scaler.transform(X_values)
            y_scaled = self.target_scaler.transform(y_values)
        return X_scaled, y_scaled.flatten()

    def fit(self, train_df, target_col='Close', validation_df=None, epochs=100, batch_size=32, verbose=1):
        X_train_scaled, y_train_scaled = self.preprocess_data(train_df, target_col, is_training=True)
        X_train_seq, y_train_seq = self.sequence_generator.create_sequences(X_train_scaled, y_train_scaled)
        if X_train_seq.shape[0] == 0:
            raise ValueError("No sequences generated from training data.")
        n_features = X_train_seq.shape[2]
        if self.model is None or self.model.input_shape[-1] != n_features:
            self._build_model(n_features=n_features)

        val_data_processed = None
        if validation_df is not None and not validation_df.empty:
            X_val_scaled, y_val_scaled = self.preprocess_data(validation_df, target_col, is_training=False)
            if len(X_val_scaled) > self.n_lags:
                X_val_seq, y_val_seq = self.sequence_generator.create_sequences(X_val_scaled, y_val_scaled)
                if X_val_seq.shape[0] > 0:
                    val_data_processed = (X_val_seq, y_val_seq)
                else:
                    print("Warning: No sequences from validation data for Keras fit.")
            else:
                print(f"Warning: Validation data too short for {self.n_lags} lags for Keras fit.")

        # Cambiar extensión a .keras para ModelCheckpoint
        checkpoint_filepath = 'best_lstm_model.keras'
        callbacks = [
            EarlyStopping(monitor='val_loss' if val_data_processed else 'loss', patience=15, restore_best_weights=True,
                          verbose=verbose),
            ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True,
                            monitor='val_loss' if val_data_processed else 'loss', verbose=verbose)
        ]
        self.history = self.model.fit(
            X_train_seq, y_train_seq, epochs=epochs, batch_size=batch_size,
            validation_data=val_data_processed, callbacks=callbacks, verbose=verbose
        )
        if os.path.exists(checkpoint_filepath):
            print(f"Loading best model from {checkpoint_filepath}")
            self.model = keras_load_model(
                checkpoint_filepath)  # No custom_objects needed for standard losses/optimizers with .keras
        return self

    def predict(self, data_df, target_col='Close'):
        if self.model is None: raise ValueError("Model not trained or loaded.")
        temp_data_df = data_df.copy()
        if target_col not in temp_data_df.columns: temp_data_df[target_col] = 0
        X_pred_scaled, _ = self.preprocess_data(temp_data_df, target_col, is_training=False)
        if len(X_pred_scaled) < self.n_lags:
            raise ValueError(f"Prediction data must have at least {self.n_lags} samples, got {len(X_pred_scaled)}.")
        dummy_y_for_seq = np.zeros(len(X_pred_scaled))
        X_pred_seq, _ = self.sequence_generator.create_sequences(X_pred_scaled, dummy_y_for_seq)
        if X_pred_seq.shape[0] == 0:
            print("Warning: No sequences for prediction. Returning empty array.")
            return np.array([])
        y_pred_scaled = self.model.predict(X_pred_seq, verbose=0)
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled)
        return y_pred.flatten()

    def optimize_hyperparameters(self, train_df, target_col='Close', validation_df=None,
                                 strategy="bayesian", max_trials=10, epochs_per_trial=50,
                                 project_name="lstm_tuning"):
        X_train_scaled, y_train_scaled = self.preprocess_data(train_df, target_col, is_training=True)
        X_train_seq, y_train_seq = self.sequence_generator.create_sequences(X_train_scaled, y_train_scaled)
        if X_train_seq.shape[0] == 0: raise ValueError("No sequences from training data for HP tuning.")
        n_features = X_train_seq.shape[2]

        val_data_processed = None
        if validation_df is not None and not validation_df.empty:
            X_val_scaled, y_val_scaled = self.preprocess_data(validation_df, target_col, is_training=False)
            if len(X_val_scaled) > self.n_lags:
                X_val_seq, y_val_seq = self.sequence_generator.create_sequences(X_val_scaled, y_val_scaled)
                if X_val_seq.shape[0] > 0:
                    val_data_processed = (X_val_seq, y_val_seq)
                else:
                    print("Warning: No sequences from validation data for Keras Tuner.")
            else:
                print(f"Warning: Validation data too short for {self.n_lags} lags for Keras Tuner.")

        hyper_model_instance_attrs = {'layers': self.layers, 'units': self.units, 'dropout_rate': self.dropout_rate,
                                      'learning_rate': self.learning_rate}
        hypermodel = LSTMHyperModel(self.n_lags, n_features, self.model_output_steps, **hyper_model_instance_attrs)

        common_tuner_params = {'hypermodel': hypermodel, 'objective': 'val_loss' if val_data_processed else 'loss',
                               'directory': 'tuner_dir', 'project_name': project_name, 'overwrite': True}
        if strategy == "random":
            tuner = RandomSearch(**common_tuner_params, max_trials=max_trials, executions_per_trial=1)
        elif strategy == "hyperband":
            tuner = Hyperband(**common_tuner_params, max_epochs=epochs_per_trial, factor=3)
        else:
            tuner = BayesianOptimization(**common_tuner_params, max_trials=max_trials, executions_per_trial=1)

        tuner.search_space_summary()
        print(f"Starting HP search with strategy: {strategy}")
        search_callbacks = [EarlyStopping(monitor='val_loss' if val_data_processed else 'loss', patience=10, verbose=1)]
        tuner.search(X_train_seq, y_train_seq, epochs=epochs_per_trial, batch_size=32,
                     validation_data=val_data_processed, callbacks=search_callbacks, verbose=1)

        best_hps_list = tuner.get_best_hyperparameters(num_trials=1)
        if not best_hps_list: raise ValueError("Keras Tuner found no best HPs.")
        self.best_hyperparameters = best_hps_list[0].values
        print(f"Best HPs: {self.best_hyperparameters}")

        self.units = self.best_hyperparameters.get('units', self.units)
        self.layers = self.best_hyperparameters.get('layers', self.layers)
        self.dropout_rate = self.best_hyperparameters.get('dropout_rate', self.dropout_rate)
        self.learning_rate = self.best_hyperparameters.get('learning_rate', self.learning_rate)

        self.model = tuner.hypermodel.build(best_hps_list[0])
        print("Retraining model with best HPs...")
        final_epochs = epochs_per_trial + 50 if epochs_per_trial < 100 else epochs_per_trial * 2
        self.fit(train_df, target_col, validation_df, epochs=final_epochs, batch_size=32, verbose=1)
        return self

    def evaluate(self, test_df, target_col='Close'):
        if self.model is None: raise ValueError("Model not trained or loaded.")
        X_test_scaled, y_test_scaled = self.preprocess_data(test_df, target_col, is_training=False)
        if len(X_test_scaled) <= self.n_lags:
            print(f"Warning: Test data too short for {self.n_lags} lags. Metrics will be NaN.")
            self.metrics = {metric: np.nan for metric in ['MSE', 'RMSE', 'MAE', 'MAPE']}
            return self.metrics
        X_test_seq, y_test_seq = self.sequence_generator.create_sequences(X_test_scaled, y_test_scaled)
        if X_test_seq.shape[0] == 0:
            print("Warning: No sequences from test data for evaluation. Metrics will be NaN.")
            self.metrics = {metric: np.nan for metric in ['MSE', 'RMSE', 'MAE', 'MAPE']}
            return self.metrics
        y_pred_scaled = self.model.predict(X_test_seq, verbose=0)
        y_pred_original_scale = self.target_scaler.inverse_transform(y_pred_scaled)
        y_true_original_scale = self.target_scaler.inverse_transform(y_test_seq)
        self.metrics = evaluate_regression(y_true_original_scale.flatten(), y_pred_original_scale.flatten())
        return self.metrics

    def predict_future(self, historical_data_df, forecast_horizon, target_col='Close'):
        if self.model is None: raise ValueError("Model not trained or loaded.")
        if not hasattr(self.feature_scaler, 'mean_'): raise ValueError("Feature scaler not fitted.")
        if not hasattr(self.target_scaler, 'mean_'): raise ValueError("Target scaler not fitted.")

        current_data_df = historical_data_df.copy()
        predictions_unscaled_list = []
        for i in range(forecast_horizon):
            processed_for_step_df = self.prepare_data(current_data_df, target_col)
            if len(processed_for_step_df) < self.n_lags:
                raise ValueError(
                    f"Step {i + 1}/{forecast_horizon}: Processed data too short ({len(processed_for_step_df)} rows) for {self.n_lags} lags.")
            latest_features_df = processed_for_step_df.iloc[-self.n_lags:]
            if target_col not in latest_features_df.columns:
                raise ValueError(f"Target column '{target_col}' missing in latest_features_df during predict_future.")
            X_latest_scaled, _ = self.preprocess_data(latest_features_df, target_col, is_training=False)
            if X_latest_scaled.shape[0] != self.n_lags:
                raise ValueError(
                    f"Step {i + 1}/{forecast_horizon}: Scaled features shape mismatch. Expected {self.n_lags} lags, got {X_latest_scaled.shape[0]}.")
            input_seq_scaled = X_latest_scaled.reshape(1, self.n_lags, X_latest_scaled.shape[1])
            prediction_scaled = self.model.predict(input_seq_scaled, verbose=0)[0]
            prediction_unscaled = self.target_scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()[0]
            predictions_unscaled_list.append(prediction_unscaled)

            last_known_index = current_data_df.index[-1]
            next_index_val = None
            if isinstance(last_known_index, pd.Timestamp):
                inferred_freq = pd.infer_freq(current_data_df.index[-3:]) if len(current_data_df.index) >= 3 else None
                base_freq = historical_data_df.index.freq
                current_offset = base_freq if base_freq else (
                    pd.tseries.frequencies.to_offset(inferred_freq) if inferred_freq else pd.Timedelta(days=1))
                next_index_val = last_known_index + current_offset
                if next_index_val.weekday() >= 5 and isinstance(current_offset,
                                                                pd.offsets.BusinessDay):  # Simple check for BDay
                    next_index_val = last_known_index + pd.offsets.BusinessDay(n=1)  # Ensure it's a business day
            elif pd.api.types.is_numeric_dtype(current_data_df.index.dtype):
                next_index_val = last_known_index + 1
            else:
                next_index_val = len(current_data_df)

            new_row_dict = {col: np.nan for col in historical_data_df.columns}
            new_row_dict[target_col] = prediction_unscaled
            if 'Open' in new_row_dict and target_col == 'Close' and 'Close' in current_data_df.columns: new_row_dict[
                'Open'] = current_data_df['Close'].iloc[-1]
            if 'High' in new_row_dict: new_row_dict['High'] = prediction_unscaled
            if 'Low' in new_row_dict: new_row_dict['Low'] = prediction_unscaled
            if 'GreenDay' in new_row_dict and target_col == 'Close' and 'Close' in current_data_df.columns:
            new_row_dict['GreenDay'] = 1 if prediction_unscaled > current_data_df['Close'].iloc[-1] else 0

            new_row_df = pd.DataFrame([new_row_dict], index=[next_index_val])
            new_row_df = new_row_df.reindex(columns=current_data_df.columns)
            current_data_df = pd.concat([current_data_df, new_row_df])
            if isinstance(current_data_df.index, pd.DatetimeIndex): current_data_df = current_data_df.sort_index()
        return np.array(predictions_unscaled_list)

    def plot_training_history(self):
        if self.history is None or not hasattr(self.history, 'history') or not self.history.history:
            print("No training history available or history is empty.")
            return
        plt.figure(figsize=(12, 6))
        if 'loss' in self.history.history:
            plt.plot(self.history.history['loss'], label='Training Loss')
        else:
            print("Warning: 'loss' not in training history.")
        if 'val_loss' in self.history.history:
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
        else:
            print("Note: 'val_loss' not in training history.")
        plt.title('Model Loss Over Epochs');
        plt.xlabel('Epochs');
        plt.ylabel('Loss');
        plt.legend();
        plt.show()

    def save_model(self, model_path_prefix="models/lstm_model", training_end_date=None):
        model_dir = os.path.dirname(model_path_prefix)
        if model_dir and not os.path.exists(model_dir): os.makedirs(model_dir, exist_ok=True)

        # Guardar en formato .keras
        model_file = f"{model_path_prefix}.keras"
        if self.model:
            self.model.save(model_file)
            print(f"Keras model saved to {model_file}")
        else:
            print(f"Warning: Model is None. Cannot save Keras model to {model_file}")

        scalers_params_file = f"{model_path_prefix}_scalers_params.pkl"
        with open(scalers_params_file, 'wb') as f:
            pickle.dump({'feature_scaler': self.feature_scaler, 'target_scaler': self.target_scaler,
                         'feature_names': self.feature_names, 'n_lags': self.n_lags, 'units': self.units,
                         'layers': self.layers, 'dropout_rate': self.dropout_rate,
                         'learning_rate': self.learning_rate, 'model_output_steps': self.model_output_steps,
                         'best_hyperparameters': self.best_hyperparameters}, f)
        print(f"Scalers and parameters saved to {scalers_params_file}")

        metadata = {'n_lags': self.n_lags, 'units': self.units, 'layers': self.layers,
                    'dropout_rate': self.dropout_rate, 'learning_rate': self.learning_rate,
                    'model_output_steps': self.model_output_steps,
                    'metrics': self.metrics if self.metrics else None,
                    'best_hyperparameters': self.best_hyperparameters if self.best_hyperparameters else None,
                    'feature_names': self.feature_names if self.feature_names else None,
                    'timestamp': pd.Timestamp.now().isoformat(), 'training_end_date': training_end_date}
        metadata_file = f"{model_path_prefix}_metadata.json"
        with open(metadata_file, 'w') as f:
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                if isinstance(obj, dict): return {k: convert_numpy_types(v) for k, v in obj.items()}
                if isinstance(obj, list): return [convert_numpy_types(i) for i in obj]
                if pd.isna(obj): return None
                return obj

            json.dump(convert_numpy_types(metadata), f, indent=4)
        print(f"Metadata saved to {metadata_file}")

    @classmethod
    def load_model(cls, model_path_prefix):
        # Cargar desde formato .keras
        model_file = f"{model_path_prefix}.keras"
        if not os.path.exists(model_file): raise FileNotFoundError(f"Keras model file not found: {model_file}")
        # No se necesitan custom_objects para pérdidas/optimizadores estándar con .keras
        keras_model_loaded = keras_load_model(model_file)

        scalers_params_file = f"{model_path_prefix}_scalers_params.pkl"
        if not os.path.exists(scalers_params_file): raise FileNotFoundError(
            f"Scalers/parameters file not found: {scalers_params_file}")
        with open(scalers_params_file, 'rb') as f:
            saved_data = pickle.load(f)

        instance = cls(units=saved_data.get('units', 50), layers=saved_data.get('layers', 1),
                       dropout_rate=saved_data.get('dropout_rate', 0.2),
                       learning_rate=saved_data.get('learning_rate', 0.001),
                       n_lags=saved_data['n_lags'])
        instance.model = keras_model_loaded
        instance.feature_scaler = saved_data['feature_scaler']
        instance.target_scaler = saved_data['target_scaler']
        instance.feature_names = saved_data.get('feature_names')
        instance.model_output_steps = saved_data.get('model_output_steps', 1)
        instance.best_hyperparameters = saved_data.get('best_hyperparameters')

        metadata_file = f"{model_path_prefix}_metadata.json"
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            instance.metrics = metadata.get('metrics', {})
            if instance.feature_names is None: instance.feature_names = metadata.get('feature_names')
        else:
            print(f"Warning: Metadata file not found at {metadata_file}. Metrics will be empty.")
            instance.metrics = {}
        print(f"TimeSeriesLSTMModel loaded from prefix: {model_path_prefix}")
        return instance


class LSTMHyperModel(HyperModel):
    def __init__(self, n_lags, n_features, model_output_steps, **kwargs):
        self.n_lags = n_lags
        self.n_features = n_features
        self.model_output_steps = model_output_steps
        self.initial_layers = kwargs.get('layers', 1)
        self.initial_units = kwargs.get('units', 50)
        self.initial_dropout_rate = kwargs.get('dropout_rate', 0.2)
        self.initial_learning_rate = kwargs.get('learning_rate', 0.001)

    def build(self, hp):
        model = Sequential()
        hp_layers = hp.Int('layers', min_value=1, max_value=3, default=self.initial_layers)
        hp_units = hp.Int('units', min_value=32, max_value=128, step=32, default=self.initial_units)
        hp_dropout = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1, default=self.initial_dropout_rate)
        for i in range(hp_layers):
            return_sequences = i < hp_layers - 1
            if i == 0:
                model.add(
                    LSTM(units=hp_units, return_sequences=return_sequences, input_shape=(self.n_lags, self.n_features)))
            else:
                model.add(LSTM(units=hp_units, return_sequences=return_sequences))
            model.add(BatchNormalization())
            model.add(Dropout(hp_dropout))
        model.add(Dense(self.model_output_steps))
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4], default=self.initial_learning_rate)
        # Usar el objeto de función para la pérdida y métricas
        model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=[tf.keras.metrics.MeanAbsoluteError()])
        return model
