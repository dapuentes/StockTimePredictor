from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
import joblib
import numpy as np
import pandas as pd
import os
import json


class TimeSeriesRandomForestModel:
    """
    A machine learning model for time series forecasting based on Random Forest.

    This class implements a Random Forest-based model configured for handling
    time series data. It includes methods for preparing data with feature
    engineering, optimizing hyperparameters, fitting the model, making predictions,
    and evaluating performance. It supports advanced functionality such as recursive
    future prediction and allows customization of feature and lag configurations.

    Attributes:
        model (RandomForestRegressor): The base Random Forest regression model.
        feature_importances_ (Optional[array]): The importance of each feature after
            training the model.
        best_params_ (Optional[dict]): The best hyperparameters selected during
            optimization.
        n_lags (int): Number of lag features to create for time series forecasting.
        feature_scaler (Optional[object]): Scaler instance used for feature scaling.
        target_scaler (Optional[object]): Scaler instance used for target scaling.
        best_pipeline_ (Optional[Pipeline]): The optimized pipeline after hyperparameter
            tuning.
        feature_names (Optional[list]): List of feature names used in training.
    """

    def __init__(self,
                 n_estimators=100,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features='log2',
                 n_lags=10
                 ):
        """
        Inicializa el modelo de Random Forest con parámetros configurables

        Parámetros:
        - n_estimators: Número de árboles
        - max_depth: Profundidad máxima de los árboles
        - min_samples_split: Mínimo de muestras para dividir un nodo interno
        - min_samples_leaf: Mínimo de muestras en un nodo hoja
        - max_features: Número de características a considerar para una mejor división
        - n_lags: Número de características de retardo a crear
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42
        )
        self.feature_importances_ = None
        self.best_params_ = None
        self.n_lags = n_lags
        self.feature_scaler = None
        self.target_scaler = None
        self.best_pipeline_ = None
        self.feature_names = None

    def prepare_data(self, data, target_col='Close'):
        """
        Preparar datos de series temporales con ingeniería de características

        Parámetros:
        - data: DataFrame de entrada que contiene los datos
        - target_col: Nombre de la columna de destino para la predicción (el valor predeterminado es 'Close')

        Devuelve:
        - DataFrame procesado con características
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

    def fit(self, X_train, y_train):
        """
        Entrenar el modelo de Random Forest

        Parámetros:
        - X_train: Características de entrenamiento
        - y_train: Valores objetivo de entrenamiento
        """
        self.best_pipeline_.fit(X_train, y_train)  # Ajustar el modelo
        return self

    def predict(self, X):
        """
        Realizar predicciones usando el modelo entrenado.

        Parámetros:
        - X: Datos de entrada para la predicción

        Devuelve:
        - Predicciones realizadas por el modelo.
        """
        return self.best_pipeline_.predict(X)

    def optimize_hyperparameters(self, X_train, y_train, feature_names=None, param_grid=None, cv=3):
        """
        Optimizar hiperparámetros utilizando GridSearchCV

        Parámetros:
        - X_train: Características de entrenamiento (array NumPy)
        - y_train: Valores objetivo de entrenamiento
        - feature_names: Lista de nombres de las características
        - param_grid: Diccionario de hiperparámetros a buscar
        - cv: Número de pliegues de validación cruzada

        Devuelve:
        - Modelo optimizado
        """
        # Guardar los nombres de las características
        self.feature_names = feature_names

        if feature_names is None or len(feature_names) == 0:
            raise ValueError("Se requiere proporcionar feature_names para la selección de características")

        estimator_for_selection = RandomForestRegressor(n_estimators=50, random_state=42)

        if param_grid is None:
            param_grid = {
                'selector__max_features': [min(10, len(feature_names)), min(15, len(feature_names)),
                                           min(20, len(feature_names))],
                'rf__n_estimators': [100, 200],
                'rf__max_depth': [10, 15, 20, None],
                'rf__min_samples_split': [2, 5, 10],
                'rf__min_samples_leaf': [3, 5, 7],
                'rf__max_features': ['sqrt', 'log2', 0.5, 0.7]
            }
            param_grid['selector__max_features'] = [mf for mf in param_grid['selector__max_features'] if
                                                    mf <= len(feature_names) and mf > 0]
            if not param_grid['selector__max_features']:  # Si la lista queda vacía
                param_grid['selector__max_features'] = [len(feature_names)]

        # Crear pipeline con selector, scaler y modelo
        pipeline = Pipeline([
            ('selector', SelectFromModel(estimator=estimator_for_selection)),
            ('rf', self.model)
        ])

        # Configurar validación cruzada para series temporales
        tscv = TimeSeriesSplit(n_splits=cv)

        # Realizar búsqueda en cuadrícula para optimizar hiperparámetros
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )

        print("Iniciando GridSearchCV con SelectFromModel...")
        grid_search.fit(X_train, y_train)
        print("GridSearchCV completado.")

        self.best_pipeline_ = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_

        selected_indices = self.best_pipeline_.named_steps['selector'].get_support(indices=True)
        self.selected_feature_names_ = [feature_names[i] for i in selected_indices]

        # Guardar los índices seleccionados en best_params_ para compatibilidad con la API
        self.best_params_['selector__features_index'] = selected_indices.tolist()

        print(f"Mejores parámetros encontrados: {self.best_params_}")
        print(f"Número de características seleccionadas por SelectFromModel: {len(self.selected_feature_names_)}")
        print(f"Características seleccionadas (nombres): {self.selected_feature_names_}")

        if hasattr(self.best_pipeline_.named_steps['rf'], 'feature_importances_'):
            # Estas son las importancias del RF final, entrenado SOLO con las features seleccionadas
            self.feature_importances_ = self.best_pipeline_.named_steps['rf'].feature_importances_

        return self

    def evaluate(self, X_test, y_test):
        """
        Evaluar el modelo utilizando métricas de rendimiento

        Parámetros:
        - X_test: Características de prueba
        - y_test: Valores objetivo de prueba

        Devuelve:
        - Diccionario con las métricas de evaluación
        """

        from utils.evaluation import evaluate_regression

        y_pred = self.best_pipeline_.predict(X_test)
        y_pred = self.target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        self.metrics = evaluate_regression(y_test, y_pred)

        return self.metrics

    def predict_future(self, historical_data_df, forecast_horizon, target_col='Close'):
        print("\n--- Entrando a predict_future (FINAL, con escalado externo consistente y chequeo de features) ---")

        # Verificaciones iniciales de atributos
        if not self.best_pipeline_: raise ValueError("El modelo (best_pipeline_) no ha sido ajustado.")
        if not hasattr(self, 'target_scaler') or self.target_scaler is None: raise ValueError(
            "target_scaler no disponible.")
        if not hasattr(self, 'feature_scaler') or self.feature_scaler is None: raise ValueError(
            "feature_scaler no disponible.")
        if not hasattr(self, 'feature_names') or self.feature_names is None or len(self.feature_names) == 0:
            raise ValueError("feature_names no establecidos o vacíos.")

        # Determinar el número de features esperadas por el feature_scaler
        n_features_expected_by_scaler = 0
        if hasattr(self.feature_scaler, 'n_features_in_'):
            n_features_expected_by_scaler = self.feature_scaler.n_features_in_
        elif hasattr(self.feature_scaler, 'get_feature_names_out'):
            n_features_expected_by_scaler = len(self.feature_scaler.get_feature_names_out())
        else:  # Fallback si no se pueden obtener las features del scaler
            n_features_expected_by_scaler = len(self.feature_names)
            print(
                f"Advertencia: No se pudo determinar n_features_in_ de self.feature_scaler. Asumiendo {n_features_expected_by_scaler} basado en len(self.feature_names).")

        if len(self.feature_names) != n_features_expected_by_scaler:
            print(
                f"ALERTA CRÍTICA INICIAL: len(self.feature_names) ({len(self.feature_names)}) no coincide con las features esperadas por self.feature_scaler ({n_features_expected_by_scaler}).")
            # Esto podría indicar un problema al guardar/cargar el modelo o feature_names.

        current_data_df = historical_data_df.copy()
        predictions_scaled_list = []
        lower_bounds_scaled_list = []
        upper_bounds_scaled_list = []

        selector = self.best_pipeline_.named_steps.get('selector')
        rf_model = self.best_pipeline_.named_steps.get('rf')

        if not all([selector, rf_model]): raise RuntimeError("Pipeline no contiene 'selector' y 'rf'.")
        if not hasattr(rf_model, 'estimators_'): raise RuntimeError("RF en pipeline no tiene estimadores.")

        for i in range(forecast_horizon):
            print(f"\n--- Paso {i + 1}/{forecast_horizon} ---")
            # ... (impresiones de current_data_df.tail(3) si deseas) ...

            processed_features_df = self.prepare_data(current_data_df.copy(), target_col=target_col)
            if processed_features_df.empty: raise ValueError("prepare_data devolvió DataFrame vacío.")

            last_prepared_row_series = processed_features_df.iloc[-1]

            # Asegurar que current_input_for_pipeline tenga las columnas correctas y en el orden de self.feature_names
            # y que self.feature_names tenga el número correcto de features (ej. 41)
            try:
                current_input_for_pipeline = pd.DataFrame([last_prepared_row_series], columns=self.feature_names)
            except Exception as e:
                print(f"ERROR creando current_input_for_pipeline con self.feature_names. ¿Hay un desajuste?")
                print(f"Longitud de self.feature_names: {len(self.feature_names)}")
                print(f"Índice de last_prepared_row_series: {last_prepared_row_series.index.tolist()}")
                raise e

            if current_input_for_pipeline.shape[1] != n_features_expected_by_scaler:
                print(
                    f"ALERTA BUCLE: current_input_for_pipeline tiene {current_input_for_pipeline.shape[1]} cols, scaler espera {n_features_expected_by_scaler}.")
                # Considera detener o manejar este error si ocurre consistentemente.

            # 1. Escalar con self.feature_scaler (MinMaxScaler)
            scaled_features_N = self.feature_scaler.transform(current_input_for_pipeline)

            # 2. Predecir con el pipeline (selector -> rf) que espera datos escalados [0,1]
            point_pred_scaled = self.best_pipeline_.predict(scaled_features_N)[0]
            predictions_scaled_list.append(point_pred_scaled)
            print(f"point_pred_scaled (pipeline sobre data [0,1]): {point_pred_scaled:.8f}")

            # 3. Para intervalos y depuración del RF (opcional, pero bueno para verificar)
            selected_and_scaled_features_k = selector.transform(scaled_features_N)
            # print(f"selected_and_scaled_features_k ({selected_and_scaled_features_k.shape[1]} features para RF):")
            # print(selected_and_scaled_features_k)
            # point_pred_scaled_direct_rf = rf_model.predict(selected_and_scaled_features_k)[0]
            # print(f"point_pred_scaled (directo de rf_model): {point_pred_scaled_direct_rf:.8f}")

            individual_tree_preds_scaled = np.array(
                [tree.predict(selected_and_scaled_features_k)[0] for tree in rf_model.estimators_])
            lower_bounds_scaled_list.append(np.percentile(individual_tree_preds_scaled, 2.5))
            upper_bounds_scaled_list.append(np.percentile(individual_tree_preds_scaled, 97.5))

            point_pred_unscaled = \
            self.target_scaler.inverse_transform(np.array(point_pred_scaled).reshape(-1, 1)).flatten()[0]
            print(f"point_pred_unscaled: {point_pred_unscaled:.4f}")

            # Propagación de OHLG
            new_row_values = {col: np.nan for col in current_data_df.columns}
            new_row_values[target_col] = point_pred_unscaled
            # ... (resto de la lógica de OHLG y next_index como en la versión anterior que te funcionó para predicciones dinámicas) ...
            if not current_data_df.empty:
                prev_open_val = current_data_df['Open'].iloc[-1];
                prev_high_val = current_data_df['High'].iloc[-1]
                prev_low_val = current_data_df['Low'].iloc[-1];
                prev_close_val = current_data_df[target_col].iloc[-1]
                current_open = prev_close_val
                if 'Open' in new_row_values: new_row_values['Open'] = current_open
                if 'High' in new_row_values: new_row_values['High'] = current_open + (prev_high_val - prev_open_val)
                if 'Low' in new_row_values: new_row_values['Low'] = current_open - (prev_open_val - prev_low_val)
                # Consistencia OHL
                if 'High' in new_row_values and 'Open' in new_row_values and new_row_values['High'] < new_row_values[
                    'Open']: new_row_values['High'] = new_row_values['Open']
                if 'Low' in new_row_values and 'Open' in new_row_values and new_row_values['Low'] > new_row_values[
                    'Open']: new_row_values['Low'] = new_row_values['Open']
                if 'High' in new_row_values and 'Low' in new_row_values and new_row_values['High'] < new_row_values[
                    'Low']:
                    avg_ohl = (new_row_values.get('High', 0) + new_row_values.get('Low', 0)) / 2.0;
                    new_row_values['High'] = avg_ohl + abs(avg_ohl * 0.001) + 0.01;
                    new_row_values['Low'] = avg_ohl - abs(avg_ohl * 0.001) - 0.01  # Asegurar spread
            else:  # Fallback
                if 'Open' in new_row_values: new_row_values['Open'] = point_pred_unscaled * 0.998
                if 'High' in new_row_values: new_row_values['High'] = point_pred_unscaled * 1.002
                if 'Low' in new_row_values: new_row_values['Low'] = point_pred_unscaled * 0.995
            if 'GreenDay' in new_row_values:
                if not current_data_df.empty:
                    new_row_values['GreenDay'] = 1 if point_pred_unscaled > current_data_df[target_col].iloc[-1] else 0
                else:
                    new_row_values['GreenDay'] = 0

            last_index_val = current_data_df.index[-1];
            next_index_val = None
            if isinstance(last_index_val, pd.Timestamp):
                freq = pd.infer_freq(current_data_df.index) if len(current_data_df.index) >= 3 else 'D'
                if freq is None: freq = 'D'
                try:
                    next_index_val = last_index_val + pd.tseries.frequencies.to_offset(freq)
                except ValueError:
                    next_index_val = last_index_val + pd.Timedelta(days=1)
            elif pd.api.types.is_numeric_dtype(current_data_df.index.dtype):
                try:
                    next_index_val = last_index_val + 1
                except TypeError:
                    next_index_val = int(last_index_val) + 1
            if next_index_val is None: next_index_val = (current_data_df.index.max() if pd.api.types.is_numeric_dtype(
                current_data_df.index) else len(current_data_df) - 1) + 1

            new_row_df = pd.DataFrame([new_row_values], index=[next_index_val])
            current_data_df = pd.concat([current_data_df, new_row_df])

        # Desescalado final
        predictions_unscaled = self.target_scaler.inverse_transform(
            np.array(predictions_scaled_list).reshape(-1, 1)).flatten()
        lower_bounds_unscaled = self.target_scaler.inverse_transform(
            np.array(lower_bounds_scaled_list).reshape(-1, 1)).flatten()
        upper_bounds_unscaled = self.target_scaler.inverse_transform(
            np.array(upper_bounds_scaled_list).reshape(-1, 1)).flatten()

        print("--- Saliendo de predict_future ---")
        return predictions_unscaled, lower_bounds_unscaled, upper_bounds_unscaled

    def plot_results(self, y_true, y_pred, title="Model Predictions"):
        """
        Graficar los resultados de la predicción

        Parámetros:
        - y_true: Valores verdaderos
        - y_pred: Valores predichos
        - title: Título del gráfico
        """
        from utils.visualizations import plot_predictions

        plot_predictions(y_true, y_pred, title=title)

    def plot_forecast(self, historical_data, forecast_values, target_col='Close'):
        """
        Graficar el pronóstico comparado con los datos históricos

        Parámetros:
        - historical_data: Datos históricos
        - forecast_values: Valores pronosticados
        - target_col: Nombre de la columna objetivo
        """
        from utils.visualizations import plot_forecast

        plot_forecast(historical_data, forecast_values, target_col=target_col)

    def save_model(self, model_path="models/model.joblib", training_end_date=None):
        """
        Guardar el modelo entrenado en un archivo

        Parámetros:
        - model_path: Ruta del archivo para guardar el modelo (el valor predeterminado es 'models/model.joblib')
        """

        # Asegurarse de que el directorio de destino existe
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Guardar el modelo
        joblib.dump(self, model_path)

        # Guardar metadatos del modelo
        metadata = {
            'best_params': self.best_params_,
            'feature_importances': self.feature_importances_.tolist() if self.feature_importances_ is not None else None,
            'metrics': self.metrics,
            'timestamp': pd.Timestamp.now().isoformat(),
            'training_end_date': training_end_date
        }

        # Generar la ruta para los metadatos, reemplazando la extensión
        if model_path.endswith('.joblib'):
            metadata_file = model_path.replace('.joblib', '_metadata.json')
        elif model_path.endswith('.pkl'):
            metadata_file = model_path.replace('.pkl', '_metadata.json')
        else:
            metadata_file = model_path + '_metadata.json'

        with open(metadata_file, 'w') as f:
            json.dump({k: str(v) if not isinstance(v, (int, float)) else v
                       for k, v in metadata.items()}, f, indent=4)

    @classmethod
    def load_model(cls, model_path):
        """
        Cargar un modelo previamente guardado

        Parámetros:
        - model_path: Ruta del archivo del modelo

        Devuelve:
        - Instancia del modelo cargado
        """
        return joblib.load(model_path)


