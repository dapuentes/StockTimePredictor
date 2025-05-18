from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import numpy as np
import pandas as pd
import os
import json


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

        # Encontrar índices de características de lag y otros indicadores
        lag_indices = [i for i, name in enumerate(feature_names) if '_lag_' in name]

        # Lista de indicadores técnicos comunes
        indicator_names = ['SMA_5', 'RSI', 'SMA_20', 'EMA_12', 'EMA_26', '20d_std', 'MACD']

        # Obtener índices de los indicadores que existen en feature_names
        indicator_indices = []
        for indicator in indicator_names:
            indices = [i for i, name in enumerate(feature_names) if name == indicator]
            indicator_indices.extend(indices)

        # Si no hay configuración de parámetros, usar valores predeterminados
        if param_grid is None:
            # Crear diferentes combinaciones de características por índice
            feature_combinations = [
                lag_indices,  # Solo características de lag
                lag_indices + [i for i, name in enumerate(feature_names) if name == 'SMA_5' and i in indicator_indices],
                lag_indices + [i for i, name in enumerate(feature_names) if name == 'RSI' and i in indicator_indices],
                lag_indices + indicator_indices  # Lag + todos los indicadores disponibles
            ]

            # Eliminar combinaciones vacías o duplicadas
            feature_combinations = [list(set(combo)) for combo in feature_combinations if combo]
            if not feature_combinations:
                # Si no hay combinaciones válidas, usar todos los índices
                feature_combinations = [list(range(len(feature_names)))]

            param_grid = {
                'selector__features_index': feature_combinations,
                'rf__n_estimators': [50, 100, 200],
                'rf__max_depth': [3, 5, 7, 10],
                'rf__min_samples_split': [5, 10],
                'rf__min_samples_leaf': [10, 20],
                'rf__max_features': ['sqrt', 'log2', 0.5]
            }

        # Crear pipeline con selector, scaler y modelo
        pipeline = Pipeline([
            ('selector', FeatureSelector()),
            ('scaler', StandardScaler()),
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
            n_jobs=-1
        )

        # Ajustar la búsqueda en cuadrícula
        grid_search.fit(X_train, y_train)

        # Guardar el mejor modelo y parámetros
        self.best_pipeline_ = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_

        # Extraer las importancias de características del modelo RF en el pipeline
        if hasattr(self.best_pipeline_, 'named_steps') and 'rf' in self.best_pipeline_.named_steps:
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

    def predict_future(self, X_test, forecast_horizon):
        """
        Predicción recursiva de valores futuros

        Parámetros:
        - X_test: Los datos de entrada que deben contener columnas de lag para la predicción.
        - forecast_horizon: Horizonte de predicción (número de pasos futuros a predecir).

        Devuelve:
        - Array de predicciones futuras
        """

        # Verificar que X_test es un DataFrame y que contiene columnas de lag
        if isinstance(X_test, pd.DataFrame):
            lag_columns = [col for col in X_test.columns if '_lag_' in col]
            if not lag_columns:
                raise ValueError("Input data must contain lag columns for prediction.")
            # Tomamos la última fila con las columnas de lag
            input_data = X_test[lag_columns].iloc[-1:].values
        else:
            input_data = np.array(X_test).reshape(1, -1)

        # Predicción recursiva
        predictions_scaled_list = []
        lower_bounds_scaled_list = []
        upper_bounds_scaled_list = []
        current_input = input_data.copy()

        if not self.best_pipeline_:
            raise ValueError("El modelo no ha sido ajustado. Por favor, ajuste el modelo antes de predecir.")

        selector = self.best_pipeline_.named_steps.get('selector')
        scaler = self.best_pipeline_.named_steps.get('scaler')
        rf_model = self.best_pipeline_.named_steps.get('rf')

        if not all([selector, scaler, rf_model]):
            raise RuntimeError("El pipeline no contiene los pasos esperados: 'selector', 'scaler' y 'rf'.")
        if not hasattr(rf_model, 'estimators_') or not rf_model.estimators_:
            raise RuntimeError("El modelo Random Forest en el pipeline no tiene estimadores.")

        for _ in range(forecast_horizon):
            # Predicción puntual
            point_pred_scaled = self.best_pipeline_.predict(current_input)[0]
            predictions_scaled_list.append(point_pred_scaled)

            # Predicciones de cada árbol
            current_input_selected = selector.transform(current_input)
            current_input_scaled_for_rf = scaler.transform(current_input_selected)

            individual_tree_preds_scaled = np.array([
                tree.predict(current_input_scaled_for_rf)[0] for tree in rf_model.estimators_
            ])

            # Calcular los límites inferior y superior
            lower_b_scaled = np.percentile(individual_tree_preds_scaled, 2.5)
            upper_b_scaled = np.percentile(individual_tree_preds_scaled, 97.5)

            lower_bounds_scaled_list.append(lower_b_scaled)
            upper_bounds_scaled_list.append(upper_b_scaled)

            # Actualizar el input deslizando las características a la izquierda y añadiendo la predicción al final
            current_input = np.roll(current_input, -1, axis=1)  # Asegurar axis=1 para array 2D
            current_input[0, -1] = point_pred_scaled  # Usar la predicción puntual para la recursión

        return np.array(predictions_scaled_list), np.array(lower_bounds_scaled_list), np.array(upper_bounds_scaled_list)

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


