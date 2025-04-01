from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import joblib
import numpy as np
import pandas as pd
import os
import json

class TimeSeriesRandomForestModel:
    """
    Una clase contenedora para el modelo de regresión de bosque aleatorio, diseñada específicamente para datos financieros
    de series temporales, con funcionalidad mejorada mediante el paquete utils.
    """

    def __init__(self,
                 n_estimators=100,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features='log2',
                 n_lags=10,
                 plotting=False,         
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
        - plotting: Si se deben graficar los resultados (el valor predeterminado es False)
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
        self.plotting = plotting

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
        self.model.fit(X_train, y_train)
        self.feature_importances_ = self.model.feature_importances_
        return self

    def predict(self, X):
        """
        Realizar predicciones usando el modelo entrenado.

        Parámetros:
        - X: Datos de entrada para la predicción

        Devuelve:
        - Predicciones realizadas por el modelo.
        """
        return self.model.predict(X)
    
    def optimize_hyperparameters(self, X_train, y_train, param_grid=None, cv=3):
        """
        Optimizar hiperparámetros utilizando GridSearchCV

        Parámetros:
        - X_train: Características de entrenamiento
        - y_train: Valores objetivo de entrenamiento
        - param_grid: Diccionario de hiperparámetros a buscar
        - cv: Número de pliegues de validación cruzada

        Devuelve:
        - Modelo optimizado
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200], 
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [10, 20], # se prueba con valores más altos para evitar el sobreajuste
                'max_features': ['sqrt', 'log2', 0.5] 
            }
        
        tscv = TimeSeriesSplit(n_splits=cv)
        
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train) # Ajustar el modelo con los mejores hiperparámetros
        
        # Guardar los mejores hiperparámetros y el modelo
        self.best_params_ = grid_search.best_params_ 
        self.model = grid_search.best_estimator_ 
        
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

        y_pred = self.model.predict(X_test)
        # Desescalar la predicción, asumiendo que target_scaler ya está definido
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
        predictions = []
        current_input = input_data.copy()

        for _ in range(forecast_horizon):
            pred = self.model.predict(current_input)[0]
            predictions.append(pred)

            # Actualizar el input deslizando las características a la izquierda y añadiendo la predicción al final
            current_input = np.roll(current_input, -1)
            current_input[0, -1] = pred

        return np.array(predictions)
    
    def plot_results(self, y_true, y_pred, title="Model Predictions"):
        """
        Graficar los resultados de la predicción

        Parámetros:
        - y_true: Valores verdaderos
        - y_pred: Valores predichos
        - title: Título del gráfico
        """
        if not self.plotting:
            return
        
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
        if not self.plotting:
            return

        from utils.visualizations import plot_forecast
        plot_forecast(historical_data, forecast_values, target_col=target_col)

    def save_model(self, model_path="models/model.joblib"):
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
            'timestamp': pd.Timestamp.now().isoformat()
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
    
def train_ts_model(data, n_lags=10, target_col='Close', train_size=0.8, save_model_path=None):
    """
    Entrenar un modelo de Random Forest para datos de series temporales

    Parámetros:
    - data: DataFrame con los datos
    - n_lags: Número de características de rezago a crear
    - target_col: Nombre de la columna objetivo para la predicción (el valor predeterminado es 'Close')
    - train_size: Proporción del conjunto de datos a usar para el entrenamiento (el valor predeterminado es 0.8)
    - save_model_path: Ruta para guardar el modelo entrenado (el valor predeterminado es None, no se guarda)

    Devuelve:
    - Modelo entrenado con sus métricas de rendimiento
    """
    
    from utils.preprocessing import scale_data

    model = TimeSeriesRandomForestModel(n_lags=n_lags)

    # Preparar los datos
    processed_data = model.prepare_data(data, target_col=target_col)

    train_size = int(len(processed_data) * train_size)
    train_data = processed_data.iloc[:train_size]
    test_data = processed_data.iloc[train_size:]

    # Separar características y objetivo
    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col].values.reshape(-1, 1)
    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col].values.reshape(-1, 1)

    feature_names = X_train.columns.tolist()
    print(f"Feature names: {feature_names}")
    
    # Escalar los datos
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, feature_scaler, target_scaler = scale_data(
        X_train, X_test, y_train, y_test
    )

    model.feature_scaler = feature_scaler
    model.target_scaler = target_scaler

    # Modelo optimizado
    model.optimize_hyperparameters(X_train_scaled,
                                   y_train_scaled.ravel()
    ) # ravel es necesario para convertir a 1D
    print(f"Best parameters: {model.best_params_}")

    model.evaluate(X_test_scaled, y_test)
    print(f"Model metrics: {model.metrics}")

    if save_model_path is not None:
        model.save_model(save_model_path)
        print(f"Model saved to {save_model_path}")
    
    return model

def forecast_future_prices(model, data, forecast_horizon=10, target_col='Close'):
    """
    Pronosticar precios futuros utilizando el modelo entrenado

    Parámetros:
    - model: Modelo entrenado
    - data: DataFrame con los datos más recientes
    - forecast_horizon: Horizonte de pronóstico en días (el valor predeterminado es 10)
    - target_col: Nombre de la columna objetivo para la predicción (el valor predeterminado es 'Close')

    Devuelve:
    - Array de precios futuros pronosticados
    """
    
    # Preparar los datos más recientes
    processed_data = model.prepare_data(data, target_col=target_col)

    # Obtener la última fila de datos para la predicción
    last_data = processed_data.iloc[-1:]
    X_last = last_data.drop(columns=[target_col])

    if model.feature_scaler:
        X_last_scaled = model.feature_scaler.transform(X_last)
    else:
        X_last_scaled = X_last.values

    # Pronosticar precios futuros
    forecast_scaled = model.predict_future(X_last_scaled, forecast_horizon)

    if model.target_scaler:
        forecast = model.target_scaler.inverse_transform(
            forecast_scaled.reshape(-1, 1)).ravel()
    else:
        forecast = forecast_scaled

    model.plot_forecast(
        data,
        forecast,
        target_col=target_col
    )

    return forecast
