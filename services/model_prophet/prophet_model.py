
from prophet import Prophet
import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
import warnings

class ProphetModel:
    """
    A wrapper class for Facebook Prophet time series model with enhanced functionality
    """
    def __init__(self, 
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                holidays_prior_scale=10.0,
                seasonality_mode='additive',
                changepoint_range=0.8,
                yearly_seasonality='auto',
                weekly_seasonality='auto',
                daily_seasonality='auto',
                n_lags=5):
        """
        Initialize the Prophet model with configurable parameters
        """
        self.model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            seasonality_mode=seasonality_mode,
            changepoint_range=changepoint_range,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality
        )
        self.n_lags = n_lags
        self.best_params_ = None

        self.seasonality_added = False
        self.has_fitted = False
        
    def prepare_data(self, data, target_col='Close', regressor_cols=None):
        """
        Preparar datos de series temporales para Prophet

        Parámetros:
        - data: DataFrame de entrada que contiene los datos
        - target_col: Nombre de la columna de destino para la predicción (el valor predeterminado es 'Close')
        - regressor_cols: Lista de columnas para usar como regresores (None = no usar regresores adicionales)

        Devuelve:
        - DataFrame procesado con formato adecuado para Prophet (ds, y)
        """
        # Asegurarse de que el índice es de tipo fecha
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("El índice del DataFrame debe ser de tipo DatetimeIndex")
        
        # IMPORTANTE: Prophet no acepta fechas con zona horaria
        # Convertir el índice a fechas sin zona horaria
        data_copy = data.copy()
        
        # Prophet requiere columnas específicas: 'ds' para fechas y 'y' para valores
        prophet_data = pd.DataFrame({
            'ds': data_copy.index.tz_localize(None),  # Eliminar zona horaria
            'y': data_copy[target_col]
        })
        
        # Agregar variables exógenas como regressores adicionales
        if regressor_cols is not None:
            for col in regressor_cols:
                if col in data_copy.columns and col != target_col:
                    prophet_data[col] = data_copy[col]
                    # Agregar como regresor al modelo
                    try:
                        self.model.add_regressor(col)
                    except:
                        pass
        
        # Para análisis adicional, opcionalmente crear características de rezagos
        if self.n_lags > 0:
            for i in range(1, self.n_lags + 1):
                lag_name = f'{target_col}_lag_{i}'
                prophet_data[lag_name] = prophet_data['y'].shift(i)
                try:
                    self.model.add_regressor(lag_name)
                except:
                    pass
            
            # Eliminar filas con NaN (debido a los rezagos)
            prophet_data = prophet_data.dropna()
        
        return prophet_data

    def add_seasonality(self):
        """
        Añadir componentes de estacionalidad adicionales al modelo
        """
        if not self.seasonality_added:
            # Añadir estacionalidad mensual (útil para datos financieros)
            self.model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            # Añadir estacionalidad trimestral (útil para datos financieros)
            self.model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)
            self.seasonality_added = True
        return self

    def fit(self, X_train, y_train=None):
        """
        Entrenar el modelo Prophet

        Parámetros:
        - X_train: DataFrame con formato Prophet (ds, y)
        - y_train: No utilizado para Prophet (incluido para compatibilidad)
        """
        # Prophet requiere un DataFrame específico en lugar de X_train, y_train separados
        if not isinstance(X_train, pd.DataFrame) or 'ds' not in X_train.columns or 'y' not in X_train.columns:
            raise ValueError("X_train debe ser un DataFrame con columnas 'ds' y 'y'")
        
        # Verificar y corregir zona horaria si es necesario
        if hasattr(X_train['ds'], 'dt') and hasattr(X_train['ds'].dt, 'tz'):
            if X_train['ds'].dt.tz is not None:
                warnings.warn("La columna ds tiene zona horaria. Eliminando zona horaria.")
                X_train = X_train.copy()
                X_train['ds'] = X_train['ds'].dt.tz_localize(None)
        
        # Agregar estacionalidad personalizada
        self.add_seasonality()
        
        # Entrenar el modelo
        self.model.fit(X_train)
        self.has_fitted = True
        return self

    def predict(self, X_test):
        """
        Realizar predicciones utilizando el modelo entrenado

        Parámetros:
        - X_test: DataFrame con fechas para predicción en formato Prophet ('ds')

        Devuelve:
        - Predicciones
        """
        if not self.has_fitted:
            raise ValueError("El modelo debe ser entrenado primero con fit()")
        
        # Si X_test es un DataFrame de Prophet, usamos las fechas
        if isinstance(X_test, pd.DataFrame) and 'ds' in X_test.columns:
            future = X_test[['ds']].copy()
            
            # Verificar y corregir zona horaria si es necesario
            if hasattr(future['ds'], 'dt') and hasattr(future['ds'].dt, 'tz'):
                if future['ds'].dt.tz is not None:
                    future['ds'] = future['ds'].dt.tz_localize(None)
            
            # Agregar regresores si están disponibles
            regressors = [col for col in X_test.columns if col not in ['ds', 'y']]
            for col in regressors:
                if col in self.model.extra_regressors:
                    future[col] = X_test[col]
        else:
            # Si es otro formato, creamos un DataFrame futuro
            if self.has_fitted:
                future = self.model.make_future_dataframe(periods=len(X_test))
            else:
                raise ValueError("No se pueden generar fechas futuras sin entrenar el modelo primero")
        
        # Realizar predicción
        forecast = self.model.predict(future)
        
        # Devolver solo los valores predichos
        return forecast['yhat'].values

    def optimize_hyperparameters(self, X_train, y_train=None, param_grid=None, n_iter=10, cv=3):
        """
        Optimizar hiperparámetros usando validación cruzada específica de Prophet
        """
        if param_grid is None:
            param_grid = {
                'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5],
                'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
                'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
                'seasonality_mode': ['additive', 'multiplicative']
            }
        
        # Verificar y corregir zona horaria
        if 'ds' in X_train.columns:
            if hasattr(X_train['ds'], 'dt') and hasattr(X_train['ds'].dt, 'tz'):
                if X_train['ds'].dt.tz is not None:
                    warnings.warn("La columna ds tiene zona horaria. Eliminando zona horaria.")
                    X_train = X_train.copy()
                    X_train['ds'] = X_train['ds'].dt.tz_localize(None)
        
        try:
            # Validación cruzada específica de Prophet
            from prophet.diagnostics import cross_validation, performance_metrics
            
            # Configurar parámetros para validación cruzada
            initial = max(int(len(X_train) * 0.5), 365)  # Al menos 365 días para la estacionalidad anual
            period = int(len(X_train) * 0.1)
            horizon = int(len(X_train) * 0.2)
            
            # Convertir a días para Prophet
            if isinstance(X_train['ds'], pd.Series) and pd.api.types.is_datetime64_any_dtype(X_train['ds']):
                dates = pd.Series(X_train['ds']).sort_values()
                avg_diff = (dates.iloc[-1] - dates.iloc[0]) / (len(dates) - 1)
                
                if avg_diff < pd.Timedelta(days=2):
                    initial_str = f'{max(initial, 5)} days'
                    period_str = f'{max(period, 1)} days'
                    horizon_str = f'{max(horizon, 1)} days'
                else:
                    initial_str = f'{max(initial, 5)} days'
                    period_str = f'{max(period, 1)} days'
                    horizon_str = f'{max(horizon, 1)} days'
            else:
                initial_str = '365 days'  # Usar al menos 365 días para la estacionalidad anual
                period_str = '30 days'
                horizon_str = '30 days'
            
            # Almacenar los resultados
            results = []
            
            # Probar combinaciones de hiperparámetros
            import random
            param_combinations = []
            for _ in range(min(n_iter, len(param_grid['changepoint_prior_scale']) * 
                            len(param_grid['seasonality_prior_scale']) * 
                            len(param_grid['holidays_prior_scale']) * 
                            len(param_grid['seasonality_mode']))):
                params = {
                    'changepoint_prior_scale': random.choice(param_grid['changepoint_prior_scale']),
                    'seasonality_prior_scale': random.choice(param_grid['seasonality_prior_scale']),
                    'holidays_prior_scale': random.choice(param_grid['holidays_prior_scale']),
                    'seasonality_mode': random.choice(param_grid['seasonality_mode'])
                }
                if params not in param_combinations:
                    param_combinations.append(params)
            
            print(f"Testing {len(param_combinations)} parameter combinations...")
            
            for params in param_combinations:
                try:
                    # Crear modelo con estos parámetros
                    m = Prophet(
                        changepoint_prior_scale=params['changepoint_prior_scale'],
                        seasonality_prior_scale=params['seasonality_prior_scale'],
                        holidays_prior_scale=params['holidays_prior_scale'],
                        seasonality_mode=params['seasonality_mode']
                    )
                    
                    # Agregar regresores si existen
                    regressors = [col for col in X_train.columns if col not in ['ds', 'y']]
                    for reg in regressors:
                        m.add_regressor(reg)
                    
                    # Entrenar
                    m.fit(X_train)
                    
                    # Validación cruzada - usar None en lugar de 'processes'
                    df_cv = cross_validation(m, initial=initial_str, 
                                            period=period_str, 
                                            horizon=horizon_str,
                                            parallel=None)  # Cambiar a None para evitar errores de multiprocesamiento
                    
                    # Calcular métricas
                    df_metrics = performance_metrics(df_cv)
                    
                    # Almacenar resultados
                    results.append((params, df_metrics['rmse'].mean()))
                    print(f"Parameters: {params}, RMSE: {df_metrics['rmse'].mean()}")
                    
                except Exception as e:
                    print(f"Error with parameters {params}: {e}")
                    continue
            
            # Resto del código sin cambios...
        except Exception as e:
            print(f"Error during hyperparameter optimization: {e}")
            print("Using default parameters instead.")
            # Establecer parámetros por defecto en caso de error
            self.best_params_ = {
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10.0,
                'holidays_prior_scale': 10.0,
                'seasonality_mode': 'additive'
            }
        
        return self

    def evaluate(self, X_test, y_test=None):
        """
        Evaluar el rendimiento del modelo

        Parámetros:
        - X_test: DataFrame con formato Prophet para pruebas
        - y_test: Valores reales (no usado, para compatibilidad)

        Devuelve:
        - Diccionario de métricas de rendimiento
        """
        if not self.has_fitted:
            raise ValueError("El modelo debe ser entrenado primero con fit()")
                
        # Para Prophet, y_test debería estar en X_test['y']
        if 'y' not in X_test.columns:
            raise ValueError("X_test debe contener una columna 'y' con los valores reales")
        
        # Hacer predicciones
        # Verificar y corregir zona horaria si es necesario
        X_test_copy = X_test.copy()
        if hasattr(X_test_copy['ds'], 'dt') and hasattr(X_test_copy['ds'].dt, 'tz'):
            if X_test_copy['ds'].dt.tz is not None:
                X_test_copy['ds'] = X_test_copy['ds'].dt.tz_localize(None)
        
        # Obtener solo las columnas necesarias para la predicción
        # Incluir todos los regresores que el modelo espera
        forecast_columns = ['ds']
        if hasattr(self.model, 'extra_regressors'):
            for regressor in self.model.extra_regressors:
                if regressor in X_test_copy.columns:
                    forecast_columns.append(regressor)
        
        # Verificar si faltan algunos regresores
        missing_regressors = [reg for reg in self.model.extra_regressors 
                            if reg not in X_test_copy.columns]
        if missing_regressors:
            print(f"Warning: Missing regressors in test data: {missing_regressors}")
            # Crear columnas para los regresores faltantes con valores 0
            for reg in missing_regressors:
                X_test_copy[reg] = 0
                forecast_columns.append(reg)
        
        # Realizar la predicción con todas las columnas necesarias
        forecast = self.model.predict(X_test_copy[forecast_columns])
        y_pred = forecast['yhat'].values
        y_true = X_test_copy['y'].values
        
        # Ya no aplicamos transformación inversa puesto que no hay escalado
        
        # Calcular métricas
        return {
            'mse': float(mean_squared_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2': float(r2_score(y_true, y_pred))
        }

    def save(self, model_path="models/prophet_model.joblib"):
        """
        Guardar el modelo entrenado en un archivo

        Parámetros:
        - model_path: Ruta para guardar el modelo
        """
        if not self.has_fitted:
            raise ValueError("No se puede guardar un modelo que no ha sido entrenado")
            
        # Asegurar que el directorio existe
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Guardar la instancia completa del modelo
        model_data = {
            'prophet_model': self.model,
            'n_lags': self.n_lags,
            'best_params_': self.best_params_,
            'seasonality_added': self.seasonality_added,
            'has_fitted': self.has_fitted
        }
        
        joblib.dump(model_data, model_path)

        # Guardar metadatos
        metadata = {
            'best_params': self.best_params_,
            'metrics': self.metrics(),
            'timestamp': pd.Timestamp.now().isoformat()
        }

        # Generar ruta del archivo de metadatos
        if model_path.endswith('.joblib'):
            metadata_file = model_path.replace('.joblib', '_metadata.json')
        elif model_path.endswith('.pkl'):
            metadata_file = model_path.replace('.pkl', '_metadata.json')
        else:
            metadata_file = model_path + '_metadata.json'

        with open(metadata_file, 'w') as f:
            json.dump({k: str(v) if not isinstance(v, (int, float, list)) else v 
                    for k, v in metadata.items()}, f, indent=4)

    @classmethod
    def load(cls, filepath):
        """
        Cargar un modelo entrenado desde un archivo

        Parámetros:
        - filepath: Ruta al modelo guardado

        Devuelve:
        - Modelo cargado
        """
        try:
            # Cargar los datos del modelo guardado
            model_data = joblib.load(filepath)
            
            # Crear una nueva instancia
            instance = cls()
            
            # Configurar atributos
            instance.model = model_data['prophet_model']
            instance.n_lags = model_data['n_lags']
            instance.best_params_ = model_data['best_params_']
            instance.seasonality_added = model_data.get('seasonality_added', False)
            instance.has_fitted = model_data.get('has_fitted', True)  # Asumir que está entrenado
            
            return instance
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def metrics(self):
        """
        Obtener métricas de rendimiento del modelo

        Devuelve:
        - Diccionario de métricas de rendimiento
        """
        return {
            'mse': 'mean_squared_error',
            'rmse': 'root_mean_squared_error',
            'mae': 'mean_absolute_error',
            'r2': 'r2_score'
        }
    def predict_future(self, data, forecast_horizon):
        """
        Predicción de valores futuros usando Prophet

        Parámetros:
        - data: DataFrame con los datos históricos
        - forecast_horizon: Horizonte de predicción (número de pasos futuros a predecir)

        Devuelve:
        - DataFrame con las predicciones futuras
        """
        if not self.has_fitted:
            raise ValueError("El modelo debe ser entrenado primero con fit()")
        
        print("[DEBUG] Creating future DataFrame for prediction...")
        # Crear DataFrame futuro para Prophet
        future = self.model.make_future_dataframe(periods=forecast_horizon)
        print(f"[DEBUG] Future DataFrame created with {len(future)} rows.")

        # Si hay regresores, necesitamos incluirlos para las fechas futuras
        if hasattr(self.model, 'extra_regressors'):
            regressors = list(self.model.extra_regressors.keys())
            print(f"[DEBUG] Extra regressors detected: {regressors}")
            
            # Para cada regresor, asegurar que esté en el dataframe futuro
            for reg in regressors:
                print(f"[DEBUG] Processing regressor: {reg}")
                # Para los regresores de rezago
                if '_lag_' in reg:
                    if isinstance(data, pd.DataFrame) and reg in data.columns:
                        print(f"[DEBUG] Found lagged regressor {reg} in data.")
                        # Obtener el último valor disponible para cada rezago
                        last_values = data[reg].iloc[-forecast_horizon:].values
                        print(f"[DEBUG] Last values for {reg}: {last_values}")
                        if len(last_values) < forecast_horizon:
                            # Rellenar con el último valor si no hay suficientes
                            last_values = np.pad(last_values, (0, forecast_horizon - len(last_values)), 
                                            'constant', constant_values=last_values[-1] if len(last_values) > 0 else 0)
                            print(f"[DEBUG] Padded values for {reg}: {last_values}")
                        # Asignar valores a las fechas futuras
                        for i, val in enumerate(last_values):
                            future.loc[len(future) - forecast_horizon + i, reg] = val
                    else:
                        print(f"[DEBUG] Lagged regressor {reg} not found in data. Filling with zeros.")
                        # Si no hay datos para este regresor, usar ceros
                        future[reg] = 0
                # Para regresores normales (no de rezago)
                else:
                    # Si el regresor está en los datos originales, usar el último valor conocido
                    if isinstance(data, pd.DataFrame) and reg in data.columns:
                        last_value = data[reg].iloc[-1]
                        print(f"[DEBUG] Using last known value for regressor {reg}: {last_value}")
                        future[reg] = last_value
                    else:
                        print(f"[DEBUG] Regressor {reg} not found in data. Filling with zeros.")
                        # Si no hay datos para este regresor, usar ceros
                        future[reg] = 0
        
        print("[DEBUG] Future DataFrame ready for prediction.")
        # Realizar predicción
        forecast = self.model.predict(future)
        print("[DEBUG] Prediction completed.")

        # Ya no aplicamos transformación inversa puesto que no hay escalado
        
        # Devolver solo las predicciones futuras
        future_predictions = forecast.iloc[-forecast_horizon:]
        print(f"[DEBUG] Returning future predictions with {len(future_predictions)} rows.")
        return future_predictions
    
    def plot_results(self, y_true, y_pred, title="Model Predictions"):
        """
        Graficar los resultados de la predicción

        Parámetros:
        - y_true: Valores verdaderos
        - y_pred: Valores predichos
        - title: Título del gráfico
        """
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, 'b-', label='Actual')
        plt.plot(y_pred, 'r--', label='Predicted')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    
    def plot_forecast(self, forecast=None, uncertainty=True, xlabel='Date', ylabel='Value', figsize=(12, 6)):
        """
        Plot the forecast generated by the Prophet model

        Parameters:
        - forecast: Prophet forecast DataFrame (if None, uses the model's most recent forecast)
        - uncertainty: Whether to plot uncertainty intervals (default: True)
        - xlabel: Label for x-axis (default: 'Date')
        - ylabel: Label for y-axis (default: 'Value')
        - figsize: Figure size as tuple (width, height) (default: (12, 6))

        Returns:
        - The matplotlib figure object
        """
        if not self.has_fitted:
            raise ValueError("The model must be trained first with fit()")
            
        try:
            # If no forecast is provided, use the model to predict on its history
            if forecast is None:
                forecast = self.model.predict(self.model.history)
            
            # Use Prophet's built-in plot function
            fig = self.model.plot(forecast, uncertainty=uncertainty)
            
            # Customize the plot
            ax = fig.gca()
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title('Prophet Model Forecast')
            ax.grid(True)
            
            # Adjust figure size
            fig.set_size_inches(figsize)
            
            plt.tight_layout()
            return fig
        
        except Exception as e:
            print(f"Error plotting forecast: {e}")
            return None


    def plot_components(self):
        """
        Graficar los componentes del modelo Prophet (tendencia, estacionalidad)
        """
        if not self.has_fitted:
            raise ValueError("El modelo debe ser entrenado primero con fit()")
            
        try:
            fig = self.model.plot_components(self.model.predict(self.model.history))
            plt.tight_layout()
            plt.show()
            return fig
        except Exception as e:
            print(f"Error plotting components: {e}")


# Función de utilidad para compatibilidad con el flujo de trabajo existente
def train_prophet_model(data, n_lags=10, target_col='Close', regressor_cols=None, train_size=0.8, save_model_path=None):
    """
    Entrenar un modelo Prophet para datos de series temporales financieras

    Parámetros:
    - data: DataFrame con los datos
    - n_lags: Número de características de rezago a crear
    - target_col: Nombre de la columna objetivo para la predicción (el valor predeterminado es 'Close')
    - regressor_cols: Lista de columnas para usar como regresores (None = no usar regresores adicionales)
    - train_size: Proporción del conjunto de datos a usar para el entrenamiento (el valor predeterminado es 0.8)
    - save_model_path: Ruta para guardar el modelo entrenado (el valor predeterminado es None, no se guarda)

    Devuelve:
    - Modelo entrenado con sus métricas de rendimiento
    """
    # Inicializar modelo
    model = ProphetModel(n_lags=n_lags)
    
    # Preparar los datos para Prophet
    prophet_data = model.prepare_data(data, target_col=target_col, regressor_cols=regressor_cols)
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    train_size_idx = int(len(prophet_data) * train_size)
    train_data = prophet_data.iloc[:train_size_idx].copy()
    test_data = prophet_data.iloc[train_size_idx:].copy()
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Testing data shape: {test_data.shape}")
    
    
    # Optimizar hiperparámetros
    model.optimize_hyperparameters(train_data, n_iter=10, cv=3)
    
    print('Optimized hyperparameters:', model.best_params_)
    
    # Entrenar con los mejores parámetros
    model.fit(train_data)
    
    # Evaluar el modelo
    metrics = model.evaluate(test_data)
    print("Evaluation Metrics:", metrics)
    
    # Plotear componentes
    model.plot_components()
    
    # Guardar el modelo si se proporciona una ruta
    if save_model_path is not None:
        model.save(save_model_path)
        print(f"Model saved to {save_model_path}")
    
    return model