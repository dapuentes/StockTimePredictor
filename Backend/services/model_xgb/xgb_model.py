
'''
# 📘 XGBoostModel (Regressor)
_Machine learning regression model for time series forecasting using XGBoost._

---

## 🔍 Overview

The `XGBoostModel` class encapsulates a robust regression workflow using `XGBRegressor`, tailored for time series prediction. It includes:

- Lag-feature engineering
- Feature scaling
- Hyperparameter optimization
- Evaluation metrics
- Recursive multi-step forecasting
- Visualization and model persistence

---

## 🧱 Class Architecture

```plaintext
        ┌──────────────┐
        │ Raw TimeSeries│
        └──────┬───────┘
               ▼
     ┌─────────────────────┐
     │ prepare_data()      │ <- feature_engineering + add_lags
     └─────────────────────┘
               ▼
     ┌─────────────────────┐
     │ scale_data()        │ <- MinMax or Robust
     └─────────────────────┘
               ▼
     ┌─────────────────────┐
     │ fit() or optimize_hyperparameters() 
     └─────────────────────┘
               ▼
     ┌─────────────┐
     │ predict()   │
     └─────────────┘
               ▼
     ┌─────────────────────┐
     │ evaluate()          │ → MAE, RMSE, R², etc
     └─────────────────────┘

''' 


from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os
import json


#services.model_xgb.xgb_model
class XGBoostModel:
    """
    A wrapper class for XGBoost Regression model with enhanced functionality
    """
    def __init__(self, 
                 objective='reg:squarederror',
                 n_estimators=100,
                 max_depth=3,
                 learning_rate=0.1,
                 subsample=1.0,
                 colsample_bytree=1.0,
                 gamma=0,
                 n_lags=5):
        """
        Initialize the XGBoost model with configurable parameters

        Parameters:
        - objective: Learning objective
        - n_estimators: Number of boosting rounds
        - max_depth: Maximum tree depth
        - learning_rate: Step size shrinkage to prevent overfitting
        - subsample: Subsample ratio of the training instances
        - colsample_bytree: Subsample ratio of columns when constructing each tree
        - gamma: Minimum loss reduction required to make a further partition
        """
        self.model = XGBRegressor(
            objective=objective,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            random_state=42
        )
        self.n_lags = n_lags
        self.feature_importances_ = None
        self.best_params_ = None
        self.feature_scaler = None
        self.target_scaler = None
        
    def prepare_data(self, data, target_col='Close'):
        """
        Preparar datos de series temporales con ingeniería de características

        Parámetros:
        - data: DataFrame de entrada que contiene los datos
        - target_col: Nombre de la columna de destino para la predicción (el valor predeterminado es 'Close')

        Devuelve:
        - DataFrame procesado con características
        """

        from Backend.utils import feature_engineering, add_lags

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
        Train the XGBoost model

        Parameters:
        - X_train: Training features
        - y_train: Training target values
        """
        self.model.fit(X_train, y_train)
        self.feature_importances_ = self.model.feature_importances_
        return self

    def predict(self, X_test):
        """
        Make predictions using the trained model

        Parameters:
        - X_test: Test features

        Returns:
        - Predictions
        """
        return self.model.predict(X_test)

    def optimize_hyperparameters(self, X_train, y_train, param_grid=None, n_iter=20, cv=3):
        """
        Optimize hyperparameters using RandomizedSearchCV

        Parameters:
        - X_train: Training features
        - y_train: Training target values
        - param_grid: Dictionary of hyperparameters to search
        - n_iter: Number of parameter settings sampled
        - cv: Number of cross-validation folds

        Returns:
        - Optimized model
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'gamma': [0, 0.1, 1]
            }

        base_model = XGBRegressor(objective='reg:squarederror', random_state=42)
        
        # Use TimeSeriesSplit to prevent temporal data leakage
        tscv = TimeSeriesSplit(n_splits=cv)
        
        search = RandomizedSearchCV(
            base_model, 
            param_grid, 
            n_iter=n_iter,
            scoring='neg_mean_squared_error', 
            n_jobs=-1,
            cv=tscv, 
            random_state=42,
            verbose=1
        )
        
        search.fit(X_train, y_train)
        
        # Update the model with best parameters
        self.model = search.best_estimator_
        self.best_params_ = search.best_params_
        self.feature_importances_ = self.model.feature_importances_
        
        print('Best Mean Squared Error: %.3f' % -search.best_score_)
        print('Best Config: %s' % search.best_params_)
        
        return self

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance

        Parameters:
        - X_test: Test features
        - y_test: True target values

        Returns:
        - Dictionary of performance metrics
        """
        y_pred = self.predict(X_test)
        y_pred = self.target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten() if self.target_scaler else y_pred
        y_test = self.target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten() if self.target_scaler else y_test
        return {
            'mse': float(mean_squared_error(y_test, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'r2': float(r2_score(y_test, y_pred))
        }

    def save(self, model_path="models/model.joblib"):
        """
        Save the trained model to a file

        Parameters:
        - model_path: Path to save the model
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the entire model instance, not just the XGBRegressor
        model_data = {
            'xgb_model': self.model,
            'n_lags': self.n_lags,
            'feature_importances_': self.feature_importances_,
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'best_params_': self.best_params_
        }
        
        joblib.dump(model_data, model_path)

        # Save metadata
        metadata = {
            'best_params': self.best_params_,
            'feature_importances': self.feature_importances_.tolist() if self.feature_importances_ is not None else None,
            'metrics': self.metrics(),
            'timestamp': pd.Timestamp.now().isoformat()
        }

        # Generate metadata file path
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
        Load a trained model from a file

        Parameters:
        - filepath: Path to the saved model

        Returns:
        - Loaded model
        """
        try:
            # Load the saved model data
            model_data = joblib.load(filepath)
            
            # Check if we're loading the old format (just the XGBRegressor)
            if not isinstance(model_data, dict):
                # Create a new instance
                instance = cls()
                # Set the XGBRegressor model
                instance.model = model_data
                # Set default values for other attributes
                instance.n_lags = 5  # Default value
                return instance
            
            # For the new format (dictionary with all attributes)
            instance = cls()
            instance.model = model_data['xgb_model']
            instance.n_lags = model_data['n_lags']
            instance.feature_importances_ = model_data['feature_importances_']
            instance.best_params_ = model_data['best_params_']
            instance.feature_scaler = model_data.get('feature_scaler', None)
            instance.target_scaler = model_data.get('target_scaler', None)
            # new add 2 last lines
            
            return instance
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def metrics(self):
        """
        Get model performance metrics

        Returns:
        - Dictionary of performance metrics
        """
        return {
            'mse': 'mean_squared_error',
            'rmse': 'root_mean_squared_error',
            'mae': 'mean_absolute_error',
            'r2': 'r2_score'
        }
    
    #  =========== NEW ADD =================#
    def predict_future(self, X_test, forecast_horizon):
        """
        Predicción recursiva de valores futuros

        Parámetros:
        - X_test: Los datos de entrada (DataFrame o array) con todas las features usadas en entrenamiento.
        - forecast_horizon: Horizonte de predicción (número de pasos futuros a predecir).

        Devuelve:
        - Array de predicciones futuras
        """

        if isinstance(X_test, pd.DataFrame):
            # Use all feature columns, not just lags
            feature_cols = X_test.columns.tolist()
            lag_columns = [col for col in feature_cols if '_lag_' in col]
            non_lag_columns = [col for col in feature_cols if '_lag_' not in col]
            
            if not lag_columns:
                raise ValueError("Input data must contain lag columns for recursive prediction.")
            
            # Start with the last row of all features
            current_features = X_test.iloc[-1:].copy()
        else:
            # Fallback for array input — use all columns
            current_features = pd.DataFrame(X_test.reshape(1, -1)) if X_test.ndim == 1 else pd.DataFrame(X_test[-1:])
            lag_columns = list(range(current_features.shape[1]))
            non_lag_columns = []

        predictions = []

        for _ in range(forecast_horizon):
            if isinstance(current_features, pd.DataFrame) and len(current_features.columns) > 0:
                pred = self.model.predict(current_features.values)[0]
            else:
                pred = self.model.predict(current_features)[0]
            predictions.append(pred)

            # Update lag features: shift lags by 1 and insert new prediction as lag_1
            if isinstance(X_test, pd.DataFrame):
                new_row = current_features.copy()
                # Sort lag columns by lag number descending to shift correctly
                sorted_lags = sorted(lag_columns, key=lambda c: int(c.split('_lag_')[-1]), reverse=True)
                for i, col in enumerate(sorted_lags):
                    if i + 1 < len(sorted_lags):
                        # Shift: lag_N = previous lag_(N-1)
                        new_row[col] = current_features[sorted_lags[i + 1]].values[0]
                    else:
                        # lag_1 = new prediction
                        new_row[col] = pred
                # Non-lag features (technical indicators) are carried forward as-is
                # This is an approximation; a full recomputation would require OHLCV data
                current_features = new_row
            else:
                current_input = current_features.values.copy() if hasattr(current_features, 'values') else current_features.copy()
                current_input = np.roll(current_input, -1)
                current_input[0, -1] = pred
                current_features = pd.DataFrame(current_input)

        return np.array(predictions)
    
    def plot_results(self, y_true, y_pred, title="Model Predictions"):
        """
        Graficar los resultados de la predicción

        Parámetros:
        - y_true: Valores verdaderos
        - y_pred: Valores predichos
        - title: Título del gráfico
        """
        #if not self.plotting:
        #    return
        
        from Backend.utils.visualizations import plot_predictions
        plot_predictions(y_true, y_pred, title=title)

    def plot_forecast(self, historical_data, forecast_values, target_col='Close'):
        """
        Graficar el pronóstico comparado con los datos históricos

        Parámetros:
        - historical_data: Datos históricos
        - forecast_values: Valores pronosticados
        - target_col: Nombre de la columna objetivo
        """
        #if not self.plotting:
        #    return

        from Backend.utils.visualizations import plot_forecast
        plot_forecast(historical_data, forecast_values, target_col=target_col)

         #  =========== NEW ADD =================#

# Utility function for backward compatibility
def train_xgb_model(data, n_lags=10, target_col='Close', train_size=0.8, save_model_path=None):
    """
    Entrenar un modelo de XGBoost para datos de series temporales

    Parámetros:
    - data: DataFrame con los datos
    - n_lags: Número de características de rezago a crear
    - target_col: Nombre de la columna objetivo para la predicción (el valor predeterminado es 'Close')
    - train_size: Proporción del conjunto de datos a usar para el entrenamiento (el valor predeterminado es 0.8)
    - save_model_path: Ruta para guardar el modelo entrenado (el valor predeterminado es None, no se guarda)

    Devuelve:
    - Modelo entrenado con sus métricas de rendimiento
    """

    from Backend.utils import scale_data, split_data

    model = XGBoostModel(n_lags=n_lags)
    # Prepara los datos
    data = model.prepare_data(data, target_col=target_col)

    # Divide los datos en conjuntos de entrenamiento y prueba

    X_train, X_test, y_train, y_test = split_data(data, train_size=train_size, shuffle=False, random_state=42)

    feature_names = X_train.columns.tolist()
    print(f"Feature names: {feature_names}")

    # Escala los datos
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, model.feature_scaler, model.target_scaler = scale_data(X_train, X_test, y_train, y_test)

    model.feature_scaler = model.feature_scaler
    model.target_scaler = model.target_scaler


    # Modelo optimizado
    model.optimize_hyperparameters(X_train_scaled, y_train_scaled.ravel(), n_iter=20, cv=3) #ravel para convertir a 1D array

    print('Optimized hyperparameters:', model.best_params_)

    metrics = model.evaluate(X_test_scaled, y_test_scaled)
    print("Evaluation Metrics:", metrics)

    if save_model_path is not None:
        model.save(save_model_path)
        print(f"Model saved to {save_model_path}")
    
    return model


