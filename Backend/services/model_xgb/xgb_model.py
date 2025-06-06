# Backend/services/model_xgb/xgb_model.py

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import joblib
import pandas as pd
import numpy as np
import os
import json
from typing import Optional, List, Dict, Tuple, Any

# Assuming XGBoostPreprocessor is correctly defined in utils.preprocessing
from utils.preprocessing import XGBoostPreprocessor # Make sure this path is correct for your structure
from utils.evaluation import evaluate_regression 


class TimeSeriesXGBoostModel:
    """
    A machine learning model for time series forecasting based on XGBoost.
    This class encapsulates an XGBoost-based model, including preprocessing steps
    delegated to an XGBoostPreprocessor, hyperparameter optimization, training,
    prediction, and evaluation.
    """

    def __init__(self,
                 preprocessor: Optional[XGBoostPreprocessor] = None, # Allows injecting a pre-configured/loaded preprocessor
                 n_estimators: int = 100,
                 max_depth: int = 3,
                 learning_rate: float = 0.1,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 objective: str = 'reg:squarederror',
                 n_lags: int = 10,  # Used if preprocessor is None or doesn't have n_lags
                 random_state: int = 42):
        """
        Initializes the TimeSeriesXGBoostModel.
        """
        self.model_base = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective=objective,
            random_state=random_state,
            n_jobs=-1  # Use all available cores
        )

        self.best_pipeline_: Optional[Pipeline] = None
        self.best_params_: Optional[Dict[str, Any]] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.selected_feature_names_: Optional[List[str]] = None # Features selected by SelectFromModel
        self.metrics: Optional[Dict[str, float]] = None
        
        # Preprocessor, scalers, and feature names handling
        if preprocessor is not None:
            self.preprocessor = preprocessor
            # Try to get n_lags from the provided preprocessor, otherwise use the default
            self.n_lags = getattr(self.preprocessor, 'n_lags', n_lags) 
            # Scalers and feature_names will be derived from this preprocessor.
            # If it's already fitted (e.g., when loading a model), these should be available.
            self.feature_scaler = getattr(self.preprocessor, 'feature_scaler', None)
            self.target_scaler = getattr(self.preprocessor, 'target_scaler', None)
            # feature_names are the names of features *after* preprocessor.prepare_data
            self.feature_names: List[str] = list(getattr(self.preprocessor, 'feature_names', []))
        else:
            self.n_lags = n_lags
            self.preprocessor = XGBoostPreprocessor(n_lags=self.n_lags)
            # These will be set by the training script after fitting
            self.feature_scaler = None 
            self.target_scaler = None 
            # This will be set after model.prepare_data() is called in the training script
            self.feature_names: List[str] = [] 


    def prepare_data(self, data: pd.DataFrame, target_col_name: str) -> pd.DataFrame:
        """
        Prepares the data for modeling using the internal preprocessor.
        This primarily involves feature engineering and lag creation.
        The preprocessor's `prepare_data` method is expected to set `self.preprocessor.feature_names`.

        Args:
            data (pd.DataFrame): Input dataframe with time series data.
            target_col_name (str): Name of the target column.

        Returns:
            pd.DataFrame: Transformed dataframe with engineered features and the target column.
        """
        if self.preprocessor is None:
            # This case should ideally not be hit if __init__ always creates one
            self.preprocessor = XGBoostPreprocessor(n_lags=self.n_lags)
            print("Advertencia: Preprocessor reinicializado en prepare_data.")
        
        # Call the preprocessor's prepare_data method.
        # Ensure your XGBoostPreprocessor.prepare_data expects 'target_col' as the argument name.
        processed_df = self.preprocessor.prepare_data(data, target_col=target_col_name) 
        
        # After preprocessor.prepare_data, self.preprocessor.feature_names should be set.
        # Assign it to the model's self.feature_names for use in optimization and other methods.
        if hasattr(self.preprocessor, 'feature_names') and self.preprocessor.feature_names:
            self.feature_names = list(self.preprocessor.feature_names)
        else:
            # Fallback: if preprocessor doesn't set it, try to infer from columns minus target
            print(f"Advertencia: `self.preprocessor.feature_names` no fue establecido por el preprocesador.")
            if target_col_name in processed_df.columns:
                 self.feature_names = processed_df.drop(columns=[target_col_name]).columns.tolist()
                 print(f"Advertencia: `feature_names` del modelo inferidos de `processed_df` (excluyendo target).")
            else: # If target_col_name is also not in processed_df, this is problematic
                 print(f"Advertencia: No se pudieron determinar `feature_names` del modelo. `target_col_name` ('{target_col_name}') no está en `processed_df.columns` y preprocessor no estableció `feature_names`.")
                 self.feature_names = processed_df.columns.tolist() # Best guess: all columns

        if not self.feature_names:
             print("Advertencia CRÍTICA: `self.feature_names` (del modelo) está vacío después de `prepare_data`.")
        
        return processed_df


    def optimize_hyperparameters(self, 
                                 X_train_scaled: np.ndarray, 
                                 y_train_scaled: np.ndarray, 
                                 feature_names: List[str], # Names of columns in X_train (original, before scaling)
                                                           # that correspond to X_train_scaled
                                 param_grid: Optional[Dict[str, Any]] = None, 
                                 cv: int = 3):
        """
        Optimizes hyperparameters for an XGBoost model within a pipeline.
        """
        if not feature_names: # These are the names for X_train_scaled
            raise ValueError("`feature_names` (para X_train_scaled) no puede estar vacío para la optimización.")
        
        # These feature_names are crucial because X_train_scaled is a numpy array.
        # They tell SelectFromModel what the original features were.
        # This self.feature_names becomes the definitive list of features entering the pipeline.
        self.feature_names = list(feature_names) 

        estimator_for_selection = XGBRegressor(
            n_estimators=50,
            random_state=self.model_base.random_state,
            objective=self.model_base.objective
        )

        if param_grid is None:
            param_grid = {
                'selector__threshold': ['median', '1.25*mean', -np.inf], # Use float -np.inf
                'xgb__n_estimators': [100, 200], 
                'xgb__max_depth': [3, 5, 7],
                'xgb__learning_rate': [0.01, 0.05, 0.1],
                'xgb__subsample': [0.7, 0.8], 
                'xgb__colsample_bytree': [0.7, 0.8], 
            }
        
        pipeline = Pipeline([
            ('selector', SelectFromModel(estimator=estimator_for_selection)),
            ('xgb', self.model_base)
        ])

        tscv = TimeSeriesSplit(n_splits=cv)
        
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1,
            refit=True
        )

        print(f"Iniciando GridSearchCV para XGBoost con {len(self.feature_names)} características de entrada al pipeline (escaladas)...")
        grid_search.fit(X_train_scaled, y_train_scaled.ravel()) # Ensure y_train_scaled is 1D
        print("GridSearchCV para XGBoost completado.")

        self.best_pipeline_ = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_

        selected_indices = self.best_pipeline_.named_steps['selector'].get_support(indices=True)
        self.selected_feature_names_ = [self.feature_names[i] for i in selected_indices]
        
        if 'selector__features_index' not in self.best_params_ : # Store for metadata
            self.best_params_['selector__features_index'] = selected_indices.tolist()

        if hasattr(self.best_pipeline_.named_steps['xgb'], 'feature_importances_'):
            self.feature_importances_ = self.best_pipeline_.named_steps['xgb'].feature_importances_
        
        print(f"Mejores parámetros encontrados: {self.best_params_}")
        print(f"Características seleccionadas por SelectFromModel: {len(self.selected_feature_names_)} ({self.selected_feature_names_})")
        return self

    def fit(self, 
            X_train_scaled: np.ndarray, 
            y_train_scaled: np.ndarray, 
            X_val_scaled: Optional[np.ndarray] = None, 
            y_val_scaled: Optional[np.ndarray] = None, 
            early_stopping_rounds: Optional[int] = 10):
        """
        Fits the best_pipeline_ (typically set by optimize_hyperparameters).
        """
        if self.best_pipeline_ is None:
            print("Advertencia: `best_pipeline_` no está configurado. Creando un pipeline por defecto y ajustándolo.")
            if not self.feature_names:
                raise ValueError("`self.feature_names` no está establecido. No se puede crear un pipeline por defecto para SelectFromModel sin conocer las características de entrada.")

            estimator_for_selection = XGBRegressor(n_estimators=50, random_state=self.model_base.random_state)
            self.best_pipeline_ = Pipeline([
                ('selector', SelectFromModel(estimator=estimator_for_selection, threshold=-np.inf)), 
                ('xgb', self.model_base)
            ])
        
        fit_params = {}
        if X_val_scaled is not None and y_val_scaled is not None and early_stopping_rounds is not None:
            try:
                X_val_transformed_for_xgb = self.best_pipeline_.named_steps['selector'].transform(X_val_scaled)
                fit_params['xgb__early_stopping_rounds'] = early_stopping_rounds
                fit_params['xgb__eval_set'] = [(X_val_transformed_for_xgb, y_val_scaled.ravel())] 
                fit_params['xgb__verbose'] = True 
                print(f"Ajustando best_pipeline_ con early stopping (eval_set shape: {X_val_transformed_for_xgb.shape})...")
            except Exception as e:
                print(f"Error al preparar eval_set para early stopping: {e}. Continuando sin early stopping.")
        else:
            print("Ajustando best_pipeline_...")
        
        self.best_pipeline_.fit(X_train_scaled, y_train_scaled.ravel(), **fit_params) 
        print("Modelo XGBoost (best_pipeline_) ajustado.")

        if 'selector' in self.best_pipeline_.named_steps:
            selected_indices = self.best_pipeline_.named_steps['selector'].get_support(indices=True)
            if self.feature_names: 
                self.selected_feature_names_ = [self.feature_names[i] for i in selected_indices]
            else:
                 print("Advertencia: `selected_feature_names_` no pudo ser actualizado porque `self.feature_names` (entrada al pipeline) no está establecido.")

        if hasattr(self.best_pipeline_.named_steps['xgb'], 'feature_importances_'):
            self.feature_importances_ = self.best_pipeline_.named_steps['xgb'].feature_importances_
        return self

    def predict(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the fitted `best_pipeline_`.
        """
        if self.best_pipeline_ is None:
            raise ValueError("El modelo (best_pipeline_) no ha sido ajustado.")
        return self.best_pipeline_.predict(X_scaled)

# model_xgb/xgb_model.py

    def evaluate(self, 
                 X_test_processed: np.ndarray, 
                 y_test_actual_unscaled: pd.Series,
                 target_scaler: object) -> Dict[str, float]:
        """
        Evalúa el modelo en datos de prueba. Maneja tanto datos escalados como no escalados.
        """
        # Las predicciones se hacen sobre datos procesados (que pueden o no estar escalados)
        y_pred = self.predict(X_test_processed)

        # Si se proporcionó un escalador, significa que las predicciones están escaladas y deben ser invertidas.
        if target_scaler:
            y_pred_unscaled = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        else:
            # Si no hay escalador, las predicciones ya están en la escala correcta.
            y_pred_unscaled = y_pred.flatten()

        self.metrics = evaluate_regression(y_test_actual_unscaled.values, y_pred_unscaled)
        print(f"Métricas de evaluación XGBoost (sobre datos desescalados): {self.metrics}")
        return self.metrics

    def predict_future(self, historical_data_df: pd.DataFrame, forecast_horizon: int, target_col: str = 'Close') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predice futuros valores del objetivo de forma recursiva.
        """
        print("\n--- Entrando a predict_future para XGBoost ---")
        if self.best_pipeline_ is None: raise ValueError("El modelo (best_pipeline_) no ha sido ajustado.")
        if self.preprocessor is None: raise ValueError("Preprocessor no disponible en el modelo.")
        
        # ELIMINAMOS LAS COMPROBACIONES ESTRICTAS DE LOS ESCALADORES
        # if self.target_scaler is None: raise ValueError("target_scaler no disponible en el modelo.")
        # if self.feature_scaler is None: raise ValueError("feature_scaler no disponible en el modelo.")

        if not hasattr(self.preprocessor, 'feature_names') or not self.preprocessor.feature_names:
             raise ValueError("`self.preprocessor.feature_names` no están establecidos.")

        current_data_df = historical_data_df.copy()
        if not isinstance(current_data_df.index, pd.DatetimeIndex):
            current_data_df.index = pd.to_datetime(current_data_df.index)
        current_data_df = current_data_df.sort_index()

        predictions_unscaled_list = []
        lower_bounds_unscaled_list = [] 
        upper_bounds_unscaled_list = [] 

        for i in range(forecast_horizon):
            print(f"\n--- Paso de predicción XGBoost {i + 1}/{forecast_horizon} ---")
            
            temp_prepared_df = self.preprocessor.prepare_data(current_data_df.copy(), target_col=target_col)
            
            if not all(isinstance(name, str) for name in self.preprocessor.feature_names):
                raise TypeError(f"self.preprocessor.feature_names debe ser una lista de strings. Obtenido: {self.preprocessor.feature_names}")

            try:
                features_df = temp_prepared_df[self.preprocessor.feature_names].iloc[-1:]
            except KeyError as e:
                missing_cols = set(self.preprocessor.feature_names) - set(temp_prepared_df.columns)
                available_cols = temp_prepared_df.columns.tolist()
                raise KeyError(f"Una o más `preprocessor.feature_names` ({missing_cols}) no encontradas en la salida de `preprocessor.prepare_data` (columnas disponibles: {available_cols}). Error: {e}")

            if features_df.isnull().any().any():
                print(f"Advertencia: NaNs en características para el paso {i+1} antes de escalar: {features_df.columns[features_df.isnull().any()].tolist()}")
                features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)

            # --- INICIO DE CAMBIOS ---
            # 1. Transformar características solo si existe un feature_scaler
            if self.feature_scaler:
                processed_feature_input_np = self.feature_scaler.transform(features_df)
            else:
                processed_feature_input_np = features_df.to_numpy()

            # 2. Realizar la predicción
            point_pred = self.best_pipeline_.predict(processed_feature_input_np)[0]

            # 3. Desescalar la predicción solo si existe un target_scaler
            if self.target_scaler:
                point_pred_unscaled = self.target_scaler.inverse_transform(np.array(point_pred).reshape(-1, 1)).flatten()[0]
            else:
                point_pred_unscaled = point_pred # La predicción ya está en la escala correcta
            # --- FIN DE CAMBIOS ---

            predictions_unscaled_list.append(point_pred_unscaled)
            lower_bounds_unscaled_list.append(point_pred_unscaled) 
            upper_bounds_unscaled_list.append(point_pred_unscaled)
            print(f"Predicción (Precio) XGBoost: {point_pred_unscaled:.4f}")

            last_index_val = current_data_df.index[-1]
            next_index_val = last_index_val + pd.tseries.offsets.BDay(1)

            new_row_data = {col: np.nan for col in historical_data_df.columns}
            new_row_data[target_col] = point_pred_unscaled

            prev_row = current_data_df.iloc[-1]
            if 'Open' in new_row_data and target_col in prev_row: new_row_data['Open'] = prev_row[target_col]
            volatility_factor = 0.01 
            if 'High' in new_row_data: new_row_data['High'] = max(point_pred_unscaled * (1 + volatility_factor), new_row_data.get('Open', point_pred_unscaled), point_pred_unscaled)
            if 'Low' in new_row_data: new_row_data['Low'] = min(point_pred_unscaled * (1 - volatility_factor), new_row_data.get('Open', point_pred_unscaled), point_pred_unscaled)
            if 'Volume' in new_row_data and 'Volume' in prev_row: new_row_data['Volume'] = prev_row['Volume'] * (0.95 + np.random.rand() * 0.1)

            new_row_df = pd.DataFrame([new_row_data], index=[next_index_val])
            for col in current_data_df.columns:
                if col not in new_row_df.columns:
                    new_row_df[col] = np.nan
            
            current_data_df = pd.concat([current_data_df, new_row_df[current_data_df.columns]])
            current_data_df = current_data_df.sort_index()

        print("--- Saliendo de predict_future para XGBoost ---")
        return np.array(predictions_unscaled_list), np.array(lower_bounds_unscaled_list), np.array(upper_bounds_unscaled_list)
    
    
    def save_model(self, model_path_prefix: str = "models/xgb_model", training_end_date: Optional[str] = None):
        """Saves the fitted model, preprocessor (with fitted scalers), and metadata."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor no está asignado. No se puede guardar el modelo.")
        
        if self.feature_scaler is not None:
            self.preprocessor.feature_scaler = self.feature_scaler
        if self.target_scaler is not None:
            self.preprocessor.target_scaler = self.target_scaler
        # self.preprocessor.feature_names should be set by its prepare_data method
        # self.feature_names (model) are the input to the GridSearchCV pipeline

        os.makedirs(os.path.dirname(model_path_prefix), exist_ok=True)

        pipeline_file = f"{model_path_prefix}_pipeline.joblib"
        joblib.dump(self.best_pipeline_, pipeline_file)
        print(f"Pipeline XGBoost guardado en: {pipeline_file}")

        components = {
            'preprocessor': self.preprocessor, 
            'pipeline_input_feature_names': self.feature_names, 
            'selected_feature_names_for_xgb': self.selected_feature_names_ 
        }
        components_file = f"{model_path_prefix}_components.joblib"
        joblib.dump(components, components_file)
        print(f"Componentes (preprocesador, nombres de características) guardados en: {components_file}")

        serializable_best_params = {}
        if self.best_params_:
            for k, v in self.best_params_.items():
                if isinstance(v, (np.ndarray, list)):
                    serializable_best_params[k] = [item.item() if hasattr(item, 'item') else item for item in v]
                elif hasattr(v, 'item'): serializable_best_params[k] = v.item()
                else: serializable_best_params[k] = v
        
        feature_importances_dict = None
        if self.feature_importances_ is not None and self.selected_feature_names_ is not None:
             if len(self.selected_feature_names_) == len(self.feature_importances_):
                  feature_importances_dict = dict(zip(self.selected_feature_names_, self.feature_importances_.tolist()))
             else:
                  print(f"Advertencia al guardar: Discrepancia en longitud. selected_feature_names_ ({len(self.selected_feature_names_)}) vs feature_importances_ ({len(self.feature_importances_)}).")

        metadata = {
            'model_type': 'XGBoost',
            'n_lags_from_model_init': self.n_lags, 
            'n_lags_from_preprocessor': getattr(self.preprocessor, 'n_lags', 'N/A'),
            'best_params': serializable_best_params,
            'feature_importances_selected': feature_importances_dict,
            'metrics': self.metrics,
            'timestamp': pd.Timestamp.now().isoformat(),
            'training_end_date': str(training_end_date) if training_end_date else None,
            'pipeline_file': os.path.basename(pipeline_file), 
            'components_file': os.path.basename(components_file), 
            'pipeline_input_feature_names': self.feature_names, 
            'preprocessor_output_feature_names': list(getattr(self.preprocessor, 'feature_names', [])),
            'selected_feature_names_by_selector': self.selected_feature_names_
        }
        
        metadata_file = f"{model_path_prefix}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadatos guardados en: {metadata_file}")

    @classmethod
    def load_model(cls, model_path_prefix: str):
        """Loads a TimeSeriesXGBoostModel from saved files."""
        pipeline_file = f"{model_path_prefix}_pipeline.joblib"
        if not os.path.exists(pipeline_file):
            raise FileNotFoundError(f"Archivo de pipeline no encontrado: {pipeline_file}")
        best_pipeline = joblib.load(pipeline_file)

        components_file = f"{model_path_prefix}_components.joblib"
        if not os.path.exists(components_file):
            raise FileNotFoundError(f"Archivo de componentes no encontrado: {components_file}")
        components = joblib.load(components_file)

        loaded_preprocessor: XGBoostPreprocessor = components.get('preprocessor')
        if loaded_preprocessor is None:
            raise ValueError("No se pudo cargar el preprocesador desde los componentes.")

        instance = cls(preprocessor=loaded_preprocessor) 
        
        instance.best_pipeline_ = best_pipeline
        
        # Restore pipeline_input_feature_names (features that selector expects)
        instance.feature_names = components.get('pipeline_input_feature_names', []) 
        if not instance.feature_names and hasattr(loaded_preprocessor, 'feature_names') and loaded_preprocessor.feature_names:
            instance.feature_names = list(loaded_preprocessor.feature_names)
            print("Advertencia al cargar: `pipeline_input_feature_names` no encontrado en componentes; usando `preprocessor.feature_names` como fallback para `instance.feature_names` (entrada al pipeline).")

        instance.selected_feature_names_ = components.get('selected_feature_names_for_xgb')
        
        # Scalers should be on instance.preprocessor and thus on instance via __init__
        if instance.feature_scaler is None and hasattr(loaded_preprocessor, 'feature_scaler'):
            instance.feature_scaler = loaded_preprocessor.feature_scaler
        if instance.target_scaler is None and hasattr(loaded_preprocessor, 'target_scaler'):
            instance.target_scaler = loaded_preprocessor.target_scaler

        metadata_file = f"{model_path_prefix}_metadata.json"
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            instance.best_params_ = metadata.get('best_params')
            instance.metrics = metadata.get('metrics')
            instance.n_lags = metadata.get('n_lags_from_preprocessor', instance.n_lags) # Prefer n_lags from preprocessor's saved state
            
            f_imp_dict = metadata.get('feature_importances_selected')
            if f_imp_dict and instance.selected_feature_names_:
                instance.feature_importances_ = np.array([f_imp_dict.get(name, 0.0) for name in instance.selected_feature_names_])
            
            pipeline_input_features_from_meta = metadata.get('pipeline_input_feature_names')
            if pipeline_input_features_from_meta:
                 instance.feature_names = pipeline_input_features_from_meta # Override if in metadata
            
            # Ensure preprocessor.feature_names is consistent with metadata if possible
            preprocessor_output_from_meta = metadata.get('preprocessor_output_feature_names')
            if preprocessor_output_from_meta and hasattr(instance.preprocessor, 'feature_names'):
                if list(instance.preprocessor.feature_names) != preprocessor_output_from_meta:
                    print(f"Advertencia al cargar: 'preprocessor.feature_names' ({list(instance.preprocessor.feature_names)}) difiere de 'preprocessor_output_feature_names' en metadatos ({preprocessor_output_from_meta}).")
                    # Optionally force update: instance.preprocessor.feature_names = preprocessor_output_from_meta
        else:
            print(f"Advertencia: Archivo de metadatos no encontrado en {metadata_file}.")

        print(f"Modelo XGBoost y componentes cargados desde el prefijo: {model_path_prefix}")
        return instance

