import pandas as pd
import numpy as np
import os
import json
import joblib
from prophet import Prophet
from typing import Optional, Dict, Any, Tuple

# Se asume que estos utils están en el PYTHONPATH
from utils.preprocessing import BasePreprocessor, ProphetPreprocessor
from utils.evaluation import evaluate_regression

class ProphetModel:
    """
    Una clase contenedora para el modelo de series de tiempo Prophet de Facebook,
    integrada con el pipeline de preprocesamiento y evaluación del proyecto.
    La responsabilidad de esta clase es modelar, no preprocesar los datos.
    """

    def __init__(self,
                 preprocessor: Optional[BasePreprocessor] = None,
                 changepoint_prior_scale: float = 0.05,
                 seasonality_prior_scale: float = 10.0,
                 holidays_prior_scale: float = 10.0,
                 seasonality_mode: str = 'additive'):
        """
        Inicializa el ProphetModel.

        Args:
            preprocessor (Optional[BasePreprocessor]): Una instancia de un preprocesador
                compatible (como ProphetPreprocessor), que se guardará con el modelo.
            changepoint_prior_scale (float): Parámetro de flexibilidad para los puntos de cambio.
            seasonality_prior_scale (float): Parámetro de flexibilidad para la estacionalidad.
            holidays_prior_scale (float): Parámetro de flexibilidad para los feriados.
            seasonality_mode (str): 'additive' o 'multiplicative'.
        """
        self.preprocessor = preprocessor
        self.model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            seasonality_mode=seasonality_mode,
        )
        self.best_params_: Optional[Dict[str, Any]] = None
        self.metrics: Optional[Dict[str, float]] = None
        self.has_fitted: bool = False
        self._regressors_added: bool = False

    def _add_regressors_from_df(self, df: pd.DataFrame):
        """
        Añade regresores al modelo Prophet basándose en las columnas del DataFrame.
        Este método configura el modelo interno y se ejecuta solo una vez.
        """
        if self._regressors_added:
            return

        # Las columnas que no son 'ds' o 'y' se consideran regresores.
        regressor_cols = [col for col in df.columns if col not in ['ds', 'y']]
        
        if regressor_cols:
            print(f"Detectando y añadiendo {len(regressor_cols)} regresores al modelo...")
            for col_name in regressor_cols:
                self.model.add_regressor(col_name)
                print(f"  -> Regresor '{col_name}' añadido.")
        else:
            print("No se detectaron regresores adicionales en los datos.")

        self._regressors_added = True

    def fit(self, df: pd.DataFrame):
        """
        Entrena (ajusta) el modelo Prophet usando un DataFrame ya preprocesado.

        Args:
            df (pd.DataFrame): DataFrame que DEBE contener las columnas 'ds' y 'y',
                y opcionalmente columnas adicionales para los regresores.
        """
        if 'ds' not in df.columns or 'y' not in df.columns:
            raise ValueError("El DataFrame para `fit` debe contener las columnas 'ds' y 'y'.")

        # Configurar el modelo con los regresores presentes en los datos de entrenamiento
        self._add_regressors_from_df(df)

        print("Ajustando el modelo Prophet...")
        self.model.fit(df)
        self.has_fitted = True
        print("Modelo ajustado exitosamente.")
        return self

    def predict(self, future_df: pd.DataFrame) -> pd.DataFrame:
        """
        Realiza una predicción usando un DataFrame futuro.

        Args:
            future_df (pd.DataFrame): DataFrame con columna 'ds' y las columnas
                de los regresores para las fechas futuras a predecir.

        Returns:
            pd.DataFrame: Un DataFrame de Prophet con el pronóstico.
        """
        if not self.has_fitted:
            raise ValueError("El modelo debe ser entrenado primero con `fit()`.")

        print(f"Realizando predicción para {len(future_df)} periodos...")
        forecast = self.model.predict(future_df)
        return forecast

    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """
        Evalúa el rendimiento del modelo en un conjunto de datos de prueba.

        Args:
            test_df (pd.DataFrame): DataFrame con formato Prophet ('ds', 'y' y regresores)
                para la evaluación.

        Returns:
            Dict[str, float]: Un diccionario con las métricas de rendimiento.
        """
        if not self.has_fitted:
            raise ValueError("El modelo debe ser entrenado primero para poder evaluar.")

        # Realizar predicción en el conjunto de prueba
        forecast = self.predict(test_df)

        # Extraer valores reales y predichos
        y_true = test_df['y']
        # Asegurar que las longitudes coincidan al alinear predicciones con datos de prueba
        y_pred = forecast.set_index('ds').loc[test_df.set_index('ds').index]['yhat']

        # Calcular métricas usando la función de utilidad
        self.metrics = evaluate_regression(y_true.values, y_pred.values)
        print(f"Métricas de evaluación: {self.metrics}")
        return self.metrics

    def save_model(self, model_path_prefix: str, training_end_date: Optional[str] = None):
        """
        Guarda el modelo Prophet, el preprocesador y los metadatos.

        Args:
            model_path_prefix (str): Prefijo para la ruta de los archivos (ej. 'models/prophet_NU').
            training_end_date (Optional[str]): La última fecha usada en el entrenamiento.
        """
        if not self.has_fitted:
            raise ValueError("Solo se puede guardar un modelo entrenado.")

        os.makedirs(os.path.dirname(model_path_prefix), exist_ok=True)

        # Guardar el modelo Prophet serializado
        model_file = f"{model_path_prefix}_prophet.json"
        with open(model_file, 'w') as fout:
            from prophet.serialize import model_to_json
            fout.write(model_to_json(self.model))
        print(f"Modelo Prophet serializado guardado en: {model_file}")

        # Guardar otros componentes (preprocesador)
        components = {'preprocessor': self.preprocessor}
        components_file = f"{model_path_prefix}_components.joblib"
        joblib.dump(components, components_file)
        print(f"Componentes (preprocesador) guardados en: {components_file}")

        # Guardar metadatos
        metadata = {
            'model_type': 'Prophet',
            'best_params': self.best_params_,
            'metrics': self.metrics,
            'timestamp': pd.Timestamp.now().isoformat(),
            'training_end_date': training_end_date,
            'prophet_model_file': os.path.basename(model_file),
            'components_file': os.path.basename(components_file)
        }
        metadata_file = f"{model_path_prefix}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadatos guardados en: {metadata_file}")

    @classmethod
    def load_model(cls, model_path_prefix: str) -> 'ProphetModel':
        """
        Carga un ProphetModel desde los archivos guardados.

        Args:
            model_path_prefix (str): El prefijo usado para guardar los archivos del modelo.

        Returns:
            ProphetModel: Una instancia de la clase con el modelo y componentes cargados.
        """
        # Cargar componentes
        components_file = f"{model_path_prefix}_components.joblib"
        if not os.path.exists(components_file):
            raise FileNotFoundError(f"Archivo de componentes no encontrado: {components_file}")
        components = joblib.load(components_file)
        preprocessor = components.get('preprocessor')

        # Crear una nueva instancia de la clase
        instance = cls(preprocessor=preprocessor)

        # Cargar el modelo Prophet serializado
        model_file = f"{model_path_prefix}_prophet.json"
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Archivo de modelo Prophet no encontrado: {model_file}")
        with open(model_file, 'r') as fin:
            from prophet.serialize import model_from_json
            instance.model = model_from_json(fin.read())
        
        instance.has_fitted = True
        instance._regressors_added = True # Asumimos que si se guardó, ya tiene regresores.

        # Cargar metadatos
        metadata_file = f"{model_path_prefix}_metadata.json"
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            instance.best_params_ = metadata.get('best_params')
            instance.metrics = metadata.get('metrics')

        print(f"Modelo Prophet y componentes cargados desde el prefijo: {model_path_prefix}")
        return instance

    def plot_forecast(self, forecast: pd.DataFrame, **kwargs):
        """Genera un gráfico del pronóstico."""
        if not self.has_fitted:
            raise ValueError("El modelo debe estar entrenado para graficar.")
        fig = self.model.plot(forecast, **kwargs)
        return fig

    def plot_components(self, forecast: pd.DataFrame, **kwargs):
        """Genera gráficos de los componentes del modelo (tendencia, estacionalidad)."""
        if not self.has_fitted:
            raise ValueError("El modelo debe estar entrenado para graficar componentes.")
        fig = self.model.plot_components(forecast, **kwargs)
        return fig

