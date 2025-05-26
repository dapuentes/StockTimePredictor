# Carpeta `model_lstm`

Esta carpeta contiene la implementación y los recursos relacionados con el modelo LSTM (Long Short-Term Memory) utilizado para predicción de series temporales financieras en este proyecto.

## Descripción

El modelo LSTM está diseñado para predecir precios futuros de acciones utilizando redes neuronales recurrentes. Incluye características avanzadas como:

- Optimización automática de hiperparámetros con Keras Tuner
- Estimación de incertidumbre mediante Monte Carlo Dropout
- Intervalos de confianza en las predicciones
- Regularización L2 y Batch Normalization
- Early stopping para prevenir sobreajuste

## Estructura de la carpeta

- `lstm_model.py`: Implementación principal de la clase `TimeSeriesLSTMModel` con arquitectura LSTM.
- `train.py`: Script para entrenar el modelo con datos históricos, incluyendo optimización de hiperparámetros.
- `forecast.py`: Módulo para realizar predicciones futuras con intervalos de confianza.
- `main.py`: Microservicio FastAPI que expone endpoints para entrenamiento y predicción.
- `requirements.txt`: Dependencias necesarias para ejecutar el código.
- `Dockerfile`: Configuración para containerización del servicio.

## Endpoints del API

El servicio expone los siguientes endpoints a través de FastAPI:

### Entrenamiento
- `POST /train/`: Entrena un nuevo modelo LSTM
- `POST /train_new_ticker/`: Entrena un modelo para un nuevo ticker
- `POST /train_update_ticker/`: Actualiza un modelo existente con nuevos datos

### Predicción
- `POST /predict/`: Realiza predicciones con un modelo entrenado
- `POST /predict_stock/`: Predice precios futuros de una acción específica

### Utilidades
- `GET /health/`: Verifica el estado del servicio
- `GET /models/`: Lista modelos disponibles

## Cómo usar

### Entrenar el modelo

```python
from train import train_lstm_model
import pandas as pd

# Cargar datos
data = pd.read_csv('data/stock_data.csv')

# Entrenar modelo con optimización de hiperparámetros
model, metrics = train_lstm_model(
    data=data,
    target_col='Close',
    sequence_length=60,
    n_lags=5,
    optimize_params=True,
    epochs=50,
    save_model_path='models/lstm_model'
)
```

### Realizar predicciones

```python
from forecast import forecast_future_prices_lstm
from lstm_model import TimeSeriesLSTMModel

# Cargar modelo entrenado
model = TimeSeriesLSTMModel.load_model('models/lstm_model')

# Realizar predicciones para los próximos 10 días
forecast, lower_bounds, upper_bounds = forecast_future_prices_lstm(
    model=model,
    data=historical_data,
    forecast_horizon=10,
    target_col='Close'
)
```

### Usar el API REST

```bash
# Entrenar modelo
curl -X POST "http://localhost:8000/train/" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "sequence_length": 60,
    "n_lags": 5,
    "optimize_params": true
  }'

# Realizar predicción
curl -X POST "http://localhost:8000/predict_stock/" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "forecast_horizon": 10
  }'
```

## Parámetros del modelo

### Arquitectura LSTM
- `lstm_units`: Número de unidades en cada capa LSTM (por defecto: 50)
- `dropout_rate`: Tasa de dropout para regularización (por defecto: 0.2)
- `sequence_length`: Longitud de secuencias de entrada (por defecto: 60)

### Entrenamiento
- `epochs`: Número máximo de épocas (por defecto: 50)
- `batch_size`: Tamaño del lote (por defecto: 32)
- `patience`: Paciencia para early stopping (por defecto: 10)
- `optimize_params`: Si optimizar hiperparámetros automáticamente

### Preprocesamiento
- `n_lags`: Número de características de retraso (por defecto: 5)
- `train_size`: Proporción de datos para entrenamiento (por defecto: 0.8)
- `validation_size`: Proporción de validación (por defecto: 0.1)

## Dependencias necesarias

Las principales dependencias incluyen:

- `tensorflow`: Framework de deep learning
- `keras`: API de alto nivel para redes neuronales
- `kerastuner`: Optimización de hiperparámetros
- `fastapi`: Framework web para el API
- `pandas`: Manipulación de datos
- `numpy`: Computación numérica
- `scikit-learn`: Herramientas de machine learning
- `uvicorn`: Servidor ASGI

Instalar todas las dependencias:

```bash
pip install -r requirements.txt
```

## Ejecutar el servicio

### Desarrollo local
```bash
python -m uvicorn main:app --reload --port 8000
```

### Con Docker
```bash
docker build -t lstm-service .
docker run -p 8000:8000 lstm-service
```

## Características técnicas

- **Regularización**: Dropout y regularización L2 para prevenir sobreajuste
- **Normalización**: Batch Normalization para estabilizar el entrenamiento
- **Optimización**: Adam optimizer con gradient clipping
- **Incertidumbre**: Monte Carlo Dropout para estimación de incertidumbre
- **Escalado**: Normalización automática de características y objetivo
- **Validación**: Early stopping basado en pérdida de validación

## Notas

- Se incluyen características de retraso (lags) y características técnicas automáticamente
- Los intervalos de confianza se calculan usando Monte Carlo Dropout
- El modelo guarda automáticamente escaladores y metadatos para reproducibilidad
- Se recomienda usar GPU para entrenamiento con datasets grandes