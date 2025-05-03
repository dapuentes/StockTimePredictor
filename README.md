# StockTimePredictor
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Descripción General

StockTimePredictor es un proyecto diseñado para pronosticar precios y tendencias del mercado de valores utilizando análisis de series temporales y diversos modelos de machine learning. Utiliza una arquitectura de microservicios orquestada por un API Gateway, permitiendo a los usuarios entrenar diferentes modelos y obtener predicciones para tickers de acciones específicos.

El objetivo principal es proporcionar una plataforma flexible para experimentar y desplegar diferentes modelos de pronóstico como LSTM, Random Forest, XGBoost, Prophet y Redes Neuronales Secuenciales.

## Arquitectura

El proyecto sigue una arquitectura de microservicios:

* **API Gateway (`api_gateway/`)**: Actúa como el único punto de entrada para todas las solicitudes. Enruta las peticiones de entrenamiento y predicción al microservicio del modelo apropiado.
* **Servicios de Modelos (`services/`)**: Cada subdirectorio (`model_lstm`, `model_rf`, `model_xgb`, etc.) contiene un microservicio separado que implementa un modelo de pronóstico específico. Cada servicio típicamente incluye:
    * `main.py` o `app.py`: La aplicación FastAPI para los endpoints de la API del servicio (entrenar, predecir).
    * `*_model.py`: La clase que define la lógica del modelo, preprocesamiento, entrenamiento y funciones de predicción.
    * `train.py`: Script que orquesta el proceso de entrenamiento del modelo.
    * `forecast.py`: Script que maneja la lógica de predicción.
    * `models/`: Directorio que almacena modelos serializados (`.joblib`, `.h5`) y metadatos (`.json`).
    * `requirements.txt`: Dependencias específicas de ese servicio.
* **Utilidades (`utils/`)**: Contiene funciones de ayuda compartidas para:
    * Importación de Datos (`import_data.py`): Carga de datos de acciones usando `yfinance`.
    * Preprocesamiento (`preprocessing.py`): Ingeniería de características (rezagos, indicadores técnicos, características estacionales), escalado, división y creación de secuencias.
    * Evaluación (`evaluation.py`): Cálculo de métricas de regresión (MSE, RMSE, MAE, MAPE).
    * Visualización (`visualizations.py`): Graficación de predicciones y pronósticos.
* **Scripts de Entrenamiento (`training/scripts`)**: Scripts de ejemplo para iniciar el entrenamiento de modelos (ej., `train_rf.py`).

## Estructura del Proyecto

```
StockTimePredictor/
│
├── requirements.txt           # Dependencias generales 
│
├── api_gateway/               # API Gateway (App FastAPI) 
│   └── app.py
│
├── services/                  # Microservicios de Modelos
│   ├── model_lstm/            # Servicio LSTM 
│   │   ├── main.py
│   │   ├── lstm_model.py
│   │   ├── train.py
│   │   ├── forecast.py
│   │   ├── models/
│   │   └── requirements.txt
│   │
│   ├── model_rf/              # Servicio Random Forest
│   │   ├── main.py
│   │   ├── rf_model2.py
│   │   ├── train.py
│   │   ├── forecast.py
│   │   ├── models/
│   │   └── requirements.txt
│   │
│   ├── model_xgb/             # Servicio XGBoost - Corre en el puerto 8003
│   │   ├── main_xgb.py        
│   │   ├── xgb_model.py
│   │   ├── forecast.py
│   │   ├── models/
│   │   └── requirements.txt   
│   │
│   ├── model_prophet/         
│   │
│
├── training/                  # Scripts y recursos de entrenamiento
│   └── scripts/
│       └── train_rf.py        # Script de ejemplo para entrenar RF
│
└── utils/                     # Funciones de utilidad compartidas
    ├── __init__.py
    ├── evaluation.py             
    ├── import_data.py            
    ├── preprocessing.py          
    └── visualizations.py          

```


---

## 🚀 Getting Started / Primeros Pasos

### 1. Clone this repo / Clona el repositorio
```bash

git clone https://github.com/dapuentes/StockTimePredictor.git
cd StockTimePredictor
```

### 2. Create virtual environment / Crea un entorno virtual

```bash
python -m venv venv
source venv/bin/activate    # on Linux/macOS
venv\Scripts\activate       # on Windows
```

### 3. Install dependencies / Instala las dependencias
```bash
pip install -r requirements.txt
```

### 4.  Usage / Uso
```
uvicorn api_gateway.main:app --reload
```

## 📈 Models Included / Modelos Incluidos
LSTM (Long Short-Term Memory)

Prophet (by Meta/Facebook)

Random Forest

XGBoost

Sequential Neural Network


## Services

Adentro se encuentra toda la logica de los modelos propuestos en el proyecto, separados por carpetas dentro de se tiene la siguiente logica:

- main.py o app.py: API correspondiente a cada modelo
- forecast.py: archivo encargado de realizar las predicciones con la logica necesaria de cada modelo
- X_model: Archivo contenedor de la clase con el modelo correspondiente y las funciones necesarias para ser llamadas en la API


## Utils / Utilidades

### evaluation:

Adentro hay funciones que entregan metricas para calificar la eficiciencia de los modelos

### import data:

Funcion encargada de importar los datos como ticker, usa la API de Yahoo Finance.  
Usa el simbolo del stock, una fecha de inicio, una fecha de fin y te entrega un dataframe con la siguientes columnas  
Date  -   Open  -  High  -  Low  -  Close  -  Volume  -  GreenDay

### preprocessing:

Archivo con funciones para procesar el dataframe y preparalo para modelamiento:
Trabajo de caracteristicas:
- add_lags
- add_technical_indicators
- add_seasonal_features
- feature_engineering
  
Preparamiento de datos:

- scale_data
- split_data
- create_sequences

### 📸 visualizations

Archivo con las funciones encargadas de las visualizaciones usadas en el proyecto
- plot_predictions: Plot true vs predicted values.
- plot_forecast: Plot the historical data and the forecast.
- plot_lstm_results: Plot LSTM model results
- .

## 🧾 License / Licencia

MIT License
See LICENSE for details.
