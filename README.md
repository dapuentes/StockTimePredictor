# StockTimePredictor
Repositorio para predecir acciones mediante análisis de series temporales y modelos de machine learning, generando pronósticos y tendencias del mercado

Estructura:
```
StockTimePredictor/
│
│
├── requirements.txt               # Project dependencies
│
├── __pycache__/                   # Python cache files
├── api_gateway/                   # API gateway for model coordination
├── services/                      # Model implementation services
│   ├── __init__.py
│   ├── model_lstm/                # LSTM model implementation
│   │   └── lstm_model.ipynb
│   ├── model_prophet/             # Prophet model implementation
│   │   └── app.py
│   │   └── models/
│   │      └── prophet_model_metadata.json
│   └── prophet_model.joblib
│   ├── model_rf/                  # Random Forest model implementation
│   ├── model_sequiential/         # Sequential neural network model implementation
│   │   └── sequential_model.ipynb
│   ├── model_xgb/                 # XGBoost model implementation
│   │   ├── app.py
│   │   └── models/
│   │       └── xgb_model_metadata.json
│   └── scripts/                   # Utility scripts
└── utils/                         # Utility functions
    ├── __init__.py
    ├── evaluation.py              # Model evaluation utilities
    ├── import_data.py             # Data import utilities
    ├── preprocessing.py           # Data preprocessing utilities
    └── visualizations.py          # Visualization utilities

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
