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

2. Create virtual environment / Crea un entorno virtual

```bash
python -m venv venv
source venv/bin/activate    # on Linux/macOS
venv\Scripts\activate       # on Windows
```

3. Install dependencies / Instala las dependencias
```bash
pip install -r requirements.txt
```

4.  Usage / Uso
```
uvicorn api_gateway.main:app --reload
```

📈 Models Included / Modelos Incluidos
LSTM (Long Short-Term Memory)

Prophet (by Meta/Facebook)

Random Forest

XGBoost

Sequential Neural Network

📸 Visualizations / Visualizaciones
All models support:

Forecast plots

Confidence intervals

Comparative performance

MAE / RMSE / MAPE metrics

🧾 License / Licencia

MIT License
See LICENSE for details.
