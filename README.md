# StockTimePredictor
Repositorio para predecir acciones mediante análisis de series temporales y modelos de machine learning, generando pronósticos y tendencias del mercado

Estructura:
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
