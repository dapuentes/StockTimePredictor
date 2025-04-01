from services.model_rf.train import train_ts_model
from services.model_rf.forecast import forecast_future_prices
from utils.import_data import load_data

data = load_data()

model_save_path = "services/model_rf/models/rf_model.joblib"

model = train_ts_model(data, save_model_path=model_save_path, n_lags=10)

'''
Esto es un ejemplo de cómo usar el modelo para pronosticar precios futuros.
Idealmente, esto debería estar en un endpoint separado o en un script diferente.
El primer argumento idealmente debería ser el modelo guardado, pero aquí lo pasamos directamente.
'''
forecast = forecast_future_prices(model, data, 10)
