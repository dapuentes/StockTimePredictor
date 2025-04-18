from services.model_lstm.train import train_lstm_model
from services.model_lstm.forecast import forecast_future_prices
from utils.import_data import load_data

data = load_data()
model_save_path = "../../services/model_lstm/models/lstm_model.keras"

model = train_lstm_model(data, save_model_path=model_save_path, n_lags=10)

model.plot_training_history()

forecast = forecast_future_prices(model, data, 10)