
from utils.import_data import *
from utils.preprocessing import feature_engineering,split_data, scale_data
from sklearn.preprocessing import RobustScaler
from utils.visualizations import plot_forecast
from utils.import_data import load_data
from services.model_prophet.prophet_model import ProphetModel, train_prophet_model
from typing import Tuple, List, Dict, Optional
from services.model_xgb.forecast import forecast_future_prices
from datetime import timedelta

# services/model_prophet/prophet_service.py

def train(
    data: pd.DataFrame,
    **kwargs
) -> Tuple[ProphetModel, dict, Optional[pd.DataFrame]]:
    """Siempre devuelve (model, metrics, future_df). Si no se pidió forecast, future_df será None."""
    result = train_prophet_model(data=data, **kwargs)
    # train_prophet_model puede devolver tupla de 2 o 3 elementos
    if len(result) == 2:
        model, metrics = result
        future_df = None
    else:
        model, metrics, future_df = result
    return model, metrics, future_df

def evaluate(
    model: ProphetModel,
    data: pd.DataFrame,
    train_size: float = 0.8
) -> Dict:
    # 1. preparar
    df = model.prepare_data(data, target_col='Close', regressor_cols=['Open','High','Low','Volume'])
    # 2. partir
    n = int(len(df) * train_size)
    test_df = df.iloc[n:]
    # 3. métricas
    metrics = model.evaluate(test_df)
    # 4. muestras
    preds = model.predict(test_df)
    samples = test_df[['ds','y']].copy()
    samples['y_pred'] = preds
    return {
        "metrics": metrics,
        "samples": samples.tail(5).to_dict(orient="records")
    }
# services/model_prophet/prophet_service.py

from datetime import timedelta
from typing import List, Dict
import pandas as pd
from services.model_prophet.prophet_model import ProphetModel

def predict(
    model: ProphetModel,
    data: pd.DataFrame,
    forecast_horizon: int,
    regressor_cols: List[str] = None,
    target_col: str = 'Close'
) -> List[Dict]:
    """
    Crea un DataFrame futuro con 'ds' y regresores fijos en su último valor,
    y llama a model.predict para obtener yhat.
    """
    # 1) Asegúrate de haber entrenado (o recarga todo el histórico)
    if not model.has_fitted:
        df_hist = model.prepare_data(data, target_col=target_col, regressor_cols=regressor_cols)
        model.fit(df_hist)

    # 2) Construye las fechas futuras
    last_date = data.index[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_horizon)]
    future = pd.DataFrame({'ds': future_dates})

    # 3) Añade cada regresor con su último valor observado
    if regressor_cols:
        for col in regressor_cols:
            if col in data.columns:
                last_val = data[col].iloc[-1]
                future[col] = last_val
            else:
                # rellena con 0 si no existe en el DataFrame
                future[col] = 0

    # 4) Llamar a model.predict, que aceptará ds + regresores
    yhat = model.predict(future)

    # 5) Empaqueta la salida con formato JSON
    return [
        {
            "date": d.strftime("%Y-%m-%d"),
            "yhat": float(pred)
        }
        for d, pred in zip(future_dates, yhat)
    ]
