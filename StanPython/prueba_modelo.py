import pandas as pd
import numpy as np
import os
import stan
from py.ModeloStan import generate_arima_price_stan, forecast_arima_prices

def prueba_modelo():
    # Paso 1: Cargar y preparar datos
    os.system("rm -rf ~/.cache/httpstan")  # Limpiar caché de httpstan
    csv_path = "data/NU_Historical_Data.csv"
    data = pd.read_csv(csv_path, parse_dates=["Date"])
    data = data.iloc[1:]  # Eliminar la primera fila, si es necesario

    # Convertir columnas a los tipos adecuados
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data["Date"] = pd.to_datetime(data["Date"], format="%m/%d/%Y", errors="coerce")

    # Filtrar datos por rango de fechas
    start_date = "2021-12-09"
    end_date = "2024-02-26"
    mask = (data["Date"] >= start_date) & (data["Date"] <= end_date)
    df_filtered = data.loc[mask].dropna(subset=["Close"])

    if df_filtered.empty:
        raise ValueError("No hay datos en el rango de fechas especificado")

    # Crear la serie de precios con índice de fechas (similar a xts en R)
    serie = pd.Series(df_filtered["Close"].values, index=df_filtered["Date"])

    # Paso 2: Diferenciar la serie (d = 1)
    d = 1
    serie_diff = np.diff(serie, n=d)  # Descartamos el primer valor
    serie_diff_vector = serie_diff.tolist()
    last_value = float(serie.iloc[-1])
    
    if len(serie_diff_vector) < 10:
        raise ValueError(f"Serie diferenciada muy corta. Longitud actual: {len(serie_diff_vector)}")

    # Paso 3: Generar el modelo Stan a partir de la serie diferenciada
    model_info = generate_arima_price_stan(np.array(serie_diff_vector))
    
    # Leer el contenido del archivo .stan generado
    with open(model_info["stan_file"], "r") as f:
        stan_code = f.read()

    print("=== Código Stan Generado ===")
    print(stan_code)
    print("=== Fin Código Stan ===")

        
    # Paso 4 & 5: Preparar el diccionario de datos para Stan y compilar el modelo
    data_dict = {
        "N": int(len(serie_diff_vector)),
        "y": serie_diff_vector,
        "H": 10,  # Horizonte de pronóstico
        "last_value": last_value
    }

    # Use stan.build() synchronously
    stan_model = stan.build(stan_code, data=data_dict)
    
    # Paso 6: Ajustar (sample) el modelo Stan
    fit = stan_model.sample(num_chains=4, num_samples=2000)

    # Paso 7: Generar el pronóstico utilizando la función correspondiente
    pronosticos = forecast_arima_prices(fit, serie, d=d, horizon=10)
    
    # Mostrar resultados
    print("=== Pronósticos ===")
    print(pronosticos)
    
    return pronosticos

if __name__ == "__main__":
    prueba_modelo()