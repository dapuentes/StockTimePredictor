import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import traceback
import numpy as np
import pandas as pd
import stan

from fastapi import FastAPI, Query
from typing import Optional

# Importar funciones de ModeloStan
from py.ModeloStan import generate_arima_price_stan, forecast_arima_prices

app = FastAPI(
    title="API de pronóstico ARIMA-Modelo Stan",
    description="API para pronóstico de series temporales ARIMA con Modelo Bayesiano Stan",
    version="0.1",
)


# Endpoint raíz
@app.get("/")
def read_root():
    """
    Endpoint raíz que muestra que la API está en funcionamiento
    """

    return {"message": "API de pronóstico ARIMA-Modelo Stan"}

# Endpoint para generar el modelo Stan
@app.get("/generate_stan")
async def generate_stan_model(
    serie: str = Query(default = "1,2,3,4,5",
                       description = "Serie numérica separada por comas"
    )
):
    """
    Endpoint para generar modelo Stan a partir de serie diferenciada
   
    Ejemplo de llamada:
      GET /generate_stan?serie=1,2,3,4,5

    Parámetros:
      - serie: Cadena de caracteres con valores numéricos separados por comas.
    
    Respuesta:
      JSON con:
        - message: Mensaje indicando que el modelo se generó correctamente.
        - output_file: Información (por ejemplo, la ruta) del archivo Stan generado.
    """
    # Manejo de errores
    try:
        # Convertir string a lista de floats
        serie_diff = [float(x.strip()) for x in serie.split(",")]

        # Validar la serie numérica
        if not serie_diff:
            raise ValueError("La serie no puede estar vacía")
        
        result = generate_arima_price_stan(np.array(serie_diff))

        return {
            "message": "Modelo Stan generado correctamente", 
            "output_file": result
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
    
@app.get("/forecast")
async def forecast_prices(
    horizon: int = Query(
        default = 10, ge = 1, le = 100,
        description = "Horizonte de pronóstico (número de pasos)"
    ),
    start_date: Optional[str] = Query(
        default = "2021-12-09", 
        description = "Fecha de inicio de pronóstico en formato YYYY-MM-DD"),
    end_date: Optional[str] = Query(
        default = "2024-02-26", 
        description = "Fecha de fin de pronóstico en formato YYYY-MM-DD"
        )
):
    """
    Endpoint para generar pronóstico de precios a partir de modelo Stan ajustado

    Ejemplo de uso:
      GET /forecast?horizon=10&start_date=2021-12-09&end_date=2024-02-26

    Parámetros:
      - horizon: Número de períodos a pronosticar (por ejemplo, 10)
      - start_date: Fecha de inicio para filtrar los datos (formato 'AAAA-MM-DD')
      - end_date: Fecha de fin para filtrar los datos (formato 'AAAA-MM-DD')

    Respuesta:
      JSON con:
        - forecast: Lista de pronósticos para cada período, con valores 'mean', 'lower' y 'upper'
        - metadata: Información adicional (rango de fechas, horizonte y número total de observaciones)
    """
    
    # Manejo de errores
    try:
        # Ruta del CSV con precios
        csv_path = "data/NU_Historical_Data.csv"

        # Verificar si el archivo existe
        if not os.path.exists(csv_path):
            raise FileNotFoundError("Archivo CSV no encontrado")
        
        # Cargar datos de precios
        data = pd.read_csv(csv_path, parse_dates=["Date"])
        data = data.iloc[1:]  # Eliminar primera fila

        # Convertir explícitamente la columna "Close" a float
        data["Close"] = pd.to_numeric(data["Close"], errors='coerce')
        # Convertir explícitamente la columna "Date" a datetime
        data["Date"] = pd.to_datetime(data["Date"], format="%m/%d/%Y", errors="coerce")

        # Filtrar datos por fecha
        mask = (data["Date"] >= start_date) & (data["Date"] <= end_date)
        df_filtered = data.loc[mask].dropna(subset=["Close"])
        if df_filtered.empty:
            raise ValueError("No hay datos disponibles para el rango de fechas proporcionado")
        
        # Serie de precios
        serie = pd.Series(df_filtered["Close"].values, index=df_filtered["Date"])

        # Diferenciar la serie (d = 1)
        d = 1
        # Aquí usamos np.diff sin descartar elementos (asegúrate de que el comportamiento sea el deseado)
        serie_diff = np.diff(serie, n=d)
        # Convertir a lista para el diccionario de datos
        serie_diff_vector = serie_diff.tolist()
        last_value = float(serie.iloc[-1])
        
        if len(serie_diff_vector) < 10:
            raise ValueError(f"Serie diferenciada muy corta. Longitud actual: {len(serie_diff_vector)}")
        
        # Generar modelo Stan (se asume que generate_arima_price_stan escribe el archivo .stan)
        model_info = generate_arima_price_stan(np.array(serie_diff_vector))
        
        # Leer el contenido del archivo .stan generado
        with open(model_info["stan_file"], "r") as f:
            stan_code = f.read()
                
        # Preparar el diccionario de datos para Stan
        data_dict = {
            "N": int(len(serie_diff_vector)),
            "y": serie_diff_vector,  # Se pasa la lista de diferencias
            "H": int(horizon),
            "last_value": float(last_value)
        }
        print("Data dict:", data_dict)
        
        # Compilar el modelo Stan en un hilo separado para evitar conflictos con el loop asíncrono
        from concurrent.futures import ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=1)
        stan_model = await asyncio.get_running_loop().run_in_executor(
            executor, lambda: stan.build(stan_code, data=data_dict)
        )
        
        # Muestreo del modelo
        fit = await asyncio.get_running_loop().run_in_executor(
            executor, lambda: stan_model.sample(num_chains=4, num_samples=2000)
        )
        
        # Generar el pronóstico
        pronosticos = forecast_arima_prices(fit, serie, d=d, horizon=horizon)
        
        # Formatear pronósticos a JSON
        forecast_df = pronosticos.reset_index()
        forecast_df["date"] = forecast_df["index"].astype(str)
        forecast_result = forecast_df.to_dict(orient="records")
        
        return {
            "forecast": forecast_result,
            "metadata": {
                "start_date": start_date,
                "end_date": end_date,
                "horizon": horizon,
                "total_obs": len(serie)
            }
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "error_details": {
                "start_date": start_date,
                "end_date": end_date,
                "horizon": horizon
            },
            "traceback": traceback.format_exc()
        }
    
# Ejectuar la aplicación
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host = "0.0.0.0", port = 8000)