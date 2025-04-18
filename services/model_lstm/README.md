# Carpeta `model_rf`

Esta carpeta contiene la implementación y los recursos relacionados con el modelo Random Forest utilizado en este proyecto.

## Estructura de la carpeta

El proyecto está dividido en varios módulos clave:


1. **`lstm_model.py` **
Contiene la implementación de la clase `TimeSeriesLSTMModel`, que gestiona:
    - La construcción del modelo LSTM.
    - El preprocesamiento de los datos mediante `create_sequences` y `prepare_data`.
    - Métodos altamente personalizables para ajustar el modelo, optimizar hiperparámetros y realizar predicciones futuras.
    - Funciones para guardar, cargar y visualizar métricas del modelo, como gráficos de resultados y del entrenamiento.

2. **`train.py` **
Este módulo coordina el flujo completo del entrenamiento del modelo. Se manejan pasos como:
    - Preprocesamiento y escalado de datos.
    - División de datos en entrenamiento, validación y prueba.
    - Secuencialización de datos para garantizar la compatibilidad con el modelo LSTM.
    - Optimización de hiperparámetros y entrenamiento del modelo.
    - Evaluación del modelo y guardado de los resultados.

3. **`forecast.py` **
Incluye la función `forecast_future_prices`, que utiliza un modelo entrenado para realizar predicciones de valores futuros. Integra los siguientes pasos:
    - Preparación de datos históricos recientes.
    - Creación de secuencias temporales para realizar la predicción.
    - Escalado y desescalado automático de datos según corresponda.
    - Visualización de los valores pronosticados.

## Requisitos
Para ejecutar el proyecto, asegúrate de tener los siguientes paquetes instalados:
- Python 3.12 o superior
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

Instala las dependencias utilizando `pip`:
``` bash
pip install -r requirements.txt
```
## Uso
### 1. Preparar los Datos
El conjunto de datos de entrada debe ser un archivo CSV o un `DataFrame` que contenga datos históricos con al menos una columna de características objetivo para predecir (como los precios "Close").
#### Ejemplo de datos:

| Date | Open | High | Low | Close | Volume |
| --- | --- | --- | --- | --- | --- |
| 2023-01-01 | 100.5 | 102.0 | 99.5 | 101.0 | 5000 |
| 2023-01-02 | 101.0 | 103.0 | 100.0 | 102.5 | 6000 |
| ... | ... | ... | ... | ... | ... |
### 2. Entrenar el Modelo
Usa la función `train_lstm_model` en el módulo `train.py` para entrenar el modelo sobre los datos.
#### Ejemplo de Código:
``` python
from services.model_lstm.train import train_lstm_model
from utils.import_data import load_data

# Cargar datos
data = load_data("tu_ticket")

# Ruta de guardado del modelo
model_save_path = "services/model_lstm/models/lstm_model.keras"

# Entrenar el modelo
model = train_lstm_model(data, save_model_path=model_save_path, n_lags=10)
```
#### Parámetros:
- **`data`**: El conjunto de datos en formato de `DataFrame`.
- **`n_lags`**: Número de pasos temporales a considerar en las secuencias.
- **`target_col`** (`default='Close'`): Columna objetivo.
- **`train_size`** (`default=0.8`): Fracción del conjunto de datos destinada al entrenamiento.
- **`validation_size`** (`default=0.2`): Fracción del conjunto de datos de entrenamiento destinada a validación.
- **`batch_size`** (`default=32`): Tamaño de lote para entrenamiento.
- **`epochs`** (`default=100`): Número de épocas de entrenamiento.
- **`save_model_path`**: Ruta donde se guardará el modelo.

### 3. Predecir Valores Futuros
Utiliza la función `forecast_future_prices` en el módulo `forecast.py` para generar predicciones basadas en un modelo entrenado.
#### Ejemplo de Código:
``` python
from services.model_lstm.forecast import forecast_future_prices
from utils.import_data import load_data

# Cargar datos históricos
data = load_data("data/stock_prices.csv")

# Cargar modelo entrenado
from services.model_lstm.lstm_model import TimeSeriesLSTMModel

model = TimeSeriesLSTMModel.load_model("services/model_lstm/models/lstm_model.keras")

# Generar pronóstico
forecast = forecast_future_prices(model, data, forecast_horizon=10)

print("Forecasted prices:", forecast)
```
#### Parámetros:
- **`model`**: Un objeto de tipo `TimeSeriesLSTMModel` previamente entrenado y cargado.
- **`data`**: El conjunto de datos históricos para realizar predicciones.
- **`forecast_horizon`**: Número de días a pronosticar.
- **`target_col`** (`default='Close'`): Columna objetivo.

### 4. Salida del Modelo
El modelo genera las siguientes salidas:
1. **Métricas de Evaluación**
Al evaluar el modelo, se generan métricas como `mean_squared_error`, `mean_absolute_error`, etc., accesibles en la propiedad `model.metrics`.
2. **Visualización**
Métodos como `plot_results` y `plot_forecast` se pueden usar para graficar las predicciones y comparar los valores reales con los estimados.
3. **Modelo Guardado**
El modelo entrenado se guarda en la ruta indicada (`save_model_path`) en formato `.keras`.

## Ejemplo de Flujo Completo
``` python
from services.model_lstm.train import train_lstm_model
from services.model_lstm.forecast import forecast_future_prices
from utils.import_data import load_data

# 1. Cargar Datos
data = load_data("data/stock_prices.csv")

# 2. Entrenar el Modelo
model_save_path = "services/model_lstm/models/lstm_model.keras"
model = train_lstm_model(data, save_model_path=model_save_path, n_lags=10)

# 3. Pronosticar Precios Futuros
forecast = forecast_future_prices(model, data, forecast_horizon=10)
print(f"Forecasted prices: {forecast}")
```
## Visualizaciones
### Ejemplo de Resultados del Modelo
``` python
# Graficar resultados del modelo
model.plot_results(data, predictions, target_col='Close')

# Graficar histórico de entrenamiento
model.plot_training_history()
```
