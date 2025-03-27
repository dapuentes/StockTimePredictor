# API de Pronóstico ARIMA con Stan

Esta API está desarrollada en Python utilizando FastAPI y permite generar y ajustar modelos ARIMA bayesianos con Stan. La API facilita tanto la generación del código Stan a partir de una serie diferenciada como la realización de pronósticos sobre series temporales.

- **IMPORTANTE:** Versión inicial, hay que corregir el modelo de Stan para que funciones de manera más general con lapsos de tiempo y un modelo ARIMA adecuado

## Descripción

La API cuenta con dos endpoints principales:

- **/generate_stan**: Genera el modelo Stan a partir de una serie numérica diferenciada que se pasa como parámetro.
- **/forecast**: Ajusta el modelo Stan utilizando datos históricos (leídos de un archivo CSV) y devuelve un pronóstico con un horizonte especificado.

Además, se incluye un script de prueba (`prueba_modelo.py`) que ejecuta el flujo completo (lectura de datos, generación del modelo, ajuste y pronóstico) de forma local.

## Estructura del Proyecto

mi_proyecto/ ├── api.py # Archivo principal de la API FastAPI ├── py/ │ └── ModeloStan.py # Módulo con las funciones generate_arima_price_stan y forecast_arima_prices ├── data/ │ └── NU_Historical_Data.csv # Archivo CSV con datos históricos de precios ├── prueba_modelo.py # Script de prueba para ejecutar el flujo completo ├── requirements.txt # Lista de dependencias del proyecto ├── Dockerfile # Archivo para construir la imagen Docker ├── .gitignore # Archivos y carpetas a ignorar (incluye el entorno virtual) └── README.md # Este archivo

## Uso Local

Para iniciar la API localmente:
uvicorn api:app --reload --host 0.0.0.0 --port 8000

### Script de Prueba

Para ejecutar el flujo completo (generación del modelo, ajuste y pronóstico) sin usar la API, ejecuta:

python prueba_modelo.py

Esto imprimirá en la consola los pronósticos generados y otros mensajes de depuración.

## Uso de Docker

El proyecto incluye un Dockerfile para facilitar el despliegue.

### Construir la Imagen

Desde la raíz del proyecto se ejecuta:

docker build -t api_stan_python .

### Ejecutar la API en Docker

Para iniciar la API en un contenedor:

docker run -d -p 8000:8000 api_stan_python

La API estará disponible en http://localhost:8000.

- **Nota:** En http://localhost:8000/docs hay ejemplos de como se ejecutan los endpoints

