# Servicio de Modelo Random Forest

Este servicio contiene la implementación y recursos relacionados con el modelo Random Forest para predicción de series temporales financieras. Es un microservicio desarrollado con FastAPI y Celery para procesamiento asíncrono.

## Estructura de la carpeta

- `main.py`: API principal con FastAPI que expone endpoints para entrenamiento y predicción.
- `rf_model.py`: Implementación de la clase `TimeSeriesRandomForestModel` para series temporales.
- `train.py`: Funciones para entrenar el modelo con datos históricos.
- `forecast.py`: Funciones para realizar predicciones futuras con el modelo entrenado.
- `celery_app.py`: Configuración de Celery para procesamiento asíncrono de tareas.
- `tasks.py`: Tareas de Celery para entrenamiento asíncrono de modelos.
- `models/`: Carpeta donde se guardan los modelos entrenados (`.joblib`) y sus metadatos (`.json`).
- `requirements.txt`: Dependencias necesarias para ejecutar el servicio.
- `Dockerfile`: Configuración para contenedorización del servicio.

## Características del modelo

- **Modelo**: Random Forest con optimización de hiperparámetros usando RandomizedSearchCV
- **Series temporales**: Soporte nativo para datos temporales con características de lag
- **Selección de características**: Integración con SelectFromModel para optimización automática
- **Validación temporal**: TimeSeriesSplit para validación apropiada de series temporales
- **Predicción recursiva**: Capacidad de predicción futura con múltiples pasos
- **Métricas de evaluación**: MAE, MSE, RMSE, MAPE  

## API Endpoints

- **POST /train**: Iniciar entrenamiento asíncrono del modelo con datos históricos
- **GET /train/status/{job_id}**: Consultar el estado de una tarea de entrenamiento
- **POST /predict**: Realizar predicciones con modelo entrenado
- **GET /model/metrics**: Obtener métricas de evaluación del modelo

## Instalación y ejecución

### Requisitos previos
- Python 3.8+
- Redis (para Celery)

### Instalar dependencias
```bash
pip install -r requirements.txt
```

### Ejecutar el servicio
```bash
# Iniciar el servidor FastAPI
uvicorn main:app --host 0.0.0.0 --port 8002

# En otra terminal, iniciar el worker de Celery
celery -A celery_app.celery_app worker --loglevel=info --queue=rf_queue
```

### Usando Docker
```bash
# Construir la imagen
docker build -t rf-model-service .

# Ejecutar el contenedor
docker run -p 8002:8002 rf-model-service
```

## Dependencias principales

- **FastAPI**: Framework web para la API REST
- **Celery**: Sistema de colas de tareas para procesamiento asíncrono
- **Redis**: Broker de mensajes para Celery
- **scikit-learn**: Biblioteca de machine learning para Random Forest
- **pandas**: Manipulación y análisis de datos
- **numpy**: Operaciones numéricas
- **yfinance**: Descarga de datos financieros
- **joblib**: Serialización de modelos
- **statsmodels**: Análisis estadístico adicional

## Configuración de entorno

El servicio requiere las siguientes variables de entorno:
- `CELERY_BROKER_URL`: URL del broker Redis para Celery
- `CELERY_RESULT_BACKEND_URL_RF`: URL del backend de resultados Redis

## Arquitectura

El servicio sigue una arquitectura de microservicios con:

1. **API Layer** (`main.py`): Endpoints REST para interacción externa
2. **Business Logic** (`rf_model.py`, `train.py`, `forecast.py`): Lógica de negocio del modelo
3. **Task Queue** (`celery_app.py`, `tasks.py`): Procesamiento asíncrono de tareas pesadas
4. **Data Layer** (`models/`): Persistencia de modelos entrenados

## Consideraciones de rendimiento

- **Memoria**: Los modelos Random Forest pueden consumir memoria significativa con muchos árboles
- **Tiempo de entrenamiento**: El entrenamiento asíncrono permite operaciones no bloqueantes
- **Optimización**: Se recomienda usar `optimize_hyperparams=True` para mejor rendimiento
- **Datos**: Mínimo recomendado de 600 puntos de datos para entrenamiento efectivo

## Monitoreo y métricas

El servicio expone métricas de evaluación del modelo:

- **MAE (Mean Absolute Error)**: Error promedio absoluto
- **MSE (Mean Squared Error)**: Error cuadrático medio
- **RMSE (Root Mean Squared Error)**: Raíz del error cuadrático medio
- **MAPE (Mean Absolute Percentage Error)**: Error porcentual absoluto medio

## Solución de problemas

### Error de conexión a Redis
Verificar que Redis esté ejecutándose con el comando `redis-cli ping`

### Error de memoria insuficiente
- Reducir `n_estimators` en la configuración del modelo
- Aumentar `test_size` para reducir el conjunto de entrenamiento
- Limitar el rango de fechas de los datos

### Tareas Celery no se procesan
Verificar workers activos con `celery -A celery_app.celery_app inspect active` o purgar cola con `celery -A celery_app.celery_app purge`

## Estructura de archivos de modelo

Los modelos se guardan con la siguiente estructura:
```
models/
├── rf_model_TICKER.joblib          # Modelo entrenado
└── rf_model_TICKER_metadata.json   # Metadatos y métricas
```

## Contribuir

Para contribuir al desarrollo:

1. Sigue las convenciones de código PEP 8
2. Agrega tests para nuevas funcionalidades
3. Actualiza la documentación según corresponda
4. Asegúrate de que las pruebas pasen antes de hacer commit

## Licencia

Este proyecto es parte del sistema de predicción de series temporales. Consulta el archivo LICENSE en la raíz del proyecto para más detalles.
