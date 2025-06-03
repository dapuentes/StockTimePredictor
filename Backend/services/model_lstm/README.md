# Servicio de Modelo LSTM

Este servicio contiene la implementación y recursos relacionados con el modelo LSTM (Long Short-Term Memory) para predicción de series temporales financieras. Es un microservicio desarrollado con FastAPI y Celery para procesamiento asíncrono.

## Estructura de la carpeta

- `main.py`: API principal con FastAPI que expone endpoints para entrenamiento y predicción.
- `lstm_model.py`: Implementación de la clase `TimeSeriesLSTMModel` para series temporales.
- `train.py`: Funciones para entrenar el modelo con datos históricos.
- `forecast.py`: Funciones para realizar predicciones futuras con el modelo entrenado.
- `celery_app.py`: Configuración de Celery para procesamiento asíncrono de tareas.
- `tasks.py`: Tareas de Celery para entrenamiento asíncrono de modelos.
- `models/`: Carpeta donde se guardan los modelos entrenados y sus metadatos.
- `requirements.txt`: Dependencias necesarias para ejecutar el servicio.
- `Dockerfile`: Configuración para contenedorización del servicio.

## Características del modelo

- **Modelo**: LSTM con optimización de hiperparámetros usando Keras Tuner
- **Series temporales**: Soporte nativo para datos temporales con secuencias
- **Regularización**: Dropout, Batch Normalization y Early Stopping
- **Estimación de incertidumbre**: Monte Carlo Dropout para intervalos de confianza
- **Soporte GPU**: Optimizado para entrenamiento con TensorFlow GPU
- **Métricas de evaluación**: MAE, MSE, RMSE, MAPE

## API Endpoints

- **POST /train**: Iniciar entrenamiento asíncrono del modelo LSTM
- **GET /training_status/{job_id}**: Consultar el estado de una tarea de entrenamiento
- **GET /predict**: Realizar predicciones con intervalos de confianza
- **GET /models**: Lista de modelos disponibles
- **GET /health**: Estado de salud del servicio

## Instalación y ejecución

### Requisitos previos
- Python 3.8+
- TensorFlow 2.13+
- Redis (para Celery)
- GPU recomendada para entrenamiento

### Instalar dependencias
```bash
pip install -r requirements.txt
```

### Ejecutar el servicio
```bash
# Iniciar el servidor FastAPI
uvicorn main:app --host 0.0.0.0 --port 8002

# En otra terminal, iniciar el worker de Celery
celery -A celery_app.celery_app worker --loglevel=info
```

### Usando Docker
```bash
# Construir la imagen
docker build -t lstm-model-service .

# Ejecutar el contenedor
docker run -p 8002:8002 lstm-model-service
```
## Dependencias principales

- **TensorFlow**: Framework de deep learning para modelos LSTM
- **Keras**: API de alto nivel para construcción de redes neuronales
- **Keras Tuner**: Optimización automática de hiperparámetros
- **FastAPI**: Framework web para la API REST
- **Celery**: Sistema de colas de tareas para procesamiento asíncrono
- **Redis**: Broker de mensajes para Celery
- **pandas**: Manipulación y análisis de datos
- **numpy**: Operaciones numéricas
- **scikit-learn**: Herramientas de machine learning complementarias

## Configuración de entorno

El servicio requiere las siguientes variables de entorno:
- `CELERY_BROKER_URL`: URL del broker Redis para Celery
- `CELERY_RESULT_BACKEND_URL`: URL del backend de resultados Redis

## Arquitectura

El servicio sigue una arquitectura de microservicios con:

1. **API Layer** (`main.py`): Endpoints REST para interacción externa
2. **Business Logic** (`lstm_model.py`, `train.py`, `forecast.py`): Lógica de negocio del modelo
3. **Task Queue** (`celery_app.py`, `tasks.py`): Procesamiento asíncrono de tareas pesadas
4. **Data Layer** (`models/`): Persistencia de modelos entrenados

## Consideraciones de rendimiento

- **GPU**: Se recomienda usar GPU para acelerar el entrenamiento
- **Memoria**: Los modelos LSTM pueden requerir memoria significativa para secuencias largas
- **Tiempo de entrenamiento**: El entrenamiento asíncrono permite operaciones no bloqueantes
- **Optimización**: Se recomienda usar `optimize_params=True` para mejor rendimiento
- **Datos**: Mínimo recomendado de 500 puntos de datos para entrenamiento efectivo

## Características técnicas

- **Regularización**: Dropout y regularización L2 para prevenir sobreajuste
- **Normalización**: Batch Normalization para estabilizar el entrenamiento
- **Optimización**: Adam optimizer con gradient clipping
- **Incertidumbre**: Monte Carlo Dropout para estimación de incertidumbre
- **Escalado**: Normalización automática de características y objetivo
- **Validación**: Early stopping basado en pérdida de validación

## Solución de problemas

### Error de conexión a Redis
Verificar que Redis esté ejecutándose con el comando `redis-cli ping`

### Error de memoria insuficiente
- Reducir `sequence_length` en la configuración del modelo
- Reducir `lstm_units` para disminuir el tamaño del modelo
- Aumentar `train_size` para reducir el conjunto de entrenamiento

### Problemas con GPU
Verificar instalación de CUDA y drivers de GPU compatibles con TensorFlow

### Tareas Celery no se procesan
Verificar workers activos con `celery -A celery_app.celery_app inspect active` o purgar cola con `celery -A celery_app.celery_app purge`

## Contribuir

Para contribuir al desarrollo:

1. Sigue las convenciones de código PEP 8
2. Agrega tests para nuevas funcionalidades
3. Actualiza la documentación según corresponda
4. Asegúrate de que las pruebas pasen antes de hacer commit

## Licencia

Este proyecto es parte del sistema de predicción de series temporales. Consulta el archivo LICENSE en la raíz del proyecto para más detalles.