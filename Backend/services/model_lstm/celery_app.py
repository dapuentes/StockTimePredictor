from celery import Celery
import os

REDIS_URL = os.getenv("CELERY_BROKER_URL")
RESULT_BACKEND_URL = os.getenv("CELERY_RESULT_BACKEND_URL_LSTM") 

if not REDIS_URL:
    raise RuntimeError("CELERY_BROKER_URL no está definida en el entorno.")
if not RESULT_BACKEND_URL:
    raise RuntimeError("CELERY_RESULT_BACKEND_URL_LSTM no está definida en el entorno.")

celery_app = Celery(
    "lstm_worker", 
    broker=REDIS_URL,
    backend=RESULT_BACKEND_URL,
    include=["model_lstm.tasks"] 
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="America/Bogota", 
    enable_utc=True,
    worker_concurrency=os.getenv("CELERY_WORKER_CONCURRENCY", 1), # Número de workers concurrentes
    worker_prefetch_multiplier=1, # Para tareas largas como el entrenamiento
    task_acks_late=True, # Para que la tarea no se quite de la cola hasta que termine o falle
)
