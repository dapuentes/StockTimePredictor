"""
Celery App Configuration for Prophet Model Service
Handles asynchronous training tasks to prevent blocking the main API
"""
from celery import Celery
import os

# Redis URLs for broker and result backend
REDIS_URL = os.getenv("CELERY_BROKER_URL")
RESULT_BACKEND_URL = os.getenv("CELERY_RESULT_BACKEND_URL_PROPHET")

if not REDIS_URL:
    raise RuntimeError("CELERY_BROKER_URL no está definida en el entorno.")
if not RESULT_BACKEND_URL:
    raise RuntimeError("CELERY_RESULT_BACKEND_URL_PROPHET no está definida en el entorno.")

celery_app = Celery(
    "prophet_worker",
    broker=REDIS_URL,
    backend=RESULT_BACKEND_URL,
    include=["model_prophet.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="America/Bogota",
    enable_utc=True,
    worker_concurrency=os.getenv("CELERY_WORKER_CONCURRENCY", 2),
    worker_prefetch_multiplier=1,  # Para tareas largas como el entrenamiento
    task_acks_late=True,  # La tarea no se quita de la cola hasta que termine o falle
    task_routes={
        'train_prophet_model_task': {'queue': 'prophet_queue'}
    }
)
