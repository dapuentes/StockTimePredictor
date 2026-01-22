"""
Celery App Configuration for XGBoost Model Service
Handles asynchronous training tasks to prevent blocking the main API
"""
from celery import Celery
import os

# Redis URLs for broker and result backend
REDIS_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
RESULT_BACKEND_URL = os.getenv("CELERY_RESULT_BACKEND_URL_XGB", "redis://localhost:6379/3")

# No levantar error si estamos en desarrollo local sin Redis
if not REDIS_URL:
    REDIS_URL = "redis://localhost:6379/0"
if not RESULT_BACKEND_URL:
    RESULT_BACKEND_URL = "redis://localhost:6379/3"

celery_app = Celery(
    "xgb_worker",
    broker=REDIS_URL,
    backend=RESULT_BACKEND_URL,
    include=["model_xgb.tasks"]
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
        'train_xgb_model_task': {'queue': 'xgb_queue'}
    }
)
