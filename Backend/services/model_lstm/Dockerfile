FROM python:3.11-slim

WORKDIR /app

# Copiar archivos de utilidades compartidas
COPY backend/utils/ /app/utils/
COPY backend/services/model_lstm/ /app/services/model_lstm/

# Instalar dependencias
RUN pip install --no-cache-dir -r /app/services/model_lstm/requirements.txt

# Crear directorio para modelos
RUN mkdir -p /app/services/model_lstm/models

EXPOSE 8002

# Usar uvicorn para consistencia con otros servicios
CMD ["uvicorn", "services.model_lstm.main:app", "--host", "0.0.0.0", "--port", "8002"]