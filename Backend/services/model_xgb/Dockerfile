FROM python:3.11-slim

WORKDIR /app

# Copiar archivos de utilidades compartidas
COPY backend/utils/ /app/utils/
COPY backend/services/model_xgb/ /app/services/model_xgb/

# Instalar dependencias
RUN pip install --no-cache-dir -r /app/services/model_xgb/requirements.txt

# Crear directorio para modelos
RUN mkdir -p /app/services/model_xgb/models

EXPOSE 8003

# Usar uvicorn para consistencia con otros servicios
CMD ["uvicorn", "services.model_xgb.main_xgb:app", "--host", "0.0.0.0", "--port", "8003"]
