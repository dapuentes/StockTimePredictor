FROM python:3.11-slim

WORKDIR /app

# Copiar archivos de utilidades compartidas
COPY backend/utils/ /app/utils/
COPY backend/services/model_prophet/ /app/services/model_prophet/

# Instalar dependencias
RUN pip install --no-cache-dir -r /app/services/model_prophet/requirements.txt

# Crear directorio para modelos
RUN mkdir -p /app/services/model_prophet/models

EXPOSE 8004

# Usar uvicorn para consistencia con otros servicios
CMD ["uvicorn", "services.model_prophet.app:app", "--host", "0.0.0.0", "--port", "8004"]
