FROM python:3.11-slim

WORKDIR /app

COPY backend/utils/ /app/utils/
COPY backend/services/model_rf/ /app/services/model_rf/

RUN pip install --no-cache-dir -r /app/services/model_rf/requirements.txt

EXPOSE 8001

RUN mkdir -p /app/services/model_rf/models # create the models directory if it doesn't exist

CMD ["uvicorn", "services.model_rf.main:app", "--host", "0.0.0.0", "--port", "8001"]