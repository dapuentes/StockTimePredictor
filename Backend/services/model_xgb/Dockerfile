FROM python:3.9-slim

WORKDIR /app


COPY backend/utils/ /app/utils/

COPY backend/services/model_xgb/ /app/services/model_xgb/


RUN pip install --no-cache-dir -r /app/services/model_xgb/requirements.txt


RUN mkdir -p /app/services/model_xgb/models

EXPOSE 8003

CMD ["uvicorn", "services.model_xgb.main_xgb:app", "--host", "0.0.0.0", "--port", "8003"]