FROM python:3.9-slim

WORKDIR /app


COPY backend/utils/ /app/utils/

COPY backend/services/model_prophet/ /app/services/model_prophet/

RUN pip install --no-cache-dir -r /app/services/model_prophet/requirements.txt

RUN mkdir -p /app/services/model_prophet/models

EXPOSE 8004

CMD ["uvicorn", "services.model_prophet.app:app", "--host", "0.0.0.0", "--port", "8004"]