FROM python:3.11-slim

WORKDIR /app

COPY backend/api_gateway/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt


COPY backend/api_gateway/app.py /app/app.py

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]