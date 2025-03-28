# Imagen de base
FROM python:3.10-slim

WORKDIR /app

# Copiar el archivo de requerimientos y el script de Python
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /app

# Exponer el puerto 8000
EXPOSE 8000

# Comando para ejecutar el script de Python
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]