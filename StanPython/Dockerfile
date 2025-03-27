# Se usa una imagen de Python en Debian
FROM python:3.10-bullseye

# Herramientas de compilación
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Instalación de dependencias
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copia de archivos al contenedor
COPY . .

# Puerto de la aplicación
EXPOSE 8000

# Comando de inicio
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]