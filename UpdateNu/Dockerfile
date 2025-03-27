# Imagen de base
FROM python:3.9-slim

# Directorio de trabajo
WORKDIR /app

# Copiar el archivo de requerimientos
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copiar el código
COPY NU_update.py /app/

# Ejecutar el script
CMD ["python", "NU_update.py"]