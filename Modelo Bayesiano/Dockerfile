# Se indica la imagen base de R
FROM rocker/r-ver:4.1.0

# Se instalan los paquetes necesarios
RUN apt-get update && apt-get install -y \
    libxml2-dev \
    libcurl4-openssl-dev \
    libssl-dev 

# Se establece el directorio de trabajo
WORKDIR /app

# Se copian los archivos necesarios
COPY . /app

# Se instalan los paquetes de R
RUN Rscript -e "install.packages(c('plumber','readr','xts','rstan'), repos='https://cloud.r-project.org')"

# Se expone el puerto 8000
EXPOSE 8000

# Se ejecuta el script de R
CMD ["Rscript", "api.R"]
