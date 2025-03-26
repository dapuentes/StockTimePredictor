library(readr)
library(xts)
library(rstan)

# Verificar que los archivos de origen existan
if (!file.exists("R/ModeloStan.R")) {
  stop("El archivo R/ModeloStan.R no existe")
}

# Cargar funciones desde los módulos organizados
tryCatch({
  source("R/ModeloStan.R")      # Contiene generate_arima_price_stan()
}, error = function(e) {
  stop(paste("Error al cargar los archivos de origen:", e$message))
})

#* @apiTitle API para Pronósticos ARIMA con Stan
#* Endpoint para generar el modelo Stan a partir de una serie diferenciada
#* Genera el código Stan a partir de una serie diferenciada
#* @param serie Un vector de números en formato "c(1,2,3,4,5)"
#* @get /generate_stan
function(serie = "c(1,2,3,4,5)") {
  tryCatch({
    serie_diff <- eval(parse(text = serie))
    
    # Validar que serie_diff sea un vector numérico
    if (!is.numeric(serie_diff)) {
      stop("La serie debe ser un vector numérico")
    }
    
    result <- generate_arima_price_stan(serie_diff)
    return(result)
  }, error = function(e) {
    return(list(error = as.character(e)))
  })
}

#* Endpoint para ajustar el modelo y pronosticar
#* Realiza el ajuste del modelo y pronostica
#* @param horizon Horizonte de pronóstico (número de períodos)
#* @param start_date Fecha de inicio del conjunto de datos (opcional)
#* @param end_date Fecha de fin del conjunto de datos (opcional)
#* @get /forecast
function(horizon = 10, start_date = "2021-12-09", end_date = "2024-02-26") {
  tryCatch({
    # Verificar que el archivo CSV exista
    csv_path <- "data/NU_Historical_Data.csv"
    if (!file.exists(csv_path)) {
      stop("Archivo CSV no encontrado")
    }
    
    # Leer y procesar datos
    data <- read_csv(csv_path)[-1,]
    data$Date <- as.Date(data$Date, format = "%m/%d/%Y")
    data$Close <- as.numeric(data$Close)
    
    # Filtrar por rango de fechas
    serie <- xts(data$Close, order.by = data$Date)[paste0(start_date, "/", end_date)]
    
    # Validaciones
    if (nrow(serie) == 0) {
      stop("No hay datos en el rango de fechas especificado")
    }
    
    # Diferenciar la serie (d = 1 por defecto)
    d <- 1
    serie_diff <- diff(serie, differences = d)[-1]
    serie_diff_vector <- as.numeric(coredata(serie_diff))
    
    # Generar el modelo Stan
    model_info <- generate_arima_price_stan(serie_diff_vector)
    
    # Ajustar el modelo usando rstan
    fit <- stan(
      file = model_info$stan_file,
      data = list(
        N = length(serie_diff_vector),
        y = serie_diff_vector,
        H = as.numeric(horizon),
        last_value = as.numeric(tail(serie, 1))
      ),
      iter = 2000,
      chains = 4
    )
    
    # Realizar pronósticos
    pronosticos <- forecast_arima_prices(fit, serie, d = d, horizon = as.numeric(horizon))
    
    # Convertir el objeto xts a formato serializable: data.frame y vector de fechas
    result_df <- as.data.frame(pronosticos)
    result_dates <- as.character(index(pronosticos))
    
    return(list(
      forecast = result_df,
      dates = result_dates,
      metadata = list(
        start_date = start_date,
        end_date = end_date,
        horizon = horizon,
        total_observations = nrow(serie)
      )
    ))
  }, error = function(e) {
    return(list(
      error = as.character(e),
      error_details = list(
        start_date = start_date,
        end_date = end_date,
        horizon = horizon
      )
    ))
  })
}
