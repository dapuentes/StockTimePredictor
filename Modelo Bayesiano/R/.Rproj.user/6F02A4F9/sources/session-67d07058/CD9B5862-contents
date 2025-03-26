# pronosticos.R
library(xts)
library(readr)
library(rstan)

# Asegúrate de cargar las funciones necesarias (por ejemplo, desde modelo.R)
source("R/modelo.R")      # Aquí se encuentra generate_arima_price_stan
source("R/pronosticos.R") # Aquí se encuentra forecast_arima_prices

# Cargar y preparar datos
data <- read_csv("NU_Historical_Data.csv")[-1,]
data$Date <- as.Date(data$Date, format = "%m/%d/%Y")
data$Close <- as.numeric(data$Close)
serie <- xts(data$Close, order.by = data$Date)["2021-12-09/2024-02-26"]

# Paso 1: Diferenciar serie
d <- 1
serie_diff <- diff(serie, differences = d)[-1]
serie_diff_vector <- as.numeric(coredata(serie_diff))

# Paso 2: Genera el modelo Stan a partir de la serie diferenciada
model_info <- generate_arima_price_stan(serie_diff_vector)

# Paso 3: Ajusta el modelo con los datos correctos
fit <- stan(
  file = model_info$stan_file,
  data = list(
    N = length(serie_diff_vector),
    y = serie_diff_vector,
    H = 10,
    last_value = as.numeric(tail(serie, 1))
  ),
  iter = 2000,
  chains = 4
)

# Paso 4: Pronosticar
pronosticos <- forecast_arima_prices(fit, serie, d = 1, horizon = 10)
print(pronosticos)
