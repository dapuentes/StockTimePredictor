library(forecast)
library(readr)
NU_Historical_Data <- read_csv("NU Historical Data.csv")
data <- data.frame(NU_Historical_Data$Date, NU_Historical_Data$Price)
data$NU_Historical_Data.Date <- as.Date(data$NU_Historical_Data.Date, format = "%m/%d/%Y")

library(xts)
serie_prueba <- xts(data[, -1], order.by = data$NU_Historical_Data.Date)
serie <- serie_prueba["2021-12-10/2024-10-08"] #Eliminando los ultimos 10 dias de la serie

library(urca)
df_prueba <- ur.df(serie, type = "trend", lags = 1)
summary(df_prueba)

serie_diferenciada <- na.omit(diff(serie))
df_prueba_diff <- ur.df(serie_diferenciada, type = "none", lags = 1)
summary(df_prueba_diff)

modelo <- Arima(serie, order = c(3,1,3))

residuos <- modelo$residuals

# Prueba para residuos al cuadrado
Box.test(residuos^2, lag = 10, type = "Ljung-Box") # La prueba muestra que se rechaza la hipotesis nula de que los residuos son ruido blanco

library(rugarch)
spec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1,1)),
                   mean.model = list(armaOrder = c(3,2), include.mean = TRUE),
                   distribution.model = "std",
                   fixed.pars = list(omega = 0)
)
garch_fit <- ugarchfit(spec = spec, data = serie)
garch_fit
plot(garch_fit)

# Verificando si los residuos estandarizados son ruido blanco
residuos_estandarizados <- residuals(garch_fit, standardize = TRUE)
Box.test(residuos_estandarizados^2, lag = 10, type = "Ljung-Box") # La prueba muestra que se rechaza la hipotesis nula de que los residuos son ruido blanco

acf(residuos_estandarizados)
pacf(residuos_estandarizados)
FinTS::ArchTest(residuos_estandarizados, lags = 10)


# Pronostico
forecast_garch <- ugarchforecast(garch_fit, n.ahead = 10)
volatilidad <- forecast_garch@forecast$sigmaFor

pronostico <- forecast_garch@forecast$seriesFor

plot(volatilidad)
plot(pronostico)
plot(forecast_garch, ylim = c(14,15))
