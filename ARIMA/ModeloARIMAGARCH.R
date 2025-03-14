library(forecast)
library(readr)
modelo <- Arima(serie, order = c(3,1,3))

residuos <- modelo$residuals

# Prueba para residuos al cuadrado
Box.test(residuos^2, lag = 10, type = "Ljung-Box") # La prueba muestra que se rechaza la hipotesis nula de que los residuos son ruido blanco

rendimiento <- diff((serie))

library(rugarch)
spec <- ugarchspec(variance.model = list(model = "eGARCH", garchOrder = c(1,1)),
                   mean.model = list(armaOrder = c(2,2), include.mean = TRUE),
                   distribution.model = "std"
)
garch_fit <- ugarchfit(spec = spec, data = rendimiento)
garch_fit # Queda un modelo egarch(1,1) - arima(2,1,2), es normal que se reduzca el orden del ARMA
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
plot(forecast_garch, ylim = c(10,15))
