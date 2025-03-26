generate_arima_price_stan <- function(serie_diff, output_file = "arima_price_model.stan") {
  
  require(forecast)
  
  # Identificar modelo ARIMA en la serie diferenciada
  arima_model <- auto.arima(serie_diff, seasonal = FALSE)
  ar_order <- arima_model$arma[1]
  ma_order <- arima_model$arma[2]
  max_order <- max(ar_order, ma_order)
  
  # Se convierten los valores a cadenas para su uso en el código Stan
  max_order_str <- as.character(max_order)
  max_order_plus_one_str <- as.character(max_order + 1)
  
  stan_code <- sprintf('
data {
  int<lower=0> N;       // Longitud serie diferenciada
  vector[N] y;          // Serie diferenciada
  int<lower=0> H;       // Horizonte pronóstico
  real last_value;      // Último valor original para revertir diferenciación
}

parameters {
  %s  // Parámetros AR
  %s  // Parámetros MA
  real<lower=0> sigma;
  real<lower=2> nu;
}

model {
  vector[N] mu;
  vector[N] e;
  
  // Inicialización
  for(t in 1:%s) {
    mu[t] = y[t];
    e[t] = 0;
  }
  
  // Definir buffers para AR/MA 
  vector[%s] buffer;
  vector[%s] e_buffer;
  
  // Componente ARIMA
  for(t in %s:N) {
    // Actualizar buffers con valores previos
    for (i in 1:%s) {
      if (t > i) {
        buffer[i] = y[t-i];
        e_buffer[i] = e[t-i];
      } else {
        buffer[i] = 0;
        e_buffer[i] = 0;
      }
    }
    
    mu[t] = %s;
    e[t] = y[t] - mu[t];
  }
  
  // Prioris
  %s
  %s
  sigma ~ gamma(0.01, 0.01);
  nu ~ gamma(2, 0.1);
  
  // Verosimilitud
  y[%s:N] ~ student_t(nu, mu[%s:N], sigma);
}

generated quantities {
  vector[H] y_hat_diff;  // Pronósticos en escala diferenciada
  vector[H] y_hat;       // Pronósticos en escala original
  
  // Buffers para componentes AR/MA
  vector[%s] buffer = rep_vector(0, %s);
  vector[%s] e_buffer = rep_vector(0, %s);
  
  // Inicializar buffers con los últimos valores de la serie
  for (i in 1:%s) {
    if (i <= N) {
      buffer[i] = y[N-i+1];
      // Simplemente usamos los valores observados para e_buffer inicial
      // sin intentar usar mu que no está disponible aquí
      e_buffer[i] = 0; // Inicializamos a cero ya que no podemos acceder a mu
    }
  }
  
  real current_value = last_value;
  
  for(h in 1:H) {
    // Generar pronóstico diferenciado
    y_hat_diff[h] = %s + student_t_rng(nu, 0, sigma);
    
    // Revertir diferenciación
    y_hat[h] = current_value + y_hat_diff[h];
    current_value = y_hat[h];
    
    // Actualizar buffers
    if(%s > 0) {
      // Desplazar valores en los buffers (enfoque correcto para Stan)
      for (i in %s:2) {
        buffer[i] = buffer[i-1];
        e_buffer[i] = e_buffer[i-1];
      }
      buffer[1] = y_hat_diff[h];
      // Calculamos el error como la diferencia entre el valor predicho y el modelo
      real pred = %s;
      e_buffer[1] = y_hat_diff[h] - pred;
    }
  }
}
',
if(ar_order > 0) paste0("vector[", ar_order, "] phi;") else "",
if(ma_order > 0) paste0("vector[", ma_order, "] theta;") else "",
max_order_str,
max_order_str,
max_order_str,
max_order_str,
max_order_str,
if(ar_order + ma_order > 0) {
  paste(
    if(ar_order > 0) "dot_product(phi, buffer)" else "0",
    if(ma_order > 0) " + dot_product(theta, e_buffer)" else ""
  )
} else "0",
if(ar_order > 0) "phi ~ student_t(3, 0, 0.5);" else "",
if(ma_order > 0) "theta ~ student_t(3, 0, 0.5);" else "",
max_order_plus_one_str,
max_order_plus_one_str,
max_order_str,
max_order_str,
max_order_str,
max_order_str,
max_order_str,
if(ar_order + ma_order > 0) {
  paste(
    if(ar_order > 0) "dot_product(phi, buffer)" else "0",
    if(ma_order > 0) " + dot_product(theta, e_buffer)" else ""
  )
} else "0",
max_order_str,
max_order_str,
if(ar_order + ma_order > 0) {
  paste(
    if(ar_order > 0) "dot_product(phi, buffer)" else "0",
    if(ma_order > 0) " + dot_product(theta, e_buffer)" else ""
  )
} else "0"
  )
  writeLines(stan_code, output_file)
  return(list(
    ar_order = ar_order,
    ma_order = ma_order,
    stan_file = output_file
  ))
}

# Generando la funcion de pronostico

forecast_arima_prices <- function(fit, original_series, d = 1, horizon = 10) {
  require(xts)
  
  # Diferenciar serie
  serie_diff <- diff(original_series, differences = d)[-1]
  last_value <- tail(original_series, 1)
  
  # Ajustar modelo
  samples <- extract(fit)
  
  # Calcular pronósticos
  forecasts <- apply(samples$y_hat, 2, function(x) {
    c(
      mean = mean(x),
      lower = quantile(x, 0.05),
      upper = quantile(x, 0.95)
    )
  })
  
  # Crear serie temporal
  forecast_dates <- seq(end(original_series), by = "day", length.out = horizon + 1)[-1]
  
  xts(
    t(forecasts),
    order.by = forecast_dates
  )
}


