data {
  int<lower=1> N;                // Número de observaciones
  vector[N] y;                   // Serie de tiempo
  int<lower=1> H;                // Horizonte de predicción
}

parameters {
  real mu;                       // Media de la serie
  real<lower=-1, upper=1> ar1;   // Parámetro AR(1)
  real<lower=-1, upper=1> ar2;   // Parámetro AR(2)
  real<lower=-1, upper=1> ar3;   // Parámetro AR(3)
  real<lower=-1, upper=1> ma1;   // Parámetro MA(1)
  real<lower=-1, upper=1> ma2;   // Parámetro MA(2)
  real<lower=0, upper=1> alpha1; // Parámetro GARCH(1,1) - alpha
  real<lower=0, upper=1-alpha1> beta1; // Parámetro GARCH(1,1) - beta
  real<lower=2> nu;              // Grados de libertad para t-student
  real<lower=0> sigma;           // Escala inicial para la varianza condicional
}

transformed parameters {
  vector[N] h;                   // Varianza condicional
  h[1] = sigma^2;                // Condición inicial de la varianza

  for (t in 2:N) {
    h[t] = alpha1 * pow(y[t-1] - mu, 2) + beta1 * h[t-1]; // Dinámica GARCH(1,1)
  }
}

model {
  // Priori para los parámetros (basados en rugarch)
  mu ~ normal(10.897, 0.255);    // Prior para mu
  ar1 ~ normal(1.503, 0.002);    // Prior para AR(1)
  ar2 ~ normal(-0.686, 0.003);   // Prior para AR(2)
  ar3 ~ normal(0.181, 0.003);    // Prior para AR(3)
  ma1 ~ normal(-0.510, 0.037);   // Prior para MA(1)
  ma2 ~ normal(0.131, 0.035);    // Prior para MA(2)
  alpha1 ~ normal(0.047, 0.010); // Prior para alpha (GARCH)
  beta1 ~ normal(0.952, 0.010);  // Prior para beta (GARCH)
  nu ~ gamma(7.4, 1.0);          // Prior para nu (t-student)
  sigma ~ cauchy(0, 5);          // Prior para la escala inicial

  // Modelo de la media (ARIMA(3,0,2)) y la varianza condicional (GARCH(1,1))
  for (t in 4:N) {
    real mean_t = mu 
                  + ar1 * y[t-1] 
                  + ar2 * y[t-2] 
                  + ar3 * y[t-3] 
                  + ma1 * (y[t-1] - mu) 
                  + ma2 * (y[t-2] - mu);
    y[t] ~ student_t(nu, mean_t, sqrt(h[t])); // t-student con varianza condicional
  }
}

generated quantities {
  vector[H] y_forecast;        // Predicciones de la serie original

  // Genera predicciones para la serie original
  for (h_pred in 1:H) {
    real mean_t = mu 
                  + ar1 * y[N - h_pred + 1]  // Último valor observado y el AR de la serie
                  + ar2 * y[N - h_pred]
                  + ar3 * y[N - h_pred - 1]
                  + ma1 * (y[N - h_pred + 1] - mu)
                  + ma2 * (y[N - h_pred] - mu);
    y_forecast[h_pred] = student_t_rng(nu, mean_t, sqrt(alpha1 + beta1));  // Predicción de la serie original
  }
}