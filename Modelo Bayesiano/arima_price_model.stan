
data {
  int<lower=0> N;       // Longitud serie diferenciada
  vector[N] y;          // Serie diferenciada
  int<lower=0> H;       // Horizonte pronóstico
  real last_value;      // Último valor original para revertir diferenciación
}

parameters {
  vector[3] phi;  // Parámetros AR
    // Parámetros MA
  real<lower=0> sigma;
  real<lower=2> nu;
}

model {
  vector[N] mu;
  vector[N] e;
  
  // Inicialización
  for(t in 1:3) {
    mu[t] = y[t];
    e[t] = 0;
  }
  
  // Definir buffers para AR/MA 
  vector[3] buffer;
  vector[3] e_buffer;
  
  // Componente ARIMA
  for(t in 3:N) {
    // Actualizar buffers con valores previos
    for (i in 1:3) {
      if (t > i) {
        buffer[i] = y[t-i];
        e_buffer[i] = e[t-i];
      } else {
        buffer[i] = 0;
        e_buffer[i] = 0;
      }
    }
    
    mu[t] = dot_product(phi, buffer) ;
    e[t] = y[t] - mu[t];
  }
  
  // Prioris
  phi ~ student_t(3, 0, 0.5);
  
  sigma ~ gamma(0.01, 0.01);
  nu ~ gamma(2, 0.1);
  
  // Verosimilitud
  y[4:N] ~ student_t(nu, mu[4:N], sigma);
}

generated quantities {
  vector[H] y_hat_diff;  // Pronósticos en escala diferenciada
  vector[H] y_hat;       // Pronósticos en escala original
  
  // Buffers para componentes AR/MA
  vector[3] buffer = rep_vector(0, 3);
  vector[3] e_buffer = rep_vector(0, 3);
  
  // Inicializar buffers con los últimos valores de la serie
  for (i in 1:3) {
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
    y_hat_diff[h] = dot_product(phi, buffer)  + student_t_rng(nu, 0, sigma);
    
    // Revertir diferenciación
    y_hat[h] = current_value + y_hat_diff[h];
    current_value = y_hat[h];
    
    // Actualizar buffers
    if(3 > 0) {
      // Desplazar valores en los buffers (enfoque correcto para Stan)
      for (i in 3:2) {
        buffer[i] = buffer[i-1];
        e_buffer[i] = e_buffer[i-1];
      }
      buffer[1] = y_hat_diff[h];
      // Calculamos el error como la diferencia entre el valor predicho y el modelo
      real pred = dot_product(phi, buffer) ;
      e_buffer[1] = y_hat_diff[h] - pred;
    }
  }
}

