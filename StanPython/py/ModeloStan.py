# Librerias
import pmdarima as pm
import numpy as np
import pandas as pd
from pmdarima.arima import ARIMA

def generate_arima_price_stan(serie_diff, output_file="arima_price_model.stan"):
    """
    Genera un modelo Stan para pronóstico ARIMA
    
    Parámetros:
    - serie_diff: Serie diferenciada de precios
    - output_file: Archivo de salida para el modelo Stan
    """

    if len(serie_diff) == 0:
        raise ValueError("La serie diferenciada no puede estar vacía")

    try:
        # Identificar el orden del modelo ARIMA
        '''arima_model = pm.auto_arima(
            serie_diff,
            seasonal=False,  # No se considera estacionalidad por ser serie financiera
            stepwise=True,
            suppress_warnings=True
        )'''
        arima_model = ARIMA(order = (3,1,0), seasonal = False)

        print("Modelo ARIMA encontrado:")
        print(arima_model)

        # Extraer los coeficientes del modelo ARIMA
        ar_order = max(0, arima_model.order[0])
        ma_order = max(0, arima_model.order[2])
        max_order = max(ar_order, ma_order)

        max_order_plus_one = max_order + 1

        # Construcción de la parte AR + MA para mu[t]
        ar_part = "dot_product(phi, buffer)" if ar_order > 0 else "0"
        ma_part = "dot_product(theta, e_buffer)" if ma_order > 0 else "0"
        ar_ma_expr = (
            f"{ar_part} + {ma_part}" if ar_order > 0 and ma_order > 0
            else (ar_part if ar_order > 0 else (ma_part if ma_order > 0 else "0"))
        )

        # Construimos el código Stan asegurando la sintaxis correcta en los bucles
        stan_code = f"""
    data {{
    int<lower=0> N;       // Longitud serie diferenciada
    vector[N] y;          // Serie diferenciada
    int<lower=0> H;       // Horizonte pronóstico
    real last_value;      // Último valor original para revertir diferenciación
    }}

    parameters {{
    {"vector[" + str(ar_order) + "] phi;" if ar_order > 0 else ""}
    {"vector[" + str(ma_order) + "] theta;" if ma_order > 0 else ""}
    real<lower=0> sigma;
    real<lower=2> nu;
    }}

    model {{
    vector[N] mu;
    vector[N] e;

    // Inicialización
    for(t in 1:{max_order}) {{
        mu[t] = y[t];
        e[t] = 0;
    }}

    // Definir buffers para AR/MA
    vector[{max_order}] buffer;
    vector[{max_order}] e_buffer;

    // Componente ARIMA
    for(t in {max_order_plus_one}:N) {{
        // Actualizar buffers con valores previos
        for(i in 1:{max_order}) {{
        if(t > i) {{
            buffer[i] = y[t - i];
            e_buffer[i] = e[t - i];
        }} else {{
            buffer[i] = 0;
            e_buffer[i] = 0;
        }}
        }}

        mu[t] = {ar_ma_expr};
        e[t] = y[t] - mu[t];
    }}

    // Prioris
    {"phi ~ student_t(3, 0, 0.5);" if ar_order > 0 else ""}
    {"theta ~ student_t(3, 0, 0.5);" if ma_order > 0 else ""}
    sigma ~ gamma(0.01, 0.01);
    nu ~ gamma(2, 0.1);

    // Verosimilitud
    y[{max_order_plus_one}:N] ~ student_t(nu, mu[{max_order_plus_one}:N], sigma);
    }}

    generated quantities {{
    vector[H] y_hat_diff;  // Pronósticos en escala diferenciada
    vector[H] y_hat;       // Pronósticos en escala original

    // Buffers para componentes AR/MA
    vector[{max_order}] buffer = rep_vector(0, {max_order});
    vector[{max_order}] e_buffer = rep_vector(0, {max_order});

    // Inicializar buffers con los últimos valores de la serie
    for(i in 1:{max_order}) {{
        if(i <= N) {{
        buffer[i] = y[N - i + 1];
        e_buffer[i] = 0; // inicializamos a 0
        }}
    }}

    real current_value = last_value;

    for(h in 1:H) {{
        // Generar pronóstico diferenciado
        y_hat_diff[h] = {ar_ma_expr} + student_t_rng(nu, 0, sigma);

        // Revertir diferenciación
        y_hat[h] = current_value + y_hat_diff[h];
        current_value = y_hat[h];

        if({max_order} > 0) {{
        // Desplazar valores en los buffers
        for(i in {max_order}:2) {{
            buffer[i] = buffer[i - 1];
            e_buffer[i] = e_buffer[i - 1];
        }}
        buffer[1] = y_hat_diff[h];
        real pred = {ar_ma_expr};
        e_buffer[1] = y_hat_diff[h] - pred;
        }}
    }}
    }}
    """
    
        # Guardar el modelo Stan en un archivo
        with open(output_file, "w") as f:
            f.write(stan_code)

        return {
            "ar_order": ar_order,
            "ma_order": ma_order,
            "stan_file": output_file
        }

    except Exception as e:
        print(f"Error generando modelo Stan: {e}")
        raise

# Se generan pronosticos
def forecast_arima_prices(fit, original_series, d = 1, horizon = 10):
    """
    Genera pronósticos de precios a partir de un modelo ARIMA
    
    Parámetros:
    - fit: Resultado de ajuste de modelo Stan
    - original_series: Serie de precios original
    - d: Orden de diferenciación
    - horizon: Horizonte de pronóstico
    """
    

    # Muestras
    samples = fit.to_frame()

    # Filtrando las muestras de y_hat
    y_hat_cols = sorted([col for col in samples.columns
                         if col.startswith("y_hat") and not col.startswith("y_hat_diff")],
                        key = lambda x: int(x.split('.')[1])
    )
    
    y_hat_samples = samples[y_hat_cols[:horizon]].values # forma (n_samples, horizon)
    for i in y_hat_samples:
        print(i)

    # Pronósticos
    forecast_data = {
        "mean": np.mean(y_hat_samples, axis = 0),
        "lower": np.percentile(y_hat_samples, 5, axis = 0),
        "upper": np.percentile(y_hat_samples, 95, axis = 0)
    }

    # Fechas
    last_date = original_series.index[-1]
    forecast_dates = pd.date_range(start = last_date, periods = horizon + 1, freq = 'D')[1:]

    # DataFrame de pronósticos
    df_forecast = pd.DataFrame(forecast_data, index = forecast_dates)

    return df_forecast