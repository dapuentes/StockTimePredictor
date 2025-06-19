import os
import sys
import pandas as pd
import argparse
from typing import Tuple, Dict

# --- Ajuste del sys.path para importaciones ---
# Esto permite que el script encuentre los módulos en el directorio 'utils' y otros.
# Se asume que el script se ejecuta desde el directorio raíz 'Backend/'.
# Si se ejecuta desde 'services/model_prophet/', el path relativo necesita ajustarse.
try:
    # Intenta importar asumiendo que el PYTHONPATH está correctamente configurado
    from utils.import_data import load_data
    from utils.preprocessing import PreprocessorFactory
    from services.model_prophet.prophet_model import ProphetModel
except ImportError:
    # Si falla, ajusta el path manualmente y reintenta
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
    if backend_dir not in sys.path:
        sys.path.append(backend_dir)
    from utils.import_data import load_data
    from utils.preprocessing import PreprocessorFactory
    from services.model_prophet.prophet_model import ProphetModel

def train_model(ticker: str,
                start_date: str,
                end_date: str,
                target_col: str = 'Close',
                train_size: float = 0.8,
                model_path_prefix: str = 'models/prophet_model') -> Tuple[ProphetModel, Dict[str, float]]:
    """
    Orquesta el pipeline completo de entrenamiento para el modelo Prophet.

    Args:
        ticker (str): El ticker de la acción a entrenar (ej. 'AAPL').
        start_date (str): Fecha de inicio para la carga de datos (YYYY-MM-DD).
        end_date (str): Fecha de fin para la carga de datos (YYYY-MM-DD).
        target_col (str): La columna objetivo a predecir.
        train_size (float): La proporción de datos a usar para el entrenamiento.
        model_path_prefix (str): El prefijo de la ruta para guardar los artefactos del modelo.

    Returns:
        Tuple[ProphetModel, Dict[str, float]]: Una tupla conteniendo el modelo entrenado
                                               y su diccionario de métricas.
    """
    print("--- Iniciando Pipeline de Entrenamiento para Prophet ---")

    # 1. Cargar Datos
    print(f"1. Cargando datos para {ticker} desde {start_date} hasta {end_date}...")
    try:
        data = load_data(ticker, start_date, end_date)
        if data.empty:
            raise ValueError("No se cargaron datos. Verifica el ticker y el rango de fechas.")
        print(f"Datos cargados exitosamente. Forma: {data.shape}")
    except Exception as e:
        print(f"Error al cargar datos: {e}")
        raise

    # 2. Preprocesar Datos
    print("2. Creando y aplicando el preprocesador de Prophet...")
    try:
        # Usar la fábrica para crear el preprocesador específico de Prophet
        prophet_preprocessor = PreprocessorFactory.create_preprocessor('prophet')
        
        # El método prepare_data del preprocesador se encarga de todo:
        # - Añadir regresores
        # - Convertir al formato ('ds', 'y')
        # - Copiar los regresores al DataFrame final
        prophet_data = prophet_preprocessor.prepare_data(data, target_col=target_col) # prepare datas no esta definido 
        print(f"Preprocesamiento completado. Forma de los datos para Prophet: {prophet_data.shape}")
        
    except Exception as e:
        print(f"Error durante el preprocesamiento: {e}")
        raise

    # 3. Dividir Datos en Entrenamiento y Prueba
    print(f"3. Dividiendo los datos (Train: {train_size*100}%, Test: {(1-train_size)*100}%)")
    train_idx = int(len(prophet_data) * train_size)
    train_df = prophet_data.iloc[:train_idx]
    test_df = prophet_data.iloc[train_idx:]
    print(f"  -> Datos de entrenamiento: {len(train_df)} filas")
    print(f"  -> Datos de prueba: {len(test_df)} filas")
    
    # Obtener la última fecha de entrenamiento para los metadatos
    training_end_date = train_df['ds'].max().strftime('%Y-%m-%d')

    # 4. Instanciar y Entrenar el Modelo
    print("4. Instanciando y entrenando el ProphetModel...")
    try:
        # Inyectar el preprocesador en la instancia del modelo para que se guarde junto a él
        model = ProphetModel(preprocessor=prophet_preprocessor)
        
        # Entrenar el modelo con el conjunto de entrenamiento (que ya está en el formato correcto)
        model.fit(train_df)
    except Exception as e:
        print(f"Error durante la instanciación o entrenamiento del modelo: {e}")
        raise

    # 5. Evaluar el Modelo
    print("5. Evaluando el modelo en el conjunto de prueba...")
    try:
        metrics = model.evaluate(test_df)
    except Exception as e:
        print(f"Error durante la evaluación: {e}")
        raise

    # 6. Guardar el Modelo y sus artefactos
    print("6. Guardando el modelo, preprocesador y metadatos...")
    try:
        # --- LÍNEA CORREGIDA ---
        # El prefijo que se recibe ya es el final y completo. No se añade el ticker aquí.
        model.save_model(model_path_prefix, training_end_date=training_end_date)
        print(f"Modelo guardado exitosamente con el prefijo: {os.path.basename(model_path_prefix)}")
    except Exception as e:
        print(f"Error al guardar el modelo: {e}")
        raise
        
    print("--- Pipeline de Entrenamiento de Prophet Finalizado Exitosamente ---")
    return model, metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script de entrenamiento para el modelo Prophet.")
    parser.add_argument("--ticker", type=str, required=True, help="Ticker de la acción (ej. 'AAPL').")
    parser.add_argument("--start-date", type=str, default="2020-01-01", help="Fecha de inicio (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, default=pd.Timestamp.now().strftime('%Y-%m-%d'), help="Fecha de fin (YYYY-MM-DD).")
    parser.add_argument("--train-size", type=float, default=0.8, help="Proporción para el conjunto de entrenamiento.")
    # La ruta por defecto asume que se ejecuta desde el directorio raíz del backend
    parser.add_argument("--model-path-prefix", type=str, default="services/model_prophet/models/prophet_model", help="Prefijo para guardar los archivos del modelo.")
    
    args = parser.parse_args()

    # Ejecutar el pipeline de entrenamiento con los argumentos proporcionados
    train_model(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        train_size=args.train_size,
        model_path_prefix=args.model_path_prefix
    )
