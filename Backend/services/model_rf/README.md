# Carpeta `model_rf`

Esta carpeta contiene la implementación y los recursos relacionados con el modelo Random Forest utilizado en este proyecto.

## Estructura de la carpeta

- `model_rf.py`: Script principal con la implementación de Random Forest.
- `train.py`: Script para entrenar el modelo con los datos de entrada.
- `predict.py`: Permite realizar predicciones utilizando el modelo preentrenado.
- `main.py`: Script principal para ejecutar el flujo completo de entrenamiento y predicción. Este script es un microservicio que será utilizado por la api orquestadora `app.py`.
- `models/`: Carpeta donde se guardan los modelos entrenados (`.pkl`).
- `requirements.txt`: Dependencias necesarias para ejecutar el código.

## Cómo usar

### Entrenar el modelo
Ejecuta el siguiente comando para entrenar el modelo con un dataset específico:
```bash
python train.py --data_path data/dataset.csv --output_path models/model_rf.pkl
```

### Realizar predicciones
Cargar el modelo ya entrenado y realizar predicciones:
```python
from joblib import load

# Cargar modelo
model_rf = load("models/model_rf.pkl")

# Realizar predicciones
predictions = model_rf.predict([[0.5, 1.2, 3.5], [0.6, 1.1, 2.8]])
print(predictions)
```

## Parámetros del modelo

Por agregar más información sobre los parámetros del modelo y su ajuste.

## Dependencias necesarias

Asegúrate de instalar las siguientes librerías antes de ejecutar cualquier script:
- `scikit-learn`
- `numpy`
- `pandas`
- `joblib`

Utiliza el siguiente comando para instalar todas las librerías:
```bash
pip install -r requirements.txt
```

## Notas

Por agregar notas adicionales sobre el modelo, su rendimiento y cualquier otro detalle relevante.
