import matplotlib.pyplot as plt
from models.random_forest import evaluate_model, train_model, MODEL_PATH

DATA_PATH = "data/NU_Historical_Data.csv"

result = evaluate_model(DATA_PATH, days_out=10)
predictions = result["predictions"]
actual = result["actual"]

plt.figure(figsize=(10, 5))
plt.plot(actual, label="Actual", marker="o")
plt.plot(predictions, label="Predicción", marker="x")
plt.title("Predicciones vs Valores Reales")
plt.xlabel("Días")
plt.ylabel("Close")
plt.legend()
plt.grid(True)


plt.show()


