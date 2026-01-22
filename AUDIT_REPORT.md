# 📊 Reporte de Auditoría de Código - StockTimePredictor

**Fecha:** 11 de Diciembre, 2025  
**Auditor:** GitHub Copilot (Arquitecto de Software Senior - MLOps)  
**Proyecto:** Aplicación Full Stack para Análisis de Series de Tiempo

---

## 1. Arquitectura Actual

### 1.1 Stack Tecnológico Detectado

| Capa | Tecnología | Versión/Detalles |
|------|------------|------------------|
| **API Gateway** | FastAPI + httpx | Proxy asíncrono hacia microservicios |
| **Microservicios ML** | FastAPI (RF, LSTM, XGBoost, Prophet) | 4 servicios independientes |
| **Cola de Tareas** | Celery + Redis | Solo RF y LSTM configurados |
| **Frontend** | React 18 + Ant Design + Chart.js | TanStack Query para mutaciones |
| **Orquestación** | Docker Compose | Multi-contenedor con GPU support |

### 1.2 Flujo de Datos

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────────────────────┐
│   Frontend  │────▶│  API Gateway │────▶│ Microservicios (RF/LSTM/XGB/P) │
│  (React)    │     │  (FastAPI)   │     │                                 │
│  :3000      │     │  :8000       │     │  :8001 / :8002 / :8003 / :8004  │
└─────────────┘     └──────────────┘     └─────────────────────────────────┘
                                                       │
                           ┌───────────────────────────┘
                           ▼
                    ┌─────────────┐     ┌────────────────┐
                    │   Redis     │────▶│ Celery Workers │
                    │   :6379     │     │  (RF / LSTM)   │
                    └─────────────┘     └────────────────┘
```

### 1.3 Estructura de Microservicios

| Servicio | Puerto | Entry Point | Celery Worker | Estado |
|----------|--------|-------------|---------------|--------|
| Random Forest | 8001 | `model_rf/main.py` | ✅ Configurado | Productivo |
| LSTM | 8002 | `model_lstm/main.py` | ✅ Configurado | Productivo |
| XGBoost | 8003 | `model_xgb/main_xgb.py` | ❌ Sin configurar | ⚠️ Bloqueante |
| Prophet | 8004 | `model_prophet/app.py` | ❌ Sin configurar | ⚠️ Bloqueante |

---

## 2. Riesgos Detectados

### 🔴 Alta Prioridad (Críticos)

#### 2.1 **ERROR CRÍTICO: Endpoints de Entrenamiento Síncronos (XGBoost y Prophet)**

**Archivo:** `Backend/services/model_xgb/main_xgb.py`  
**Línea:** 186-264 (`def train_model()`)

**Archivo:** `Backend/services/model_prophet/app.py`  
**Línea:** 114-176 (`def train_model()`)

**Problema:**  
Los endpoints `/train` de XGBoost y Prophet son **funciones síncronas** que bloquean el hilo principal del servidor Uvicorn durante todo el proceso de entrenamiento (puede durar minutos).

```python
# XGBoost - main_xgb.py:186 - ❌ BLOQUEANTE
@app.get("/train") 
def train_model():  # <-- No es async, bloquea el servidor
    ...
    model = XGBoostModel()
    model.fit(X_train_scaled, y_train_scaled)  # <-- Operación pesada síncrona
    ...

# Prophet - app.py:114 - ❌ BLOQUEANTE
@app.get('/train')
def train_model():  # <-- No es async, bloquea el servidor
    ...
    model, metrics, _ = prophet_service.train(...)  # <-- Operación pesada síncrona
    ...
```

**Impacto:**  
- El servidor no puede procesar otras requests mientras entrena
- Timeout del API Gateway (30s) puede ocurrir antes de completar
- La UI del frontend se "congela" esperando respuesta

**Contraste con RF/LSTM (implementación correcta):**
```python
# RF - main.py:258 - ✅ NO BLOQUEANTE
@app.post("/train", status_code=202)
async def train_model(request: TrainRequest):
    from .tasks import train_rf_model_task
    task = train_rf_model_task.delay(request.model_dump())  # <-- Delega a Celery
    return {"job_id": task.id, "status": "queued", ...}
```

---

#### 2.2 **ERROR CRÍTICO: Predicción Síncrona en Todos los Microservicios**

**Archivos:**  
- `Backend/services/model_rf/main.py` líneas 300-510 (`async def predict()`)
- `Backend/services/model_lstm/main.py` líneas 393-490 (`async def predict()`)
- `Backend/services/model_xgb/main_xgb.py` líneas 269-389 (`async def predict()`)
- `Backend/services/model_prophet/app.py` líneas 209-236 (`def predict()`)

**Problema:**  
Aunque los endpoints de predicción están decorados con `async`, ejecutan operaciones CPU-bound síncronamente:

```python
# RF - main.py:443 - ⚠️ PSEUDO-ASYNC
async def predict(...):
    ...
    # Esta llamada es síncrona y bloquea el event loop
    forecast, lower_bounds, upper_bounds = forecast_future_prices(
        model=model,
        data=data.copy(),
        forecast_horizon=forecast_horizon,
        target_col=target_col
    )  # <-- Bloquea el thread principal
```

**Impacto:**  
- Aunque usa `async def`, el cálculo de predicción bloquea el event loop de asyncio
- Con múltiples requests concurrentes, se forma un cuello de botella
- Timeout de 420 segundos en API Gateway indica que ya hay problemas conocidos

---

### 🟡 Media Prioridad

#### 2.3 **Payload JSON Potencialmente Grande**

**Archivo:** `Backend/services/model_rf/tasks.py` líneas 55-68

**Problema:**  
El resultado del entrenamiento incluye arrays de residuales y valores ACF/PACF sin limitación de tamaño:

```python
result_payload = {
    ...
    "residuals": residuals.tolist(),  # <-- Puede ser muy grande
    "residual_dates": [d.strftime("%Y-%m-%d") for d in residual_dates],  # <-- Duplica fechas
    "acf": {"values": acf_vals.tolist(), ...},
    "pacf": {"values": pacf_vals.tolist(), ...}
}
```

**Impacto:**  
- Con 3 años de datos (≈750 días hábiles), el payload puede superar 100KB solo en residuales
- Redundancia: Las fechas se envían como strings en lugar de usar índices

#### 2.4 **Precisión Flotante Excesiva**

**Archivo:** `Backend/services/model_rf/forecast.py` líneas 28-31

```python
# forecast.py - Precision excesiva en logs (no afecta payload pero indica patrón)
print(f"Day {i + 1}: {forecast[i]:.4f}")  # 4 decimales
```

El frontend no necesita más de 2-4 decimales para precios de acciones. Los valores flotantes en JSON mantienen precisión completa de float64.

---

#### 2.5 **Falta de Memoización en Componente de Gráficas**

**Archivo:** `Frontend/src/components/GraphDisplay.js` línea 38

**Problema:**  
El componente `GraphDisplay` no usa `React.memo` ni `useMemo` para los datos procesados:

```javascript
// GraphDisplay.js - Sin memoización
function GraphDisplay({ historicalData, forecastData, ticker }) {
    const chartRef = useRef(null);
    // <-- Todos estos cálculos se re-ejecutan en cada render
    const historicalDates = hasHistoricalData ? historicalData.dates : [];
    const forecastDates = hasForecastData ? forecastData.map(p => p.date) : [];
    const allDates = [...new Set([...historicalDates, ...forecastDates])]
        .sort((a, b) => new Date(a) - new Date(b));
    // ... más procesamiento
}
```

**Impacto:**  
- Re-renderizado innecesario cuando cambian props no relacionadas en el padre
- Recálculo de mapas y ordenamiento de fechas en cada render
- Con datasets grandes (>1000 puntos), puede causar lag perceptible

---

### 🟢 Baja Prioridad

#### 2.6 **Estados de Carga Bien Implementados ✅**

El frontend implementa correctamente los estados de carga:

```javascript
// App.js - Correcto uso de TanStack Query
const trainMutation = useTrainModelMutation({
    onMutate: () => {
        message.loading({ content: 'Enviando trabajo...', key: 'trainSubmit' });
    },
    ...
});
```

#### 2.7 **Sistema de Polling para Jobs de Entrenamiento ✅**

El frontend implementa correctamente polling con cleanup:

```javascript
// App.js:290-310 - Polling configurado con timeout de 10 minutos
setTimeout(() => {
    if (pollingIntervals[key] === intervalId) {
        stopPollingForJob(ticker, modelType);
        updateActiveTrainingJob(ticker, modelType, { status: 'timeout' });
    }
}, 600000);
```

---

## 3. Propuesta de Soluciones

### 3.1 Solución para XGBoost y Prophet (Alta Prioridad)

**Opción A: Implementar Workers Celery (Recomendado)**

Crear archivos `celery_app.py` y `tasks.py` siguiendo el patrón de RF/LSTM:

```python
# Backend/services/model_xgb/celery_app.py (NUEVO)
from celery import Celery
import os

REDIS_URL = os.getenv("CELERY_BROKER_URL")
RESULT_BACKEND_URL = os.getenv("CELERY_RESULT_BACKEND_URL_XGB")

celery_app = Celery(
    "xgb_worker",
    broker=REDIS_URL,
    backend=RESULT_BACKEND_URL,
    include=["model_xgb.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    worker_concurrency=2,
    task_acks_late=True,
)
```

```python
# Backend/services/model_xgb/tasks.py (NUEVO)
from .celery_app import celery_app
from celery import Task

@celery_app.task(bind=True, name="train_xgb_model_task")
def train_xgb_model_task(self: Task, request_data_dict: dict):
    # Mover lógica de train_model() aquí
    self.update_state(state='PROGRESS', meta={'progress': 50})
    ...
    return result_payload
```

**Modificar docker-compose.yml:**
```yaml
  model-xgb-worker:
    build:
      context: ./Backend
      dockerfile: services/model_xgb/Dockerfile
    command: celery -A model_xgb.celery_app worker -l info
    environment:
      - CELERY_BROKER_URL=redis://redis_broker:6379/0
      - CELERY_RESULT_BACKEND_URL_XGB=redis://redis_broker:6379/3
```

---

### 3.2 Solución para Predicciones Bloqueantes (Alta Prioridad)

**Opción A: Usar run_in_executor (Rápida, mínimos cambios)**

```python
# Backend/services/model_rf/main.py - MODIFICAR predict()
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

@app.get("/predict")
async def predict(...):
    ...
    # Ejecutar operación CPU-bound en thread separado
    loop = asyncio.get_event_loop()
    forecast, lower_bounds, upper_bounds = await loop.run_in_executor(
        executor,
        forecast_future_prices,  # función
        model, data.copy(), forecast_horizon, target_col  # argumentos
    )
    ...
```

**Opción B: Mover predicciones a Celery (Mayor escalabilidad)**

Para predicciones que tarden más de 5 segundos, considerar el mismo patrón de tareas asíncronas.

---

### 3.3 Optimización de Payload (Media Prioridad)

**Reducir tamaño de respuesta de entrenamiento:**

```python
# Backend/services/model_rf/tasks.py
import numpy as np

def serialize_with_precision(arr, decimals=4):
    """Serializa arrays con precisión limitada"""
    return [round(float(x), decimals) for x in arr]

result_payload = {
    ...
    # Limitar residuales a últimos 365 días
    "residuals": serialize_with_precision(residuals[-365:], 4),
    "residual_dates": [d.strftime("%Y-%m-%d") for d in residual_dates[-365:]],
    # ACF/PACF solo necesitan ~40 lags típicamente
    "acf": {
        "values": serialize_with_precision(acf_vals[:40], 4),
        ...
    }
}
```

---

### 3.4 Memoización en Frontend (Media Prioridad)

```javascript
// Frontend/src/components/GraphDisplay.js - MODIFICAR
import React, { useRef, useMemo, memo } from 'react';

function GraphDisplay({ historicalData, forecastData, ticker }) {
    const chartRef = useRef(null);

    // Memoizar procesamiento de datos
    const { allDates, chartData } = useMemo(() => {
        const historicalDates = historicalData?.dates || [];
        const forecastDates = forecastData?.map(p => p.date) || [];
        
        const allDates = [...new Set([...historicalDates, ...forecastDates])]
            .sort((a, b) => new Date(a) - new Date(b));
        
        // ... resto del procesamiento
        return { allDates, chartData };
    }, [historicalData, forecastData]);  // Solo recalcular si cambian estos props

    // ... resto del componente
}

// Prevenir re-renders innecesarios cuando props no cambian
export default memo(GraphDisplay);
```

También aplicar `memo` a:
- `ResidualsDisplay`
- `MetricsDisplay`
- `ModelComparisonTable`

---

## 4. Resumen Ejecutivo

| Categoría | Issues Encontrados | Estado Actual |
|-----------|-------------------|---------------|
| **Concurrencia Backend** | 2 críticos (XGB/Prophet bloqueantes) | 🔴 Requiere acción inmediata |
| **Predicciones** | 4 servicios con pseudo-async | 🟡 Mejorable |
| **Payload Size** | Redundancia moderada | 🟢 Funcional |
| **Frontend Performance** | Falta memoización | 🟡 Mejorable |
| **Estado de Carga UI** | Bien implementado | ✅ OK |
| **Sistema de Polling** | Bien implementado | ✅ OK |

### Priorización de Trabajo

1. **Semana 1:** Implementar Celery workers para XGBoost y Prophet
2. **Semana 2:** Aplicar `run_in_executor` a todos los endpoints `/predict`
3. **Semana 3:** Optimizar payloads y añadir memoización al frontend

---

*Reporte generado automáticamente por GitHub Copilot*
