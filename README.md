# 📊 StockTimePredictor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**StockTimePredictor** es una plataforma interactiva para el pronóstico de precios bursátiles utilizando análisis de series temporales y modelos de machine learning como LSTM, Random Forest, y XGBoost. Está compuesta por un backend modular con microservicios en Python y una interfaz web intuitiva construida en React.

---

## 🧠 Características Principales

- Entrenamiento y despliegue de modelos ML para predicción financiera.
- Arquitectura de microservicios orquestada por un API Gateway.
- Interfaz gráfica (frontend) para configurar modelos y visualizar resultados.
- Visualización de series temporales, métricas y detalles del modelo.

---

## 🏗️ Estructura del Proyecto

```bash
StockTimePredictor/
├── docker-compose.yml         # Orquestador Docker
├── frontend/                  # Interfaz de usuario (React)
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── services/
│   │   └── App.js
├── backend/
│   ├── api_gateway/           # Entrada única al backend (FastAPI)
│   ├── services/              # Microservicios por modelo (LSTM, RF, XGB)
│   ├── utils/                 # Funciones compartidas
│   └── training/              # Scripts de entrenamiento
```

---

## ⚙️ Tecnologías Utilizadas

### Backend

- Python 3.9+
- FastAPI
- Docker & Docker Compose
- Modelos ML: LSTM, Random Forest, XGBoost (otros en desarrollo)

### Frontend

- React
- Ant Design
- Chart.js
- Axios
- @tanstack/react-query
- Day.js

---

## 📈 Modelos Disponibles

| Modelo | Estado |
|--------|--------|
| LSTM   | ✅ Implementado |
| Random Forest | ✅ Implementado |
| XGBoost | ✅ Implementado |
| Prophet (Meta) | 🔜 Por implementar |
| Red Neuronal Secuencial | 🔜 Por implementar |
| Ensemble | 🔜 Por implementar |

---

## 📋 Requisitos

### Generales

- Git
- Docker y Docker Compose

### Backend

- Python 3.9+
- `pip`

### Frontend

- Node.js 16+
- `npm` o `yarn`

---

## 🚀 Instalación y Ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/dapuentes/StockTimePredictor.git
cd StockTimePredictor
```

### 2. Iniciar el Backend (Docker Compose)

```bash
docker-compose build
docker-compose up -d
```

- API Gateway: `http://localhost:8000`
- Los microservicios de modelos corren en puertos como `8001`, `8002`, etc., accesibles a través del gateway.

### 3. Iniciar el Frontend

```bash
cd frontend
npm install
npm start
```

- Interfaz disponible en `http://localhost:3000`

⚠️ Asegúrate de que el backend esté ejecutándose antes de abrir el frontend.

---

## 🧪 Cómo Usar la App

1. **Seleccionar Configuración**
   - Ticker (ej. AAPL, TSLA)
   - Rango de fechas
   - Modelo (RF, LSTM, XGBoost)
   - Número de lags (días históricos)

2. **Entrenamiento**
   - Clic en “Entrenar Modelo”
   - Verás indicadores de carga y luego métricas detalladas

3. **Pronóstico**
   - Definir horizonte (días a futuro)
   - Clic en “Generar Pronóstico”
   - Visualizar resultados en el gráfico

4. **Explorar Resultados**
   - Gráfico + predicciones
   - Métricas
   - Detalles del modelo

---

## 🌐 API - Comunicación Frontend <-> Backend

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/train/{modelType}` | POST | Envia parámetros del modelo. Usa `FormData`. |
| `/predict/{modelType}` | GET | Devuelve predicciones en JSON. Parámetros vía query string. |

Implementado en: `frontend/src/services/api.js`

---

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas!  
Abre un pull request o issue para sugerencias, mejoras o reportes de errores.

---

## 📄 Licencia

Este proyecto está bajo licencia [MIT](https://opensource.org/licenses/MIT).

---
