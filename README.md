# 📊 StockTimePredictor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![React](https://img.shields.io/badge/React-18.3+-61DAFB.svg)](https://reactjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://docker.com)

**StockTimePredictor** es una plataforma completa e interactiva para el pronóstico de precios bursátiles utilizando técnicas avanzadas de análisis de series temporales y modelos de machine learning. La plataforma combina múltiples algoritmos como LSTM, Random Forest, XGBoost y Prophet en una arquitectura de microservicios escalable con una interfaz web moderna y intuitiva.

---

## 🧠 Características Principales

### 🔮 Predicción Avanzada
- **Múltiples modelos**: LSTM, Random Forest, XGBoost, Prophet
- **Intervalos de confianza**: Estimación de incertidumbre en predicciones
- **Optimización automática**: Hyperparameter tuning con Keras Tuner y Optuna
- **Análisis técnico**: Indicadores financieros integrados automaticamente

### 🏗️ Arquitectura Moderna
- **Microservicios**: Cada modelo como servicio independiente
- **API Gateway**: Punto de entrada unificado con balanceador de carga
- **Containerización**: Despliegue con Docker y Docker Compose
- **Escalabilidad**: Fácil adición de nuevos modelos y servicios

### 💻 Interfaz Intuitiva
- **Dashboard interactivo**: Visualización en tiempo real de predicciones
- **Configuración flexible**: Parámetros ajustables por modelo
- **Métricas detalladas**: MAE, RMSE, MAPE y análisis de residuos
- **Exportación**: Descarga de resultados y gráficos

### 📊 Análisis Completo
- **Preprocesamiento inteligente**: Limpieza y transformación automática
- **Validación cruzada**: Evaluación robusta de modelos
- **Comparación de modelos**: Benchmarking automático
- **Visualización avanzada**: Gráficos interactivos con Chart.js

---

## 🏗️ Arquitectura del Sistema

```bash
StockTimePredictor/
├── 🐳 docker-compose.yml                # Orquestación de servicios
├── 📄 README.md                         # Documentación principal
│
├── 🎨 Frontend/                         # Interfaz de Usuario (React)
│   ├── 📱 public/
│   │   └── index.html                   # Punto de entrada HTML
│   ├── ⚛️ src/
│   │   ├── 🧩 components/               # Componentes React reutilizables
│   │   │   ├── ConfigurationPanel_AntD.js   # Panel de configuración
│   │   │   ├── GraphDisplay.js              # Visualización de gráficos
│   │   │   ├── MetricsDisplay_AntD.js       # Métricas y resultados
│   │   │   ├── ModelComparisonTable.js     # Comparación de modelos
│   │   │   └── LoadingSpinner.js           # Indicadores de carga
│   │   ├── 🎣 hooks/                    # Custom hooks de React
│   │   │   └── useApiMutations.js          # Gestión de estado API
│   │   ├── 🌐 services/                 # Servicios de comunicación
│   │   │   └── api.js                      # Cliente HTTP con Axios
│   │   ├── App.js                      # Componente principal
│   │   └── index.js                    # Punto de entrada React
│   └── 📦 package.json                 # Dependencias y scripts
│
├── 🔧 Backend/                          # Servicios Backend (Python)
│   ├── 🚪 api_gateway/                  # API Gateway (FastAPI)
│   │   ├── app.py                      # Orquestador principal
│   │   ├── Dockerfile                  # Imagen Docker
│   │   └── requirements.txt            # Dependencias Python
│   │
│   ├── 🎯 services/                     # Microservicios de Modelos
│   │   ├── 🧠 model_lstm/               # Servicio LSTM
│   │   │   ├── lstm_model.py           # Clase modelo LSTM
│   │   │   ├── train.py                # Entrenamiento con optimización
│   │   │   ├── forecast.py             # Predicciones con incertidumbre
│   │   │   ├── main.py                 # API FastAPI del servicio
│   │   │   ├── requirements.txt        # TensorFlow, Keras, etc.
│   │   │   └── README.md               # Documentación específica
│   │   │
│   │   ├── 🌳 model_rf/                 # Servicio Random Forest
│   │   │   ├── rf_model.py             # Implementación Random Forest
│   │   │   ├── train.py                # Pipeline de entrenamiento
│   │   │   ├── forecast.py             # Predicciones ensemble
│   │   │   ├── main.py                 # API del servicio
│   │   │   ├── models/                 # Modelos persistidos (.joblib)
│   │   │   └── README.md               # Documentación
│   │   │
│   │   ├── ⚡ model_xgb/                # Servicio XGBoost
│   │   │   ├── xgb_model.py            # Modelo XGBoost optimizado
│   │   │   ├── main_xgb.py             # API del servicio
│   │   │   ├── forecast.py             # Predicciones con boosting
│   │   │   └── requirements.txt        # XGBoost, Optuna, etc.
│   │   │
│   │   └── 📈 model_prophet/            # Servicio Prophet (Meta)
│   │       ├── prophet_model.py        # Modelo Facebook Prophet
│   │       ├── prophet_service.py      # Lógica de negocio
│   │       └── app.py                  # API del servicio
│   │
│   ├── 🛠️ utils/                        # Utilidades Compartidas
│   │   ├── preprocessing.py            # Preprocesamiento de datos
│   │   ├── import_data.py              # Descarga datos yfinance
│   │   ├── evaluation.py               # Métricas y evaluación
│   │   ├── visualizations.py           # Gráficos y plots
│   │   └── imports.py                  # Imports centralizados
│   │
│   └── 🎓 training/                     # Scripts de Entrenamiento
│       └── scripts/
│           └── train_rf.py             # Entrenamiento Random Forest
```

---

## ⚙️ Stack Tecnológico

### 🐍 Backend (Python 3.9+)

| Categoría | Tecnologías |
|-----------|-------------|
| **Framework Web** | FastAPI, Uvicorn |
| **Machine Learning** | TensorFlow/Keras, Scikit-learn, XGBoost |
| **Deep Learning** | LSTM, Neural Networks, Keras Tuner |
| **Series Temporales** | Prophet (Meta), Statsmodels |
| **Datos Financieros** | yfinance, pandas, numpy |
| **Containerización** | Docker, Docker Compose |
| **Persistencia** | Joblib, Pickle, JSON |
| **HTTP Client** | httpx, requests |

### ⚛️ Frontend (React 18.3+)

| Categoría | Tecnologías |
|-----------|-------------|
| **Framework UI** | React, JavaScript ES6+ |
| **UI Components** | Ant Design (antd) |
| **Visualización** | Chart.js, React-ChartJS-2 |
| **Estado/Queries** | TanStack React Query |
| **HTTP Client** | Axios |
| **Fechas** | Day.js, React-DatePicker |
| **Utilidades** | PapaParse, HammerJS |
| **Testing** | Jest, React Testing Library |

### 🔧 DevOps & Desarrollo

| Herramienta | Propósito |
|-------------|-----------|
| **Docker Compose** | Orquestación de servicios |
| **Git** | Control de versiones |
| **CORS Middleware** | Comunicación cross-origin |
| **Environment Variables** | Configuración de servicios |
| **Volumes** | Persistencia de modelos |

---

## 📈 Modelos de Machine Learning

### Modelos Implementados ✅

| Modelo | Tipo | Características | Casos de Uso |
|--------|------|----------------|--------------|
| **LSTM** | Deep Learning | • Redes recurrentes<br>• Memoria a largo plazo<br>• Optimización automática<br>• Intervalos de confianza | Series temporales complejas<br>Patrones no lineales<br>Dependencias temporales |
| **Random Forest** | Ensemble | • Múltiples árboles<br>• Bootstrapping<br>• Reducción varianza<br>• Feature importance | Robustez general<br>Datos tabulares<br>Interpretabilidad |
| **XGBoost** | Gradient Boosting | • Boosting secuencial<br>• Regularización L1/L2<br>• Optimización Optuna<br>• Alta precisión | Competencias ML<br>Datos estructurados<br>Alto rendimiento |

### Modelos en Desarrollo 🔜

| Modelo | Estado | Características Planeadas |
|--------|--------|---------------------------|
| **Prophet** | 🔄 En desarrollo | • Estacionalidad automática<br>• Tendencias<br>• Días festivos<br>• Incertidumbre bayesiana |
| **Neural Network** | 📋 Planeado | • Red densa<br>• Dropout<br>• Batch normalization<br>• Early stopping |
| **Ensemble Meta-Model** | 💡 Conceptual | • Combinación de modelos<br>• Voting/Stacking<br>• Pesos dinámicos<br>• Meta-aprendizaje |

---

## 🚀 Instalación y Configuración

### Prerrequisitos

| Herramienta | Versión Mínima | Propósito |
|-------------|----------------|-----------|
| **Git** | 2.0+ | Clonación del repositorio |
| **Docker** | 20.0+ | Containerización |
| **Docker Compose** | 2.0+ | Orquestación de servicios |
| **Node.js** | 16.0+ | Frontend development |
| **Python** | 3.9+ | Backend development (opcional) |

### 🔧 Instalación Completa

#### 1. Clonar el Repositorio

```bash
git clone https://github.com/dapuentes/StockTimePredictor.git
cd StockTimePredictor
```

#### 2. Configurar Variables de Entorno (Opcional)

```bash
# Crear archivo .env para configuración personalizada
echo "RF_SERVICE_URL=http://model-rf:8001" > .env
echo "LSTM_SERVICE_URL=http://model-lstm:8002" >> .env
echo "XGB_SERVICE_URL=http://model-xgb:8003" >> .env
```

#### 3. Iniciar Backend con Docker

```bash
# Construir todas las imágenes
docker-compose build

# Iniciar todos los servicios en segundo plano
docker-compose up -d

# Verificar que los servicios estén ejecutándose
docker-compose ps
```

**Servicios Disponibles:**
- 🚪 **API Gateway**: `http://localhost:8000`
- 🧠 **LSTM Service**: `http://localhost:8002` (interno)
- 🌳 **Random Forest**: `http://localhost:8001` (interno)
- ⚡ **XGBoost**: `http://localhost:8003` (interno)

#### 4. Configurar y Ejecutar Frontend

```bash
# Navegar al directorio frontend
cd Frontend

# Instalar dependencias
npm install

# Iniciar servidor de desarrollo
npm start
```

**Frontend Disponible**: `http://localhost:3000`

#### 5. Verificar Instalación

```bash
# Verificar API Gateway
curl http://localhost:8000/

# Verificar salud de servicios
curl http://localhost:8000/health

# Ver logs de servicios
docker-compose logs api-gateway
docker-compose logs model-lstm
```

### 🛠️ Desarrollo Local (Opcional)

Para desarrollo sin Docker:

```bash
# Backend - Instalar dependencias por servicio
cd Backend/api_gateway
pip install -r requirements.txt

cd ../services/model_lstm
pip install -r requirements.txt

# Frontend
cd ../../../Frontend
npm install
npm start
```

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

### Ejemplos de Uso de API

#### Entrenar Modelo LSTM
```bash
curl -X POST "http://localhost:8000/train/lstm" \
  -F "ticket=AAPL" \
  -F "start_date=2022-01-01" \
  -F "end_date=2024-01-01" \
  -F "n_lags=10" \
  -F "sequence_length=30" \
  -F "optimize_params=true"
```

#### Realizar Predicción
```bash
curl -X GET "http://localhost:8000/predict/lstm?ticker=AAPL&forecast_horizon=10"
```

#### Respuesta JSON Típica
```json
{
  "status": "success",
  "model_type": "lstm",
  "ticker": "AAPL",
  "predictions": [180.5, 181.2, 179.8, ...],
  "confidence_intervals": {
    "lower": [175.1, 176.3, 174.9, ...],
    "upper": [185.9, 186.1, 184.7, ...]
  },
  "metrics": {
    "mae": 2.34,
    "rmse": 3.12,
    "mape": 1.89,
    "r2": 0.94
  },
  "metadata": {
    "training_time": "2024-01-15T10:30:00Z",
    "data_points": 504,
    "features_used": 15
  }
}
```

---

## 🛠️ Desarrollo y Contribución

### 📋 Roadmap del Proyecto

#### Próximas Funcionalidades
- [ ] **Modelo Prophet**: Integración completa de Meta Prophet
- [ ] **Real-time Data**: Streaming de datos en tiempo real
- [ ] **Advanced Analytics**: Análisis de sentimiento y noticias
- [ ] **Mobile App**: Aplicación móvil React Native
- [ ] **Cloud Deployment**: Deploy en AWS/GCP/Azure

#### Mejoras Técnicas
- [ ] **Testing Coverage**: Pruebas unitarias y de integración
- [ ] **CI/CD Pipeline**: GitHub Actions para deployment
- [ ] **Performance Monitoring**: Métricas de rendimiento en producción
- [ ] **API Rate Limiting**: Control de uso de endpoints
- [ ] **Caching Layer**: Redis para optimización
- [ ] **Database Integration**: PostgreSQL para persistencia

### 🤝 Guía de Contribución

#### Configuración para Desarrollo

```bash
# 1. Fork del repositorio en GitHub
# 2. Clonar tu fork
git clone https://github.com/TU_USUARIO/StockTimePredictor.git
cd StockTimePredictor

# 3. Crear rama para feature
git checkout -b feature/nueva-funcionalidad

# 4. Configurar entorno de desarrollo
# Backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r Backend/api_gateway/requirements.txt

# Frontend
cd Frontend
npm install
```

#### Estándares de Código

```python
# Python: Seguir PEP 8
# Usar type hints
def predict_prices(ticker: str, days: int = 10) -> List[float]:
    """Predice precios futuros para un ticker dado."""
    pass

# Documentación obligatoria para funciones públicas
# Tests unitarios para nuevas funcionalidades
```

```javascript
// JavaScript: ESLint + Prettier
// Componentes funcionales con hooks
const PredictionChart = ({ data, isLoading }) => {
  const [selectedModel, setSelectedModel] = useState('lstm');
  
  return (
    <div className="chart-container">
      {/* JSX aquí */}
    </div>
  );
};
```

#### Pull Request Guidelines

1. **Descripción Clara**: Explica qué cambios introduces
2. **Tests**: Incluye pruebas para nuevas funcionalidades
3. **Documentación**: Actualiza README y docstrings
4. **Screenshots**: Para cambios de UI, incluye capturas
5. **Breaking Changes**: Marca claramente cambios incompatibles

---

## 📊 Monitoreo y Métricas

### Métricas de Rendimiento

| Servicio | Métricas Clave | Objetivo |
|----------|----------------|----------|
| **API Gateway** | Latencia, Throughput, Error Rate | < 200ms, > 100 req/s, < 1% |
| **LSTM Model** | Tiempo entrenamiento, Precisión | < 5 min, > 90% R² |
| **Frontend** | Load Time, Bundle Size | < 3s, < 2MB |

### Logs y Debugging

```bash
# Ver logs en tiempo real
docker-compose logs -f api-gateway
docker-compose logs -f model-lstm

# Debugging de servicios individuales
docker-compose exec api-gateway /bin/bash
docker-compose exec model-lstm python -c "import tensorflow; print(tensorflow.__version__)"
```

---

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas!  
Abre un pull request o issue para sugerencias, mejoras o reportes de errores.

---

## 📄 Licencia

Este proyecto está bajo licencia [MIT](https://opensource.org/licenses/MIT).

---
