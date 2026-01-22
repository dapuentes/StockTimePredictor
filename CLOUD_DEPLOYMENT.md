# 🚀 StockTime Predictor - Cloud Deployment Guide

## Arquitectura en la Nube

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           USUARIOS                                        │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     FRONTEND (Vercel/Netlify)                            │
│                        React Application                                  │
│                    https://your-app.vercel.app                           │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    API GATEWAY (Cloud Run)                               │
│              https://api-gateway-xxxxx.run.app                           │
│                         Puerto 8000                                       │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
        ┌───────────┬───────────┼───────────┬───────────┬───────────┐
        ▼           ▼           ▼           ▼           ▼           ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
   │   RF    │ │  LSTM   │ │ XGBoost │ │ Prophet │ │  SHAP   │ │Ensemble │
   │  :8001  │ │  :8002  │ │  :8003  │ │  :8004  │ │  :8005  │ │  :8006  │
   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └─────────┘
        │           │           │           │           │
        └───────────┴───────────┴───────────┴───────────┘
                                │
                                ▼
                ┌───────────────────────────┐
                │   Google Cloud Storage    │
                │      (Modelos)            │
                └───────────────────────────┘
```

## 📋 Requisitos Previos

### Google Cloud Platform
1. Cuenta de GCP con facturación habilitada (tier gratuito incluye 2M requests/mes)
2. Proyecto de GCP creado
3. APIs habilitadas:
   - Cloud Run API
   - Cloud Build API
   - Container Registry API
   - Cloud Storage API

### Herramientas Locales
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
- Docker (para pruebas locales)
- Node.js 18+ (para frontend)

## 🔧 Configuración Inicial

### 1. Configurar GCP
```bash
# Autenticarse en GCP
gcloud auth login

# Configurar proyecto
gcloud config set project YOUR_PROJECT_ID

# Habilitar APIs necesarias
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable storage.googleapis.com

# Crear bucket para modelos
gsutil mb -l us-central1 gs://stocktime-predictor-models
```

### 2. Configurar Permisos de Cloud Build
```bash
# Obtener número de proyecto
PROJECT_NUMBER=$(gcloud projects describe YOUR_PROJECT_ID --format='value(projectNumber)')

# Dar permisos a Cloud Build para desplegar en Cloud Run
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:$PROJECT_NUMBER@cloudbuild.gserviceaccount.com" \
    --role="roles/run.admin"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:$PROJECT_NUMBER@cloudbuild.gserviceaccount.com" \
    --role="roles/iam.serviceAccountUser"
```

## 🚀 Despliegue

### Opción 1: Despliegue Automático con Cloud Build

```bash
# Desde el directorio raíz del proyecto
gcloud builds submit --config cloudbuild.yaml
```

Este comando:
1. Construye las 7 imágenes Docker
2. Las sube a Container Registry
3. Despliega cada servicio en Cloud Run
4. Configura las variables de entorno automáticamente

### Opción 2: Despliegue Manual por Servicio

```bash
# Construir y desplegar un servicio individual
cd Backend

# Ejemplo: Desplegar RF Service
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/model-rf
gcloud run deploy model-rf \
    --image gcr.io/YOUR_PROJECT_ID/model-rf \
    --region us-central1 \
    --memory 2Gi \
    --timeout 600 \
    --set-env-vars GCS_BUCKET_NAME=stocktime-predictor-models
```

## 🌐 Frontend en Vercel

### 1. Configurar Variables de Entorno en Vercel

En el dashboard de Vercel, añade:
```
REACT_APP_API_URL=https://api-gateway-xxxxx-uc.a.run.app
```

### 2. Desplegar Frontend

```bash
# Instalar Vercel CLI
npm install -g vercel

# Desplegar
cd Frontend
vercel --prod
```

O simplemente conecta tu repositorio de GitHub a Vercel para despliegues automáticos.

## 📊 Estimación de Costos (Tier Gratuito)

| Servicio | Límite Gratuito | Uso Estimado |
|----------|-----------------|--------------|
| Cloud Run | 2M requests/mes | ✅ Suficiente |
| Cloud Run | 360K vCPU-sec | ✅ Suficiente |
| Cloud Run | 180K GiB-sec | ✅ Suficiente |
| Cloud Storage | 5GB | ✅ Suficiente |
| Cloud Build | 120 min/día | ✅ Suficiente |
| Container Registry | 0.5GB gratis | ✅ Suficiente |

**Costo estimado mensual:** $0 (dentro del tier gratuito)

## 🔒 Seguridad

### Servicios Internos (No públicos)
Los servicios de modelos (RF, LSTM, XGBoost, Prophet, SHAP) están configurados con `--no-allow-unauthenticated`, lo que significa que solo el API Gateway puede acceder a ellos.

### API Gateway (Público)
Solo el API Gateway tiene acceso público. Para producción, considera:
1. Añadir autenticación (Firebase Auth, Auth0)
2. Implementar rate limiting
3. Configurar Cloud Armor para WAF

## 🧪 Testing Local (Cloud Config)

```bash
# Probar la configuración cloud localmente
docker-compose -f docker-compose.cloud.yml up --build

# Los servicios estarán disponibles en:
# - API Gateway: http://localhost:8000
# - RF: http://localhost:8001
# - LSTM: http://localhost:8002
# - XGBoost: http://localhost:8003
# - Prophet: http://localhost:8004
# - SHAP: http://localhost:8005
# - Ensemble: http://localhost:8006
```

## 📝 Diferencias entre Local y Cloud

| Aspecto | Local (new_main) | Cloud (cloud-v2) |
|---------|------------------|------------------|
| Training | Async (Celery + Redis) | Sync (timeout extendido) |
| Workers | 5 Celery workers | No hay workers |
| Storage | Docker volumes | Google Cloud Storage |
| Redis | Sí | No |
| Timeout | Ilimitado | Hasta 60 min por request |

## 🐛 Troubleshooting

### Error: "Service unavailable"
```bash
# Verificar logs del servicio
gcloud run services logs read model-rf --region us-central1
```

### Error: "Timeout"
- Aumenta el timeout en Cloud Run (máximo 60 min)
- Reduce el rango de datos de entrenamiento
- Considera usar Cloud Tasks para jobs muy largos

### Error: "Permission denied" en GCS
```bash
# Verificar permisos del service account
gcloud run services describe model-rf --region us-central1 --format='value(spec.template.spec.serviceAccountName)'
```

## 📚 Enlaces Útiles

- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Cloud Build Documentation](https://cloud.google.com/build/docs)
- [Vercel Documentation](https://vercel.com/docs)
- [GCS Python Client](https://googleapis.dev/python/storage/latest/index.html)
