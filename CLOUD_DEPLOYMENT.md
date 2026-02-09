# 🚀 StockTime Predictor — Cloud Deployment Guide (v3 Unified)

> **Arquitectura**: 1 servicio Cloud Run (backend) + Vercel (frontend)  
> **Costo estimado**: ~$5–15 USD/mes con tráfico moderado

---

## Arquitectura en la Nube

```
┌─────────────────────────────────────────────────────────┐
│                       USUARIOS                           │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              FRONTEND (Vercel — Free Tier)                │
│                  React Application                        │
│              https://stocktime.vercel.app                 │
└────────────────────────┬────────────────────────────────┘
                         │  HTTPS
                         ▼
┌─────────────────────────────────────────────────────────┐
│         UNIFIED BACKEND (Cloud Run — Single Service)     │
│              https://stocktime-api-xxx.run.app            │
│                                                           │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │    RF    │ │   LSTM   │ │ XGBoost  │ │ Prophet  │  │
│   └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
│   ┌──────────┐ ┌──────────┐                              │
│   │   SHAP   │ │ Ensemble │    FastAPI + Gunicorn        │
│   └──────────┘ └──────────┘                              │
│                                                           │
│   4 Gi RAM · 2 vCPU · 900s timeout · 0→2 instancias     │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│          Google Cloud Storage (Modelos entrenados)        │
│          stocktime-predictor-models bucket                │
└─────────────────────────────────────────────────────────┘
```

---

## Requisitos Previos

1. **Google Cloud** — cuenta con billing habilitado
2. **Node.js 18+** — para el build del frontend
3. **gcloud CLI** — [instalación](https://cloud.google.com/sdk/docs/install)
4. **Vercel CLI** — `npm i -g vercel`

---

## 1. Configurar Google Cloud

```bash
# Autenticarse
gcloud auth login

# Crear proyecto (o seleccionar uno existente)
gcloud projects create stocktime-predictor --name="StockTime Predictor"
gcloud config set project stocktime-predictor

# Habilitar APIs necesarias
gcloud services enable \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com

# Crear repositorio en Artifact Registry (reemplaza gcr.io)
gcloud artifacts repositories create stocktime \
  --repository-format=docker \
  --location=us-central1 \
  --description="StockTime Predictor Docker images"

# Crear bucket GCS para modelos
gcloud storage buckets create gs://stocktime-predictor-models \
  --location=us-central1 \
  --uniform-bucket-level-access

# Dar permisos a Cloud Build
PROJECT_NUMBER=$(gcloud projects describe $(gcloud config get-value project) --format='value(projectNumber)')
gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
  --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
  --role="roles/run.admin"
gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
  --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"
```

---

## 2. Deploy Backend (Cloud Run)

### Opción A — Cloud Build (CI/CD)

```bash
gcloud builds submit --config cloudbuild.unified.yaml
```

Esto construye la imagen, la sube a Artifact Registry y despliega en Cloud Run automáticamente.

### Opción B — Deploy manual

```bash
# Construir imagen localmente
docker build -t stocktime-api -f Backend/cloud/Dockerfile Backend/

# Tag y push
docker tag stocktime-api us-central1-docker.pkg.dev/PROJECT_ID/stocktime/stocktime-api:latest
docker push us-central1-docker.pkg.dev/PROJECT_ID/stocktime/stocktime-api:latest

# Deploy
gcloud run deploy stocktime-api \
  --image us-central1-docker.pkg.dev/PROJECT_ID/stocktime/stocktime-api:latest \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 900 \
  --concurrency 4 \
  --min-instances 0 \
  --max-instances 2 \
  --set-env-vars "GCS_BUCKET_NAME=stocktime-predictor-models" \
  --allow-unauthenticated
```

### Verificar deployment

```bash
# Obtener URL del servicio
BACKEND_URL=$(gcloud run services describe stocktime-api \
  --region=us-central1 --format='value(status.url)')

# Test health
curl "$BACKEND_URL/health"

# Test root
curl "$BACKEND_URL/"
```

---

## 3. Deploy Frontend (Vercel)

```bash
# Actualizar la URL del backend en .env.production
cd Frontend
echo "REACT_APP_API_URL=$BACKEND_URL" > .env.production

# Build local (opcional, para verificar)
npm run build

# Deploy a Vercel
cd ..
vercel --prod
```

### Variables de entorno en Vercel Dashboard

En el [dashboard de Vercel](https://vercel.com/dashboard), configurar:

| Variable | Valor |
|----------|-------|
| `REACT_APP_API_URL` | `https://stocktime-api-XXXXXXXXXX-uc.a.run.app` |

---

## 4. Estructura de Archivos Cloud

```
Backend/cloud/
├── __init__.py
├── Dockerfile          # Imagen unificada (python:3.11-slim)
├── main.py             # FastAPI con TODOS los endpoints
└── requirements.txt    # Dependencias consolidadas

cloudbuild.unified.yaml # 1 build + 1 deploy
vercel.json             # Configuración Vercel (rewrites, headers)
Frontend/.env.production # URL del backend
```

---

## 5. Endpoints API

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/` | Info del servicio |
| `GET` | `/health` | Health check |
| `POST` | `/train/{model_type}` | Entrenar modelo (rf, lstm, xgboost, prophet) |
| `GET` | `/predict/{model_type}` | Predicciones futuras |
| `GET` | `/models/{model_type}` | Listar modelos guardados |
| `POST` | `/explain` | SHAP explicación local |
| `POST` | `/explain/importance/{model_type}` | Importancia global |
| `GET` | `/explain/plot/{model_type}` | Summary plot SHAP |
| `GET` | `/explain/waterfall/{model_type}` | Waterfall plot |
| `POST` | `/ensemble/predict` | Predicción ensemble |
| `GET` | `/ensemble/compare` | Comparar todos los modelos |
| `GET` | `/ensemble/models` | Modelos disponibles para ensemble |

### Ejemplo: Entrenar RF

```bash
curl -X POST "$BACKEND_URL/train/rf" \
  -H "Content-Type: application/json" \
  -d '{
    "ticket": "AAPL",
    "training_period": "3_years",
    "n_lags": 10,
    "target_col": "Close",
    "train_size": 0.8
  }'
```

### Ejemplo: Predicción

```bash
curl "$BACKEND_URL/predict/rf?ticket=AAPL&forecast_horizon=10&target_col=Close"
```

---

## 6. Costos Estimados

| Servicio | Tier | Costo Estimado |
|----------|------|----------------|
| **Cloud Run** | 4Gi / 2 CPU, 0→2 instancias | ~$5–10/mes |
| **Cloud Storage** | ~100 MB modelos | ~$0.02/mes |
| **Artifact Registry** | ~2 GB imagen | ~$0.20/mes |
| **Cloud Build** | 120 min/día gratis | $0/mes |
| **Vercel** | Hobby tier | $0/mes |
| **Total** | | **~$5–10/mes** |

> 💡 Con scale-to-zero y tráfico bajo, el costo puede ser **< $1/mes**.

---

## 7. Troubleshooting

### El training tarda mucho
- Cloud Run soporta hasta **60 min** por request. El timeout está configurado en 900s (15 min).
- LSTM con `optimize_params=true` puede tardar 10–20 min.
- Prophet y RF tardan 1–5 min típicamente.

### Error de memoria
- La instancia tiene 4 Gi. Si un modelo necesita más, subir a 8 Gi:
  ```bash
  gcloud run services update stocktime-api --memory 8Gi
  ```

### CORS errors en el frontend
- Verificar que `ALLOWED_ORIGINS` incluya tu dominio de Vercel.
- O agregar tu dominio explícitamente:
  ```bash
  gcloud run services update stocktime-api \
    --set-env-vars "ALLOWED_ORIGINS=https://tu-dominio.vercel.app"
  ```

### Modelos no se guardan
- Verificar que el bucket GCS existe y que el service account tiene permisos:
  ```bash
  gcloud storage buckets describe gs://stocktime-predictor-models
  ```

---

## 8. Desarrollo Local (Docker)

```bash
# Construir y correr localmente
docker build -t stocktime-local -f Backend/cloud/Dockerfile Backend/
docker run -p 8080:8080 \
  -e GCS_BUCKET_NAME=stocktime-predictor-models \
  -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/keys/sa.json \
  -v ~/.config/gcloud/application_default_credentials.json:/tmp/keys/sa.json \
  stocktime-local

# Probar
curl http://localhost:8080/health
```

Para desarrollo sin Docker, usar el compose original:
```bash
docker-compose up
```
