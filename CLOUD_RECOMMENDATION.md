# ☁️ Recomendación de Arquitectura Cloud — StockTimePredictor

> Documento de decisión arquitectónica para el despliegue en la nube.  
> Fecha: Febrero 2026

---

## 1. Estado Actual (`cloud-v2`)

La implementación existente despliega **7 microservicios independientes** en Google Cloud Run:

```
Frontend (Vercel)
    └── API Gateway (Cloud Run) ─── :8000
            ├── RF Service          ─── :8001
            ├── LSTM Service        ─── :8002
            ├── XGBoost Service     ─── :8003
            ├── Prophet Service     ─── :8004
            ├── SHAP Service        ─── :8005
            └── Ensemble Service    ─── :8006
```

### Problemas detectados

| Severidad | Problema | Impacto |
|---|---|---|
| 🔴 Bloqueante | Servicios con `--no-allow-unauthenticated` pero el Gateway no envía token OIDC | 403 Forbidden en todas las llamadas service-to-service |
| 🔴 Bloqueante | Timeout del Gateway Cloud Run = 300s, pero servicios necesitan hasta 900s | Gateway muere antes de que el entrenamiento termine |
| 🟡 Importante | Gunicorn con 1 worker + entrenamiento síncrono | Servicio completamente bloqueado durante training |
| 🟡 Importante | `cloudbuild.yaml` usa `gcr.io` (Container Registry deprecado) | Migrar a Artifact Registry eventualmente |
| 🟡 Importante | Ensemble se despliega sin URLs de los otros servicios | Ensemble no puede contactar modelos |
| 🟡 Importante | 7 cold starts independientes (~15-30s cada uno con libs ML) | UX degradada en cada request tras inactividad |
| 🟢 Menor | CORS wildcard `*` en gateway cloud | Seguridad débil en producción |
| 🟢 Menor | Cache en memoria se pierde con scale-to-zero | Re-descarga de modelos desde GCS |

---

## 2. Análisis de Alternativas

### Opción A: 7 Microservicios (actual)

```
                ┌───────────┐
                │  Gateway  │ Cloud Run (1Gi/1CPU)
                └─────┬─────┘
      ┌───┬───┬───┬───┼───┬───┐
      ▼   ▼   ▼   ▼   ▼   ▼   ▼
     RF  LSTM XGB  PR SHAP ENS  ← 6 servicios Cloud Run adicionales
```

| Aspecto | Valor |
|---|---|
| Servicios Cloud Run | 7 |
| Imágenes Docker | 7 |
| Cold starts | 7 independientes (~15-30s c/u con TensorFlow, Prophet, etc.) |
| Complejidad de auth | Alta (OIDC service-to-service entre 7 servicios) |
| Build time (Cloud Build) | ~15-20 min (7 builds paralelos) |
| Costo estimado (fuera de free tier) | ~$15-40/mes |
| Mantenimiento | 7 Dockerfiles, 7 requirements.txt, 7 deploys |

**Veredicto:** Sobreingeniería para un proyecto de portfolio. La complejidad operativa no se justifica.

---

### Opción B: Backend Unificado (recomendada) ✅

```
┌──────────────────────────────────────┐
│       Frontend — Vercel (GRATIS)     │
│       React build estático + CDN     │
└───────────────┬──────────────────────┘
                │ HTTPS
                ▼
┌──────────────────────────────────────┐
│   Backend UNIFICADO — Cloud Run      │
│   4Gi RAM / 2 CPU / timeout 900s    │
│                                      │
│   FastAPI (UN solo proceso)          │
│   ├── /train/{model}     (síncrono)  │
│   ├── /predict/{model}               │
│   ├── /models/{model}                │
│   ├── /shap/explain                  │
│   ├── /ensemble/predict              │
│   └── /health                        │
│                                      │
│   Importa directamente:             │
│   rf_model.py, lstm_model.py,       │
│   xgb_model.py, prophet_model.py,   │
│   shap_explainer.py, ensemble_model  │
└───────────────┬──────────────────────┘
                │
                ▼
┌──────────────────────────────────────┐
│    Google Cloud Storage (5GB free)   │
│    Modelos persistidos (.joblib/.h5) │
└──────────────────────────────────────┘
```

| Aspecto | Valor |
|---|---|
| Servicios Cloud Run | **1** |
| Imágenes Docker | **1** |
| Cold starts | **1** (~20-30s, incluye todas las libs) |
| Complejidad de auth | **Ninguna** (todo en un proceso, sin llamadas entre servicios) |
| Build time | ~5-8 min (1 imagen) |
| Costo estimado | **$0** (free tier) a ~$7/mes con uso moderado |
| Mantenimiento | 1 Dockerfile, 1 requirements.txt, 1 deploy |

**Ventajas clave:**
- Sin auth service-to-service (eliminamos el bug #1 completamente)
- Sin timeout cruzado (eliminamos el bug #2 completamente)
- Un solo cold start en vez de 7
- Código de modelos reutilizado tal cual (mismos `rf_model.py`, `lstm_model.py`, etc.)
- GCS storage reutilizado (`gcs_storage.py` sin cambios)
- Deploy trivial: `gcloud run deploy` con 1 comando

**Trade-offs aceptables:**
- Sin escalamiento independiente por modelo → irrelevante para portfolio
- Imagen Docker más pesada (~2-3 GB con TensorFlow + Prophet) → aceptable
- Un crash en un modelo afecta todo el servicio → mitigable con try/except robusto

---

### Opción C: Railway ($5/mes)

| Aspecto | Valor |
|---|---|
| Costo | $5/mes (plan Hobby) |
| Deploy | Push a GitHub → deploy automático |
| Cold starts | **Ninguno** (servicio siempre activo en Hobby) |
| RAM | 8GB compartido |
| Storage | Volúmenes persistentes (sin necesidad de GCS) |

**Cuándo elegir Railway:** Si los cold starts de Cloud Run (~20-30s) son inaceptables y se puede pagar $5/mes.

---

### Opción D: Render (gratis limitado)

| Aspecto | Valor |
|---|---|
| Costo | $0 (free tier) |
| RAM | 512 MB (free) → **insuficiente para LSTM/TensorFlow** |
| Cold starts | ~30-60s en free tier |

**Veredicto:** RAM insuficiente. Descartada.

---

### Opción E: Fly.io ($0-5/mes)

| Aspecto | Valor |
|---|---|
| Costo | $0-5/mes |
| Máquinas | Configurables (hasta 8GB RAM) |
| Scale-to-zero | Sí |
| Complejidad | Mayor que Cloud Run (Fly CLI, `fly.toml`, etc.) |

**Cuándo elegir Fly.io:** Si se necesita flexibilidad en recursos sin vendor-lock con GCP.

---

## 3. Decisión: Opción B — Backend Unificado en Cloud Run

### Justificación

1. **Costo $0:** El free tier de Cloud Run (2M requests, 360K vCPU-sec, 180K GiB-sec) es más que suficiente para un proyecto de portfolio.
2. **Complejidad mínima:** 1 servicio, 1 Dockerfile, 0 auth entre servicios.
3. **Reutilización máxima:** Todo el código de modelos (`rf_model.py`, `lstm_model.py`, `xgb_model.py`, `prophet_model.py`, `shap_explainer.py`, `ensemble_model.py`) y utilidades (`gcs_storage.py`, `preprocessing.py`, `evaluation.py`) se importa directamente.
4. **GCS ya implementado:** `gcs_storage.py` funciona sin cambios.
5. **Frontend en Vercel** gratis con deploy automático desde GitHub.

### Recursos Cloud Run

```yaml
memory:    4Gi        # Suficiente para TensorFlow + Prophet + XGBoost en memoria
cpu:       2          # 2 vCPUs para entrenamiento
timeout:   900s       # 15 min max (LSTM puede tardar ~10 min)
concurrency: 10       # Limitar requests concurrentes (entrenamiento es pesado)
min-instances: 0      # Scale-to-zero (gratis cuando no se usa)
max-instances: 2      # Limitar costos
```

### Estimación de costos (uso de portfolio)

| Escenario | Requests/mes | vCPU-sec | GiB-sec | Costo |
|---|---|---|---|---|
| **Demo ocasional** | ~500 | ~5,000 | ~10,000 | **$0** |
| **Uso activo** | ~5,000 | ~50,000 | ~100,000 | **$0** |
| **Uso intenso** | ~20,000 | ~200,000 | ~400,000 | **~$3-5** |
| Free tier límite | 2,000,000 | 360,000 | 180,000 | $0 |

---

## 4. Plan de Implementación

### Archivos a crear

```
Backend/
├── cloud/
│   ├── Dockerfile              # Imagen unificada con todas las dependencias
│   ├── requirements.txt        # Todas las deps consolidadas
│   └── main.py                 # FastAPI unificado con todos los endpoints
```

### Archivos reutilizados sin cambios

```
Backend/
├── services/
│   ├── model_rf/rf_model.py            ← import directo
│   ├── model_lstm/lstm_model.py        ← import directo
│   ├── model_xgb/xgb_model.py         ← import directo
│   ├── model_prophet/prophet_model.py  ← import directo
│   ├── shap_explainer/shap_explainer.py ← import directo
│   └── model_ensemble/ensemble_model.py ← import directo
├── utils/
│   ├── gcs_storage.py                  ← sin cambios
│   ├── preprocessing.py                ← sin cambios
│   ├── evaluation.py                   ← sin cambios
│   └── import_data.py                  ← sin cambios
```

### Frontend (Vercel)

```
Frontend/
├── vercel.json        # Config de rewrites para API
└── .env.production    # REACT_APP_API_URL=https://backend-xxx.run.app
```

### Pasos

1. Crear `Backend/cloud/main.py` — FastAPI unificado
2. Crear `Backend/cloud/Dockerfile` — Imagen única
3. Crear `Backend/cloud/requirements.txt` — Deps consolidadas
4. Crear `cloudbuild.yaml` simplificado (1 build + 1 deploy)
5. Configurar Vercel para el frontend
6. Actualizar `CLOUD_DEPLOYMENT.md`

---

## 5. Comparativa Final

| Criterio | 7 Microservicios | Backend Unificado |
|---|---|---|
| Costo mensual | $15-40 | **$0** |
| Tiempo de deploy | ~20 min | **~5 min** |
| Cold start total | ~30s × 7 rutas | **~25s × 1** |
| Bugs de auth | Sí (OIDC pendiente) | **No aplica** |
| Bugs de timeout | Sí (cascada) | **No aplica** |
| Archivos de infra | 25 | **~5** |
| Código de modelos | Mismo | **Mismo** |
| Escalabilidad | Alta (independiente) | Suficiente (portfolio) |

> **Nota:** La arquitectura de microservicios local (Docker Compose + Celery + Redis) se mantiene intacta en `new_main` para desarrollo y entrenamiento pesado. La versión cloud es una **adaptación para deploy ligero**.
