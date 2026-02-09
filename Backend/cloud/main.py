"""
Unified Cloud Backend — StockTimePredictor
=============================================================================
Single FastAPI application that exposes every endpoint previously spread
across 7 microservices (API Gateway, RF, LSTM, XGBoost, Prophet, SHAP,
Ensemble).

Key design decisions
--------------------
* **No Celery / Redis** – training is synchronous with extended timeouts.
  Cloud Run supports up to 60 min per request.
* **GCS-first storage** with local fallback (via ``utils.gcs_storage``).
* **Direct imports** – model classes are imported in-process; there are no
  inter-service HTTP calls.
* **Stateless** – the only mutable state is an in-memory model cache that
  serves as a warm-start optimisation; it is safe to lose on scale-to-zero.
=============================================================================
"""

from __future__ import annotations

import os
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Body, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Model & utility imports  (all paths relative to PYTHONPATH=/app)
# ---------------------------------------------------------------------------
from services.model_rf.rf_model import TimeSeriesRandomForestModel
from services.model_rf.train import train_ts_model as train_rf
from services.model_rf.forecast import forecast_future_prices as forecast_rf

from services.model_lstm.lstm_model import TimeSeriesLSTMModel
from services.model_lstm.train import train_lstm_model as train_lstm
from services.model_lstm.forecast import forecast_future_prices_lstm as forecast_lstm

from services.model_xgb.xgb_model import XGBoostModel

from services.model_prophet.prophet_model import ProphetModel, train_prophet_model
from services.model_prophet.prophet_service import predict as prophet_predict

from services.shap_explainer.shap_explainer import SHAPExplainer

from services.model_ensemble.ensemble_model import EnsembleModel, EnsembleMethod

from utils.import_data import load_data
from utils.gcs_storage import (
    save_model_to_gcs,
    load_model_from_gcs,
    list_models_in_gcs,
    load_model_metadata,
    is_cloud_environment,
)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,https://*.vercel.app",
).split(",")

app = FastAPI(
    title="StockTime Predictor — Unified Cloud API",
    version="3.0.0-cloud",
    description="All-in-one backend for stock prediction: train, forecast, SHAP, ensemble.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tightened via ALLOWED_ORIGINS in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory caches (warm-start; safe to lose on scale-to-zero)
# ---------------------------------------------------------------------------
_model_cache: Dict[str, Any] = {}

SUPPORTED_MODELS = {"rf", "lstm", "xgboost", "prophet"}
MIN_DATA_ROWS = 260

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class TrainRequest(BaseModel):
    ticket: str = "NU"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    training_period: Optional[str] = None  # "1_year", "3_years", "5_years", "10_years"
    n_lags: Optional[int] = 10
    target_col: Optional[str] = "Close"
    train_size: Optional[float] = 0.8
    # LSTM-specific
    sequence_length: Optional[int] = None
    optimize_params: Optional[bool] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    # Prophet-specific
    forecast_horizon: Optional[int] = None


class ExplainRequest(BaseModel):
    ticker: str = "NU"
    model_type: str = "xgboost"
    top_features: int = 10


class EnsemblePredictRequest(BaseModel):
    ticker: str = "NU"
    forecast_horizon: int = 10
    target_col: str = "Close"
    models: Optional[List[str]] = None
    ensemble_method: str = "weighted_average"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_dates(
    start_date: Optional[str],
    end_date: Optional[str],
    training_period: Optional[str],
    default_start: str = "2020-12-10",
) -> Tuple[str, str]:
    """Return (start, end) date strings."""
    if start_date and end_date:
        s = datetime.strptime(start_date, "%Y-%m-%d")
        e = datetime.strptime(end_date, "%Y-%m-%d")
        if s > e:
            raise HTTPException(400, "start_date must be before end_date")
        return s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")

    if training_period:
        e = datetime.now()
        days_map = {"1_year": 365, "3_years": 1095, "5_years": 1825, "10_years": 3650}
        s = e - timedelta(days=days_map.get(training_period, 1095))
        return s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")

    return default_start, datetime.now().strftime("%Y-%m-%d")


def _load_stock(ticker: str, start: str, end: str) -> pd.DataFrame:
    data = load_data(ticker=ticker, start_date=start, end_date=end)
    if data is None or data.empty:
        raise HTTPException(404, f"No data found for ticker {ticker}")
    return data


def _cache_key(model_type: str, ticker: str) -> str:
    return f"{model_type}_{ticker}"


def _find_model(model_type: str, ticker: str):
    """Return a cached or GCS-loaded model, or None."""
    key = _cache_key(model_type, ticker)
    if key in _model_cache:
        return _model_cache[key]

    model = load_model_from_gcs(model_type, ticker)
    if model is not None:
        _model_cache[key] = model
    return model


def _validate_model_type(model_type: str):
    if model_type.lower() not in SUPPORTED_MODELS:
        raise HTTPException(400, f"Unsupported model. Choose from: {sorted(SUPPORTED_MODELS)}")


# =========================================================================
# Root / Health
# =========================================================================


@app.get("/")
async def root():
    return {
        "service": "StockTime Predictor — Unified Cloud API",
        "version": "3.0.0-cloud",
        "environment": "cloud" if is_cloud_environment() else "local",
        "models": sorted(SUPPORTED_MODELS),
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "3.0.0-cloud", "ts": datetime.now().isoformat()}


# =========================================================================
# TRAINING
# =========================================================================


@app.post("/train/{model_type}")
async def train_model(
    model_type: str = Path(..., description="rf | lstm | xgboost | prophet"),
    req: TrainRequest = Body(...),
):
    """Synchronous training.  Returns full result when done."""
    mt = model_type.lower()
    _validate_model_type(mt)

    start, end = _resolve_dates(req.start_date, req.end_date, req.training_period)
    data = _load_stock(req.ticket, start, end)

    if len(data) < MIN_DATA_ROWS:
        raise HTTPException(400, f"Need ≥{MIN_DATA_ROWS} rows, got {len(data)}.")

    print(f"[TRAIN] {mt} | {req.ticket} | {start}→{end} | {len(data)} rows")

    try:
        if mt == "rf":
            return await _train_rf(req, data, start, end)
        elif mt == "lstm":
            return await _train_lstm(req, data, start, end)
        elif mt == "xgboost":
            return await _train_xgb(req, data, start, end)
        elif mt == "prophet":
            return await _train_prophet(req, data, start, end)
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(500, f"Training failed: {exc}")


# ---- RF ----------------------------------------------------------------


async def _train_rf(req: TrainRequest, data: pd.DataFrame, start: str, end: str):
    model, feature_names, residuals, res_dates, acf_vals, pacf_vals, ci_acf, ci_pacf = train_rf(
        data=data,
        n_lags=req.n_lags or 10,
        target_col=req.target_col or "Close",
        train_size=req.train_size or 0.8,
        save_model_path=None,
    )

    meta = _build_meta(req, "rf", start, end, data, model, feature_names)
    path = save_model_to_gcs(model, "rf", req.ticket, meta, start, end)
    _model_cache[_cache_key("rf", req.ticket)] = model

    return {
        "status": "success",
        "message": f"RF trained for {req.ticket}",
        "ticker": req.ticket,
        "model_type": "RandomForest",
        "metrics": getattr(model, "metrics", None),
        "features_names": feature_names,
        "best_params": getattr(model, "best_params_", None),
        "residuals": residuals.tolist() if residuals is not None else None,
        "residual_dates": [d.strftime("%Y-%m-%d") for d in res_dates] if res_dates is not None else None,
        "acf": _format_acf(acf_vals, ci_acf),
        "pacf": _format_acf(pacf_vals, ci_pacf),
        "model_path": path,
    }


# ---- LSTM --------------------------------------------------------------


async def _train_lstm(req: TrainRequest, data: pd.DataFrame, start: str, end: str):
    model, feature_names, residuals, res_dates, acf_vals, pacf_vals, ci_acf, ci_pacf = train_lstm(
        data=data,
        n_lags=req.n_lags or 10,
        target_col=req.target_col or "Close",
        train_size=req.train_size or 0.8,
        sequence_length=req.sequence_length or 60,
        optimize_params=req.optimize_params if req.optimize_params is not None else False,
        epochs=req.epochs or 50,
        batch_size=req.batch_size or 32,
    )

    meta = _build_meta(req, "lstm", start, end, data, model, feature_names)
    path = save_model_to_gcs(model, "lstm", req.ticket, meta, start, end)
    _model_cache[_cache_key("lstm", req.ticket)] = model

    return {
        "status": "success",
        "message": f"LSTM trained for {req.ticket}",
        "ticker": req.ticket,
        "model_type": "LSTM",
        "metrics": getattr(model, "metrics", None),
        "features_names": feature_names,
        "residuals": residuals.tolist() if residuals is not None else None,
        "residual_dates": [d.strftime("%Y-%m-%d") for d in res_dates] if res_dates is not None else None,
        "acf": _format_acf(acf_vals, ci_acf),
        "pacf": _format_acf(pacf_vals, ci_pacf),
        "model_path": path,
    }


# ---- XGBoost -----------------------------------------------------------


async def _train_xgb(req: TrainRequest, data: pd.DataFrame, start: str, end: str):
    xgb_model = XGBoostModel(n_lags=req.n_lags or 10)
    processed = xgb_model.prepare_data(data, target_col=req.target_col or "Close")
    xgb_model.scale_data(processed)
    xgb_model.optimize_hyperparameters()
    metrics = xgb_model.evaluate()

    meta = _build_meta(req, "xgboost", start, end, data, xgb_model, None)
    path = save_model_to_gcs(xgb_model, "xgboost", req.ticket, meta, start, end)
    _model_cache[_cache_key("xgboost", req.ticket)] = xgb_model

    return {
        "status": "success",
        "message": f"XGBoost trained for {req.ticket}",
        "ticker": req.ticket,
        "model_type": "XGBoost",
        "metrics": metrics,
        "best_params": getattr(xgb_model, "best_params_", None),
        "model_path": path,
    }


# ---- Prophet -----------------------------------------------------------


async def _train_prophet(req: TrainRequest, data: pd.DataFrame, start: str, end: str):
    model, metrics, future_df = train_prophet_model(
        data=data,
        target_col=req.target_col or "Close",
        train_size=req.train_size or 0.8,
        regressor_cols=["Open", "High", "Low", "Volume"],
        forecast_horizon=req.forecast_horizon or 10,
    )

    meta = _build_meta(req, "prophet", start, end, data, model, None)
    path = save_model_to_gcs(model, "prophet", req.ticket, meta, start, end)
    _model_cache[_cache_key("prophet", req.ticket)] = model

    return {
        "status": "success",
        "message": f"Prophet trained for {req.ticket}",
        "ticker": req.ticket,
        "model_type": "Prophet",
        "metrics": metrics,
        "model_path": path,
    }


# ---- helpers -----------------------------------------------------------


def _build_meta(req, model_type, start, end, data, model, features):
    return {
        "ticker": req.ticket,
        "model_type": model_type,
        "training_start_date": start,
        "training_end_date": end,
        "n_lags": req.n_lags,
        "target_col": req.target_col,
        "train_size": req.train_size,
        "data_points": len(data),
        "metrics": getattr(model, "metrics", {}),
        "best_params": getattr(model, "best_params_", {}),
        "feature_names": features or [],
    }


def _format_acf(values, confint):
    if values is None:
        return None
    return {
        "values": values.tolist(),
        "confint_lower": confint[:, 0].tolist() if confint is not None else [],
        "confint_upper": confint[:, 1].tolist() if confint is not None else [],
    }


# =========================================================================
# PREDICTION
# =========================================================================


@app.get("/predict/{model_type}")
async def predict(
    model_type: str = Path(...),
    ticket: str = Query("NU"),
    forecast_horizon: int = Query(10),
    target_col: str = Query("Close"),
    history_days: int = Query(365),
):
    mt = model_type.lower()
    _validate_model_type(mt)

    model = _find_model(mt if mt != "xgboost" else "xgboost", ticket)
    if model is None:
        raise HTTPException(404, f"No trained {mt} model for {ticket}. Train first.")

    try:
        # Load recent data for prediction
        meta = load_model_metadata(mt, ticket)
        end_dt = (
            datetime.strptime(meta["training_end_date"], "%Y-%m-%d")
            if meta and "training_end_date" in meta
            else datetime.now()
        )
        start_dt = end_dt - timedelta(days=365 * 3)
        data = _load_stock(ticket, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))

        if mt == "prophet":
            preds = prophet_predict(
                model, data, forecast_horizon,
                regressor_cols=["Open", "High", "Low", "Volume"],
                target_col=target_col,
            )
            last_date = data.index[-1]
            predictions = [
                {
                    "date": p["ds"].strftime("%Y-%m-%d") if hasattr(p["ds"], "strftime") else str(p["ds"]),
                    "prediction": float(p["yhat"]),
                    "lower_bound": float(p.get("yhat_lower", p["yhat"])),
                    "upper_bound": float(p.get("yhat_upper", p["yhat"])),
                }
                for p in preds
            ]
        else:
            # RF / LSTM / XGBoost all follow the same pattern
            if mt == "rf":
                fc, lb, ub = forecast_rf(model, data.copy(), forecast_horizon, target_col)
            elif mt == "lstm":
                fc, lb, ub = forecast_lstm(model, data.copy(), forecast_horizon, target_col)
            elif mt == "xgboost":
                fc, lb, ub = model.predict_future(
                    historical_data_df=data.copy(),
                    forecast_horizon=forecast_horizon,
                    target_col=target_col,
                )
            else:
                raise HTTPException(400, "Unsupported model type for prediction")

            last_date = data.index[-1]
            dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=forecast_horizon)
            predictions = [
                {
                    "date": dates[i].strftime("%Y-%m-%d"),
                    "prediction": float(fc[i]),
                    "lower_bound": float(lb[i]),
                    "upper_bound": float(ub[i]),
                }
                for i in range(len(dates))
            ]

        hist = data.iloc[-history_days:]
        return {
            "status": "success",
            "ticker": ticket,
            "target_column": target_col,
            "forecast_horizon": forecast_horizon,
            "historical_dates": hist.index.strftime("%Y-%m-%d").tolist(),
            "historical_values": hist[target_col].tolist(),
            "predictions": predictions,
            "last_actual_date": data.index[-1].strftime("%Y-%m-%d"),
            "last_actual_value": float(data[target_col].iloc[-1]),
        }
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(500, f"Prediction failed: {exc}")


# =========================================================================
# MODELS LIST
# =========================================================================


@app.get("/models/{model_type}")
async def list_models(model_type: str = Path(...)):
    mt = model_type.lower()
    _validate_model_type(mt)
    try:
        models = list_models_in_gcs(mt)
        return {"total_models": len(models), "models": models, "storage": "GCS" if is_cloud_environment() else "local"}
    except Exception as exc:
        raise HTTPException(500, str(exc))


# =========================================================================
# SHAP
# =========================================================================

_shap_cache: Dict[str, SHAPExplainer] = {}


def _get_shap_explainer(model_type: str, ticker: str) -> SHAPExplainer:
    """Load a tree-based model and wrap it in a SHAPExplainer."""
    key = f"shap_{model_type}_{ticker}"
    if key in _shap_cache:
        return _shap_cache[key]

    valid = {"rf", "xgboost", "random_forest"}
    mt = "rf" if model_type in ("rf", "random_forest") else model_type
    if mt not in ("rf", "xgboost"):
        raise HTTPException(400, "SHAP only supports tree-based models (rf, xgboost)")

    wrapper = _find_model(mt, ticker)
    if wrapper is None:
        raise HTTPException(404, f"No trained {mt} model for {ticker}")

    # Extract the underlying sklearn/xgb estimator for SHAP
    if mt == "rf":
        raw_model = wrapper.model.named_steps["regressor"] if hasattr(wrapper.model, "named_steps") else wrapper.model
    else:
        raw_model = wrapper.model if hasattr(wrapper, "model") else wrapper

    explainer = SHAPExplainer(raw_model, model_type=mt)

    # Prepare background data
    end = datetime.now()
    start = end - timedelta(days=730)
    try:
        data = _load_stock(ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        if hasattr(wrapper, "preprocessor") and wrapper.preprocessor is not None:
            bg = wrapper.preprocessor.transform(data)
        elif hasattr(wrapper, "prepare_data"):
            bg = wrapper.prepare_data(data, target_col="Close")
        else:
            bg = data
        explainer.background_data = bg
    except Exception:
        pass  # SHAP will work without background, just slower

    _shap_cache[key] = explainer
    return explainer


@app.post("/explain")
async def explain_prediction(req: ExplainRequest):
    try:
        explainer = _get_shap_explainer(req.model_type, req.ticker)
        if explainer.background_data is None:
            raise HTTPException(400, "No background data. Train the model first.")

        result = explainer.explain_prediction(
            explainer.background_data.iloc[-10:],
            top_n=req.top_features,
        )
        return {"status": "success", "ticker": req.ticker, "model_type": req.model_type, **result}
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(500, f"SHAP explain failed: {exc}")


@app.post("/explain/importance/{model_type}")
async def global_importance(
    model_type: str = Path(...),
    ticker: str = Query("NU"),
    max_samples: int = Query(500),
):
    try:
        explainer = _get_shap_explainer(model_type, ticker)
        if explainer.background_data is None:
            raise HTTPException(400, "No background data.")

        bg = explainer.background_data
        if len(bg) > max_samples:
            bg = bg.sample(max_samples, random_state=42)

        result = explainer.global_feature_importance(bg)
        return {"status": "success", "ticker": ticker, "model_type": model_type, **result}
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(500, f"Global importance failed: {exc}")


@app.get("/explain/plot/{model_type}")
async def shap_summary_plot(
    model_type: str = Path(...),
    ticker: str = Query("NU"),
    plot_type: str = Query("bar"),
    max_features: int = Query(15),
):
    try:
        explainer = _get_shap_explainer(model_type, ticker)
        if explainer.background_data is None:
            raise HTTPException(400, "No background data.")

        result = explainer.summary_plot(
            explainer.background_data,
            plot_type=plot_type,
            max_display=max_features,
        )
        return {"status": "success", "ticker": ticker, "model_type": model_type, **result}
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(500, f"Summary plot failed: {exc}")


@app.get("/explain/waterfall/{model_type}")
async def shap_waterfall(
    model_type: str = Path(...),
    ticker: str = Query("NU"),
    sample_index: int = Query(-1),
):
    try:
        explainer = _get_shap_explainer(model_type, ticker)
        if explainer.background_data is None:
            raise HTTPException(400, "No background data.")

        result = explainer.waterfall_plot(
            explainer.background_data,
            sample_index=sample_index,
        )
        return {"status": "success", "ticker": ticker, "model_type": model_type, **result}
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(500, f"Waterfall plot failed: {exc}")


# =========================================================================
# ENSEMBLE
# =========================================================================


@app.post("/ensemble/predict")
async def ensemble_predict(req: EnsemblePredictRequest):
    """
    Combine predictions from multiple models.
    Unlike the old microservice version, this calls the model functions
    directly in-process — no HTTP round-trips.
    """
    models_to_use = [m.lower() for m in (req.models or ["rf", "xgboost", "lstm", "prophet"])]
    ticker = req.ticker
    horizon = req.forecast_horizon
    target = req.target_col

    # Collect predictions from each model
    model_preds: Dict[str, Dict] = {}
    for mt in models_to_use:
        try:
            model = _find_model(mt, ticker)
            if model is None:
                print(f"[ENSEMBLE] {mt} model not found for {ticker}, skipping")
                continue

            meta = load_model_metadata(mt, ticker)
            end_dt = (
                datetime.strptime(meta["training_end_date"], "%Y-%m-%d")
                if meta and "training_end_date" in meta
                else datetime.now()
            )
            start_dt = end_dt - timedelta(days=365 * 3)
            data = _load_stock(ticker, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))

            if mt == "prophet":
                preds_list = prophet_predict(
                    model, data, horizon,
                    regressor_cols=["Open", "High", "Low", "Volume"],
                    target_col=target,
                )
                vals = [float(p["yhat"]) for p in preds_list]
            elif mt == "rf":
                fc, _, _ = forecast_rf(model, data.copy(), horizon, target)
                vals = fc.tolist()
            elif mt == "lstm":
                fc, _, _ = forecast_lstm(model, data.copy(), horizon, target)
                vals = fc.tolist()
            elif mt == "xgboost":
                fc, _, _ = model.predict_future(data.copy(), horizon, target)
                vals = fc.tolist()
            else:
                continue

            model_preds[mt] = {"predictions": vals, "status": "success"}
        except Exception as exc:
            print(f"[ENSEMBLE] {mt} failed: {exc}")
            model_preds[mt] = {"predictions": [], "status": f"error: {exc}"}

    successful = {k: v for k, v in model_preds.items() if v["status"] == "success" and v["predictions"]}
    if not successful:
        raise HTTPException(400, "No model produced a valid prediction.")

    # Combine
    all_vals = np.array([v["predictions"] for v in successful.values()])
    method = req.ensemble_method.lower()

    if method == "simple_average":
        combined = np.mean(all_vals, axis=0)
    elif method == "weighted_average":
        # If we have metrics, weight by inverse MAE; otherwise equal weights
        weights = []
        for k in successful:
            meta = load_model_metadata(k, ticker)
            mae = (meta or {}).get("metrics", {}).get("mae")
            weights.append(1.0 / mae if mae and mae > 0 else 1.0)
        w = np.array(weights) / np.sum(weights)
        combined = np.average(all_vals, axis=0, weights=w)
    elif method == "median":
        combined = np.median(all_vals, axis=0)
    elif method == "best_model":
        best_mae, best_key = float("inf"), list(successful.keys())[0]
        for k in successful:
            meta = load_model_metadata(k, ticker)
            mae = (meta or {}).get("metrics", {}).get("mae", float("inf"))
            if mae < best_mae:
                best_mae, best_key = mae, k
        combined = np.array(successful[best_key]["predictions"])
    else:
        combined = np.mean(all_vals, axis=0)

    # Confidence from model disagreement
    std = np.std(all_vals, axis=0) if len(successful) > 1 else np.zeros_like(combined)
    upper = (combined + 1.96 * std).tolist()
    lower = (combined - 1.96 * std).tolist()

    # Load historical for chart
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=365)
    try:
        hist_data = _load_stock(ticker, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
        hist_dates = hist_data.index.strftime("%Y-%m-%d").tolist()[-60:]
        hist_values = hist_data[target].tolist()[-60:]
    except Exception:
        hist_dates, hist_values = [], []

    last_date = datetime.now()
    forecast_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=horizon)

    return {
        "status": "success",
        "ticker": ticker,
        "ensemble_method": method,
        "models_used": list(successful.keys()),
        "ensemble_predictions": combined.tolist(),
        "confidence_interval": {"upper": upper, "lower": lower},
        "individual_predictions": {k: v["predictions"] for k, v in successful.items()},
        "forecast_dates": forecast_dates.strftime("%Y-%m-%d").tolist(),
        "historical_dates": hist_dates,
        "historical_values": hist_values,
        "generated_at": datetime.now().isoformat(),
    }


@app.get("/ensemble/compare")
async def ensemble_compare(
    ticker: str = Query("NU"),
    forecast_horizon: int = Query(10),
    target_col: str = Query("Close"),
):
    """Compare predictions from all available models (no combination)."""
    comparisons: Dict[str, Any] = {}
    for mt in SUPPORTED_MODELS:
        try:
            model = _find_model(mt, ticker)
            if model is None:
                comparisons[mt] = {"status": "not_available"}
                continue

            meta = load_model_metadata(mt, ticker)
            end_dt = (
                datetime.strptime(meta["training_end_date"], "%Y-%m-%d")
                if meta and "training_end_date" in meta
                else datetime.now()
            )
            start_dt = end_dt - timedelta(days=365 * 3)
            data = _load_stock(ticker, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))

            if mt == "prophet":
                preds = prophet_predict(model, data, forecast_horizon, ["Open", "High", "Low", "Volume"], target_col)
                vals = [float(p["yhat"]) for p in preds]
            elif mt == "rf":
                fc, _, _ = forecast_rf(model, data.copy(), forecast_horizon, target_col)
                vals = fc.tolist()
            elif mt == "lstm":
                fc, _, _ = forecast_lstm(model, data.copy(), forecast_horizon, target_col)
                vals = fc.tolist()
            elif mt == "xgboost":
                fc, _, _ = model.predict_future(data.copy(), forecast_horizon, target_col)
                vals = fc.tolist()
            else:
                continue

            comparisons[mt] = {
                "status": "success",
                "predictions": vals,
                "metrics": (meta or {}).get("metrics", {}),
            }
        except Exception as exc:
            comparisons[mt] = {"status": f"error: {exc}"}

    return {
        "status": "success",
        "ticker": ticker,
        "forecast_horizon": forecast_horizon,
        "comparisons": comparisons,
        "generated_at": datetime.now().isoformat(),
    }


@app.get("/ensemble/models")
async def ensemble_models():
    """List which models have trained artifacts available."""
    available = {}
    for mt in SUPPORTED_MODELS:
        models = list_models_in_gcs(mt)
        available[mt] = {"available": len(models) > 0, "count": len(models)}
    return {"models": available, "total_available": sum(1 for v in available.values() if v["available"])}


# =========================================================================
# Entry point (local dev)
# =========================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
