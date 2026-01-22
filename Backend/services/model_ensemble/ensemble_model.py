"""
Ensemble Model for Stock Price Prediction
Combines predictions from RF, LSTM, XGBoost, and Prophet models
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json


class EnsembleMethod(Enum):
    """Available ensemble methods."""
    SIMPLE_AVERAGE = "simple_average"
    WEIGHTED_AVERAGE = "weighted_average"
    MEDIAN = "median"
    STACKING = "stacking"
    BEST_MODEL = "best_model"


@dataclass
class ModelPrediction:
    """Container for individual model prediction."""
    model_name: str
    predictions: np.ndarray
    confidence: Optional[float] = None
    metrics: Optional[Dict[str, float]] = None


class EnsembleModel:
    """
    Ensemble model that combines predictions from multiple models.
    
    Supported models:
    - Random Forest (RF)
    - LSTM (Deep Learning)
    - XGBoost (Gradient Boosting)
    - Prophet (Time Series)
    
    Ensemble strategies:
    - Simple Average: Equal weight to all models
    - Weighted Average: Weights based on model performance
    - Median: Robust to outliers
    - Stacking: Meta-learner on top of base predictions
    - Best Model: Use only the best performing model
    """
    
    def __init__(
        self, 
        method: EnsembleMethod = EnsembleMethod.WEIGHTED_AVERAGE,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize ensemble model.
        
        Args:
            method: Ensemble combination method
            weights: Model weights for weighted average (default: equal weights)
        """
        self.method = method
        self.weights = weights or {}
        self.model_predictions: Dict[str, ModelPrediction] = {}
        self.meta_model = None
        self.performance_history: Dict[str, List[float]] = {}
        
    def add_prediction(
        self, 
        model_name: str, 
        predictions: np.ndarray,
        confidence: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        Add a model's predictions to the ensemble.
        
        Args:
            model_name: Name of the model (e.g., "rf", "lstm", "xgboost", "prophet")
            predictions: Array of predictions
            confidence: Model confidence score (0-1)
            metrics: Performance metrics (mae, rmse, r2, etc.)
        """
        self.model_predictions[model_name] = ModelPrediction(
            model_name=model_name,
            predictions=np.array(predictions),
            confidence=confidence,
            metrics=metrics
        )
        
    def _calculate_weights(self) -> Dict[str, float]:
        """
        Calculate model weights based on performance metrics.
        
        Uses inverse MAE weighting - models with lower MAE get higher weight.
        """
        if self.weights:
            return self.weights
            
        # Calculate weights from metrics
        weights = {}
        total_inverse_mae = 0
        
        for name, pred in self.model_predictions.items():
            if pred.metrics and 'mae' in pred.metrics:
                mae = pred.metrics['mae']
                # Inverse MAE weighting (add small epsilon to avoid division by zero)
                inverse_mae = 1 / (mae + 1e-6)
                weights[name] = inverse_mae
                total_inverse_mae += inverse_mae
            else:
                # Equal weight if no metrics
                weights[name] = 1
                total_inverse_mae += 1
        
        # Normalize weights to sum to 1
        for name in weights:
            weights[name] /= total_inverse_mae
            
        return weights
    
    def _validate_predictions(self):
        """Validate that all predictions have the same length."""
        if not self.model_predictions:
            raise ValueError("No predictions added to ensemble")
            
        lengths = [len(p.predictions) for p in self.model_predictions.values()]
        if len(set(lengths)) > 1:
            raise ValueError(f"Prediction lengths don't match: {dict(zip(self.model_predictions.keys(), lengths))}")
    
    def combine(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Combine predictions using the specified method.
        
        Returns:
            Tuple of (combined_predictions, metadata)
        """
        self._validate_predictions()
        
        # Stack all predictions
        all_preds = np.array([p.predictions for p in self.model_predictions.values()])
        model_names = list(self.model_predictions.keys())
        
        metadata = {
            "method": self.method.value,
            "models_used": model_names,
            "num_models": len(model_names)
        }
        
        if self.method == EnsembleMethod.SIMPLE_AVERAGE:
            combined = np.mean(all_preds, axis=0)
            weights_used = {name: 1/len(model_names) for name in model_names}
            
        elif self.method == EnsembleMethod.WEIGHTED_AVERAGE:
            weights = self._calculate_weights()
            weight_array = np.array([weights[name] for name in model_names])
            combined = np.average(all_preds, axis=0, weights=weight_array)
            weights_used = weights
            
        elif self.method == EnsembleMethod.MEDIAN:
            combined = np.median(all_preds, axis=0)
            weights_used = {name: "N/A (median)" for name in model_names}
            
        elif self.method == EnsembleMethod.BEST_MODEL:
            # Select the model with lowest MAE
            best_model = min(
                self.model_predictions.items(),
                key=lambda x: x[1].metrics.get('mae', float('inf')) if x[1].metrics else float('inf')
            )
            combined = best_model[1].predictions
            weights_used = {name: 1.0 if name == best_model[0] else 0.0 for name in model_names}
            metadata["best_model"] = best_model[0]
            
        else:
            raise ValueError(f"Unsupported ensemble method: {self.method}")
        
        metadata["weights"] = weights_used
        
        # Calculate ensemble uncertainty (std across models)
        metadata["uncertainty"] = float(np.mean(np.std(all_preds, axis=0)))
        metadata["prediction_std"] = np.std(all_preds, axis=0).tolist()
        
        return combined, metadata
    
    def predict_with_confidence(self) -> Dict[str, Any]:
        """
        Generate ensemble prediction with confidence intervals.
        
        Returns:
            Dictionary with predictions, confidence bounds, and metadata
        """
        self._validate_predictions()
        
        all_preds = np.array([p.predictions for p in self.model_predictions.values()])
        
        combined, metadata = self.combine()
        
        # Calculate confidence intervals (using model disagreement)
        std = np.std(all_preds, axis=0)
        
        return {
            "predictions": combined.tolist(),
            "lower_bound": (combined - 1.96 * std).tolist(),  # 95% CI
            "upper_bound": (combined + 1.96 * std).tolist(),
            "confidence_level": 0.95,
            "model_agreement": float(1 - np.mean(std) / (np.mean(combined) + 1e-6)),  # Higher = more agreement
            "metadata": metadata
        }
    
    def get_model_contributions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get contribution details for each model in the ensemble.
        
        Returns:
            Dictionary with model contributions and statistics
        """
        contributions = {}
        weights = self._calculate_weights()
        
        for name, pred in self.model_predictions.items():
            contributions[name] = {
                "weight": weights.get(name, 0),
                "mean_prediction": float(np.mean(pred.predictions)),
                "std_prediction": float(np.std(pred.predictions)),
                "min_prediction": float(np.min(pred.predictions)),
                "max_prediction": float(np.max(pred.predictions)),
                "metrics": pred.metrics,
                "confidence": pred.confidence
            }
            
        return contributions


def create_ensemble_from_services(
    predictions: Dict[str, Dict[str, Any]],
    method: str = "weighted_average"
) -> Dict[str, Any]:
    """
    Convenience function to create ensemble from service responses.
    
    Args:
        predictions: Dictionary of model predictions with format:
            {
                "rf": {"predictions": [...], "metrics": {...}},
                "lstm": {"predictions": [...], "metrics": {...}},
                ...
            }
        method: Ensemble method to use
        
    Returns:
        Ensemble prediction result
    """
    ensemble = EnsembleModel(method=EnsembleMethod(method))
    
    for model_name, data in predictions.items():
        if "predictions" in data:
            preds = data["predictions"]
            # Handle nested prediction format
            if isinstance(preds, list) and len(preds) > 0:
                if isinstance(preds[0], dict) and "prediction" in preds[0]:
                    preds = [p["prediction"] for p in preds]
            
            ensemble.add_prediction(
                model_name=model_name,
                predictions=np.array(preds),
                metrics=data.get("metrics")
            )
    
    result = ensemble.predict_with_confidence()
    result["model_contributions"] = ensemble.get_model_contributions()
    
    return result
