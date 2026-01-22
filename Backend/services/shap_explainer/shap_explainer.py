"""
SHAP Explainer Module for Model Interpretability
Provides explanations for XGBoost and Random Forest predictions
"""
import shap
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
import matplotlib.pyplot as plt
import io
import base64
import warnings

warnings.filterwarnings('ignore')


class SHAPExplainer:
    """
    Unified SHAP explainer for tree-based models (XGBoost, Random Forest).
    
    Provides:
    - Feature importance (global)
    - Individual prediction explanations (local)
    - Visualization generation
    """
    
    def __init__(self, model, model_type: str = "xgboost"):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model (XGBoost or RandomForest)
            model_type: Either "xgboost" or "random_forest"
        """
        self.model_type = model_type.lower()
        self.model = model
        self.explainer = None
        self.feature_names = None
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize the appropriate SHAP explainer based on model type."""
        try:
            if self.model_type in ["xgboost", "xgb"]:
                # XGBoost has native SHAP support - very fast
                self.explainer = shap.TreeExplainer(self.model)
            elif self.model_type in ["random_forest", "rf"]:
                # RandomForest also uses TreeExplainer
                self.explainer = shap.TreeExplainer(self.model)
            else:
                # Fallback to KernelSHAP (slower but universal)
                raise ValueError(f"Model type {self.model_type} not directly supported. Use 'xgboost' or 'random_forest'")
        except Exception as e:
            print(f"Warning: Could not initialize TreeExplainer: {e}")
            print("Will use KernelExplainer as fallback (slower)")
            self.explainer = None
    
    def explain_prediction(
        self, 
        X: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Explain individual predictions with SHAP values.
        
        Args:
            X: Input features (single row or multiple rows)
            feature_names: Names of features for labeling
            
        Returns:
            Dictionary with SHAP values and explanations
        """
        if feature_names is not None:
            self.feature_names = feature_names
        elif isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # Convert to numpy if DataFrame
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Ensure 2D
        if X_array.ndim == 1:
            X_array = X_array.reshape(1, -1)
        
        # Calculate SHAP values
        if self.explainer is None:
            # Fallback: use KernelExplainer with background data
            background = shap.sample(X_array, min(100, len(X_array)))
            self.explainer = shap.KernelExplainer(self.model.predict, background)
        
        shap_values = self.explainer.shap_values(X_array)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # For multi-output models
            shap_values = shap_values[0] if len(shap_values) == 1 else shap_values
        
        # Get base value (expected value)
        if hasattr(self.explainer, 'expected_value'):
            base_value = self.explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = float(base_value[0]) if len(base_value) == 1 else base_value.tolist()
            else:
                base_value = float(base_value)
        else:
            base_value = None
        
        # Build feature importance for each prediction
        explanations = []
        for i in range(len(X_array)):
            row_shap = shap_values[i] if shap_values.ndim > 1 else shap_values
            
            # Create feature contribution list
            contributions = []
            for j, shap_val in enumerate(row_shap):
                feature_name = self.feature_names[j] if self.feature_names else f"feature_{j}"
                contributions.append({
                    "feature": feature_name,
                    "value": float(X_array[i, j]),
                    "shap_value": float(shap_val),
                    "impact": "positive" if shap_val > 0 else "negative"
                })
            
            # Sort by absolute SHAP value
            contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
            
            explanations.append({
                "prediction_index": i,
                "base_value": base_value,
                "contributions": contributions,
                "top_positive": [c for c in contributions if c["shap_value"] > 0][:5],
                "top_negative": [c for c in contributions if c["shap_value"] < 0][:5]
            })
        
        return {
            "explanations": explanations,
            "feature_names": self.feature_names,
            "shap_values_raw": shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
            "model_type": self.model_type
        }
    
    def get_global_importance(
        self, 
        X: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
        max_samples: int = 500
    ) -> Dict[str, Any]:
        """
        Calculate global feature importance using SHAP.
        
        Args:
            X: Training/test data for importance calculation
            feature_names: Names of features
            max_samples: Maximum samples to use (for speed)
            
        Returns:
            Dictionary with global feature importance
        """
        if feature_names is not None:
            self.feature_names = feature_names
        elif isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # Convert to numpy
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Subsample if too large
        if len(X_array) > max_samples:
            indices = np.random.choice(len(X_array), max_samples, replace=False)
            X_array = X_array[indices]
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_array)
        
        # Handle list output
        if isinstance(shap_values, list):
            shap_values = shap_values[0] if len(shap_values) == 1 else np.mean(shap_values, axis=0)
        
        # Calculate mean absolute SHAP value per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Create importance ranking
        importance_list = []
        for i, importance in enumerate(mean_abs_shap):
            feature_name = self.feature_names[i] if self.feature_names else f"feature_{i}"
            importance_list.append({
                "feature": feature_name,
                "importance": float(importance),
                "rank": 0  # Will be set after sorting
            })
        
        # Sort and assign ranks
        importance_list.sort(key=lambda x: x["importance"], reverse=True)
        for rank, item in enumerate(importance_list, 1):
            item["rank"] = rank
        
        # Calculate relative importance (percentage)
        total_importance = sum(item["importance"] for item in importance_list)
        for item in importance_list:
            item["relative_importance"] = round(item["importance"] / total_importance * 100, 2) if total_importance > 0 else 0
        
        return {
            "global_importance": importance_list,
            "total_features": len(importance_list),
            "top_5_features": importance_list[:5],
            "model_type": self.model_type
        }
    
    def generate_summary_plot(
        self, 
        X: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
        max_display: int = 15,
        plot_type: str = "bar"
    ) -> str:
        """
        Generate SHAP summary plot as base64 encoded image.
        
        Args:
            X: Input data
            feature_names: Feature names
            max_display: Maximum features to display
            plot_type: "bar" for bar plot, "dot" for beeswarm plot
            
        Returns:
            Base64 encoded PNG image
        """
        if feature_names is not None:
            self.feature_names = feature_names
        elif isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Subsample for speed
        if len(X_array) > 500:
            indices = np.random.choice(len(X_array), 500, replace=False)
            X_array = X_array[indices]
        
        shap_values = self.explainer.shap_values(X_array)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        if plot_type == "bar":
            shap.summary_plot(
                shap_values, 
                X_array, 
                feature_names=self.feature_names,
                plot_type="bar",
                max_display=max_display,
                show=False
            )
        else:
            shap.summary_plot(
                shap_values, 
                X_array, 
                feature_names=self.feature_names,
                max_display=max_display,
                show=False
            )
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return image_base64
    
    def generate_waterfall_plot(
        self, 
        X: Union[pd.DataFrame, np.ndarray],
        prediction_index: int = 0,
        feature_names: Optional[List[str]] = None,
        max_display: int = 10
    ) -> str:
        """
        Generate SHAP waterfall plot for a single prediction.
        
        Args:
            X: Input data
            prediction_index: Which prediction to explain
            feature_names: Feature names
            max_display: Maximum features to display
            
        Returns:
            Base64 encoded PNG image
        """
        if feature_names is not None:
            self.feature_names = feature_names
        elif isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        if X_array.ndim == 1:
            X_array = X_array.reshape(1, -1)
            prediction_index = 0
        
        # Get SHAP explanation object
        shap_values = self.explainer(X_array)
        
        # Create waterfall plot
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values[prediction_index], max_display=max_display, show=False)
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return image_base64
    
    def generate_force_plot(
        self, 
        X: Union[pd.DataFrame, np.ndarray],
        prediction_index: int = 0,
        feature_names: Optional[List[str]] = None
    ) -> str:
        """
        Generate SHAP force plot for a single prediction (as HTML).
        
        Args:
            X: Input data
            prediction_index: Which prediction to explain
            feature_names: Feature names
            
        Returns:
            HTML string for the force plot
        """
        if feature_names is not None:
            self.feature_names = feature_names
        elif isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        if X_array.ndim == 1:
            X_array = X_array.reshape(1, -1)
            prediction_index = 0
        
        shap_values = self.explainer.shap_values(X_array)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Generate force plot HTML
        force_plot = shap.force_plot(
            self.explainer.expected_value if not isinstance(self.explainer.expected_value, np.ndarray) else self.explainer.expected_value[0],
            shap_values[prediction_index],
            X_array[prediction_index],
            feature_names=self.feature_names,
            matplotlib=False
        )
        
        return shap.getjs() + force_plot.html()


def explain_xgboost_prediction(model, X, feature_names=None):
    """Convenience function for XGBoost explanations."""
    explainer = SHAPExplainer(model, model_type="xgboost")
    return explainer.explain_prediction(X, feature_names)


def explain_rf_prediction(model, X, feature_names=None):
    """Convenience function for Random Forest explanations."""
    explainer = SHAPExplainer(model, model_type="random_forest")
    return explainer.explain_prediction(X, feature_names)
