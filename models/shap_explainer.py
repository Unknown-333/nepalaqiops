"""
SHAP Explainability — explains model predictions using SHAP values.
Surfaces "What drove today's PM2.5 spike?" insights.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """SHAP-based explainability for AQI model predictions."""

    def __init__(self):
        self.tree_explainer = None
        self.deep_explainer = None

    def explain_isolation_forest(
        self,
        model,
        X: pd.DataFrame | np.ndarray,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Explain Isolation Forest predictions using TreeExplainer.

        Returns:
            Dict with shap_values, feature_importance, and top_contributors.
        """
        import shap

        self.tree_explainer = shap.TreeExplainer(model)
        shap_values = self.tree_explainer.shap_values(X)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # Mean absolute SHAP values per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance_order = np.argsort(mean_abs_shap)[::-1]

        feature_importance = {
            feature_names[i]: float(mean_abs_shap[i])
            for i in importance_order
        }

        # Top contributors for latest prediction
        if len(shap_values) > 0:
            latest_shap = shap_values[-1]
            top_indices = np.argsort(np.abs(latest_shap))[::-1][:10]
            top_contributors = [
                {
                    "feature": feature_names[i],
                    "shap_value": float(latest_shap[i]),
                    "contribution_direction": "increases" if latest_shap[i] > 0 else "decreases",
                }
                for i in top_indices
            ]
        else:
            top_contributors = []

        return {
            "shap_values": shap_values,
            "feature_importance": feature_importance,
            "top_contributors": top_contributors,
        }

    def explain_lstm(
        self,
        model,
        X_sample: np.ndarray,
        X_background: np.ndarray,
        feature_names: list[str] | None = None,
        n_samples: int = 100,
    ) -> dict[str, Any]:
        """
        Explain LSTM predictions using DeepExplainer.

        Args:
            model: Trained Keras LSTM model.
            X_sample: Sample data to explain, shape (n, seq_len, features).
            X_background: Background data for SHAP, shape (n_bg, seq_len, features).
            feature_names: Names of input features.
            n_samples: Max samples to explain.

        Returns:
            Dict with shap_values, feature_importance, and interpretation.
        """
        import shap

        # Limit samples for computational efficiency
        if X_sample.shape[0] > n_samples:
            X_sample = X_sample[:n_samples]
        if X_background.shape[0] > 50:
            X_background = X_background[:50]

        self.deep_explainer = shap.DeepExplainer(model, X_background)
        shap_values = self.deep_explainer.shap_values(X_sample)

        # Average over sequence dimension to get per-feature importance
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Shape: (n_samples, seq_len, n_features) → average over samples and time
        mean_abs_shap = np.abs(shap_values).mean(axis=(0, 1))

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(mean_abs_shap.shape[0])]

        importance_order = np.argsort(mean_abs_shap)[::-1]
        feature_importance = {
            feature_names[i]: float(mean_abs_shap[i])
            for i in importance_order
        }

        return {
            "shap_values": shap_values,
            "feature_importance": feature_importance,
            "top_features": [feature_names[i] for i in importance_order[:10]],
        }

    def generate_interpretation(
        self,
        top_contributors: list[dict[str, Any]],
        prediction_value: float,
    ) -> str:
        """
        Generate plain English interpretation of SHAP values.
        e.g., "Tihar festival fireworks contributed +18.3 µg/m³ to today's reading"
        """
        interpretations = []

        feature_descriptions = {
            "is_tihar": "Tihar festival fireworks",
            "is_dashain": "Dashain traffic surge",
            "is_brick_kiln_season": "Brick kiln operations",
            "is_monsoon": "Monsoon season (dust suppression)",
            "is_pre_monsoon": "Pre-monsoon dry conditions",
            "temp_c": "Temperature",
            "humidity_pct": "Humidity level",
            "wind_speed_kmh": "Wind speed",
            "pm25_lag_24h": "Yesterday's PM2.5 level",
            "pm25_24h_mean": "24-hour rolling average",
            "precip_mm": "Precipitation (rain washout)",
            "pressure_hpa": "Atmospheric pressure (inversion)",
        }

        for contrib in top_contributors[:5]:
            feature = contrib["feature"]
            shap_val = contrib["shap_value"]
            desc = feature_descriptions.get(feature, feature.replace("_", " "))

            if abs(shap_val) < 0.5:
                continue

            direction = "increased" if shap_val > 0 else "decreased"
            interpretations.append(
                f"{desc} {direction} PM2.5 by {abs(shap_val):.1f} µg/m³"
            )

        if not interpretations:
            return f"Current PM2.5 prediction: {prediction_value:.1f} µg/m³ (no dominant single factor)"

        summary = f"Predicted PM2.5: {prediction_value:.1f} µg/m³. Key drivers: "
        summary += "; ".join(interpretations) + "."
        return summary
