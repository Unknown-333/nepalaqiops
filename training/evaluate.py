"""
Model Evaluation — computes RMSE, MAE, MAPE vs rolling baseline.
Used for champion/challenger comparison and drift detection.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates models and compares against rolling baseline."""

    def __init__(self, rmse_degradation_threshold: float = 0.15):
        self.rmse_degradation_threshold = rmse_degradation_threshold

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """Compute standard regression metrics."""
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            return {"rmse": float("nan"), "mae": float("nan"), "mape": float("nan"), "r2": float("nan")}

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape),
            "r2": float(r2),
        }

    def evaluate_per_horizon(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        horizon: int = 24,
    ) -> dict[str, float]:
        """Compute RMSE per forecast horizon (h+1 through h+24)."""
        metrics = {}
        for h in range(min(horizon, y_true.shape[1] if y_true.ndim > 1 else 1)):
            if y_true.ndim > 1:
                h_true = y_true[:, h]
                h_pred = y_pred[:, h]
            else:
                h_true = y_true
                h_pred = y_pred
            h_rmse = np.sqrt(np.mean((h_true - h_pred) ** 2))
            metrics[f"rmse_h{h + 1}"] = float(h_rmse)
        return metrics

    def compare_models(
        self,
        champion_rmse: float,
        challenger_rmse: float,
    ) -> dict[str, Any]:
        """
        Compare challenger vs champion performance.

        Returns:
            Dict with comparison result and whether to promote.
        """
        if np.isnan(champion_rmse) or np.isnan(challenger_rmse):
            return {"should_promote": False, "reason": "Missing metrics"}

        improvement = (champion_rmse - challenger_rmse) / champion_rmse
        should_promote = challenger_rmse < champion_rmse

        result = {
            "champion_rmse": champion_rmse,
            "challenger_rmse": challenger_rmse,
            "improvement_pct": float(improvement * 100),
            "should_promote": should_promote,
            "reason": (
                f"Challenger RMSE ({challenger_rmse:.2f}) is "
                f"{'better' if should_promote else 'worse'} than "
                f"champion ({champion_rmse:.2f}). "
                f"{'Promoting.' if should_promote else 'Keeping current champion.'}"
            ),
        }

        logger.info(result["reason"])
        return result

    def check_degradation(
        self,
        current_rmse: float,
        baseline_rmse: float,
    ) -> dict[str, Any]:
        """
        Check if model RMSE has degraded beyond threshold.
        Triggers retraining if degradation > 15%.
        """
        if np.isnan(baseline_rmse) or baseline_rmse == 0:
            return {"degraded": False, "reason": "No valid baseline"}

        degradation = (current_rmse - baseline_rmse) / baseline_rmse

        result = {
            "current_rmse": float(current_rmse),
            "baseline_rmse": float(baseline_rmse),
            "degradation_pct": float(degradation * 100),
            "threshold_pct": float(self.rmse_degradation_threshold * 100),
            "degraded": degradation > self.rmse_degradation_threshold,
        }

        if result["degraded"]:
            result["reason"] = (
                f"Model degraded by {degradation * 100:.1f}% "
                f"(threshold: {self.rmse_degradation_threshold * 100}%). "
                f"Triggering retraining."
            )
            logger.warning(result["reason"])
        else:
            result["reason"] = (
                f"Model performance within bounds: "
                f"{degradation * 100:.1f}% degradation (threshold: {self.rmse_degradation_threshold * 100}%)"
            )

        return result

    def compute_rolling_baseline(
        self,
        historical_rmse: list[float],
        window: int = 7,
    ) -> float:
        """Compute rolling baseline RMSE from historical evaluations."""
        if not historical_rmse:
            return float("nan")
        recent = historical_rmse[-window:]
        return float(np.mean(recent))
