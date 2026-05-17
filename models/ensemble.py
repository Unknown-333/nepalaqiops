"""
Ensemble Model — weighted combination of Prophet and LSTM forecasts.
Weights are configurable via environment variables and recalculated during retraining.
"""

import os
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class EnsembleModel:
    """Weighted ensemble of Prophet + LSTM forecasts."""

    def __init__(
        self,
        prophet_weight: float | None = None,
        lstm_weight: float | None = None,
    ):
        self.prophet_weight = prophet_weight or float(os.getenv("PROPHET_WEIGHT", "0.4"))
        self.lstm_weight = lstm_weight or float(os.getenv("LSTM_WEIGHT", "0.6"))

        # Normalize weights to sum to 1
        total = self.prophet_weight + self.lstm_weight
        self.prophet_weight /= total
        self.lstm_weight /= total

    def predict(
        self,
        prophet_forecast: np.ndarray,
        lstm_forecast: np.ndarray,
    ) -> np.ndarray:
        """
        Combine Prophet and LSTM forecasts.

        Args:
            prophet_forecast: Array of shape (24,) from Prophet.
            lstm_forecast: Array of shape (24,) from LSTM.

        Returns:
            Weighted ensemble forecast of shape (24,).
        """
        ensemble = (
            self.prophet_weight * prophet_forecast +
            self.lstm_weight * lstm_forecast
        )
        return ensemble

    def optimize_weights(
        self,
        prophet_predictions: np.ndarray,
        lstm_predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> tuple[float, float]:
        """
        Find optimal weights that minimize RMSE on validation data.
        Uses grid search over weight space.

        Args:
            prophet_predictions: Prophet predictions on val set.
            lstm_predictions: LSTM predictions on val set.
            actuals: Actual values on val set.

        Returns:
            Tuple of (optimal_prophet_weight, optimal_lstm_weight).
        """
        best_rmse = float("inf")
        best_prophet_w = self.prophet_weight

        for w in np.arange(0.0, 1.05, 0.05):
            ensemble_pred = w * prophet_predictions + (1 - w) * lstm_predictions
            rmse = np.sqrt(np.mean((actuals - ensemble_pred) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_prophet_w = w

        best_lstm_w = 1.0 - best_prophet_w

        self.prophet_weight = best_prophet_w
        self.lstm_weight = best_lstm_w

        logger.info(
            f"Optimal ensemble weights: Prophet={best_prophet_w:.2f}, "
            f"LSTM={best_lstm_w:.2f} (RMSE={best_rmse:.2f})"
        )
        return best_prophet_w, best_lstm_w

    def get_confidence_interval(
        self,
        prophet_lower: np.ndarray,
        prophet_upper: np.ndarray,
        lstm_forecast: np.ndarray,
        lstm_std: float = 10.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute ensemble confidence intervals.
        Uses Prophet's built-in intervals weighted with LSTM uncertainty estimate.
        """
        # Simple approach: blend Prophet CI with LSTM +/- estimated std
        lstm_lower = lstm_forecast - 1.96 * lstm_std
        lstm_upper = lstm_forecast + 1.96 * lstm_std

        ensemble_lower = self.prophet_weight * prophet_lower + self.lstm_weight * lstm_lower
        ensemble_upper = self.prophet_weight * prophet_upper + self.lstm_weight * lstm_upper

        return ensemble_lower, ensemble_upper

    def get_weights(self) -> dict[str, float]:
        """Return current weights."""
        return {
            "prophet_weight": self.prophet_weight,
            "lstm_weight": self.lstm_weight,
        }
