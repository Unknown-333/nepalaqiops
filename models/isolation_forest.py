"""
Isolation Forest — anomaly detection for AQI sensor readings.
Detects pollution spikes, sensor faults, and unusual patterns.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest as SklearnIF

logger = logging.getLogger(__name__)

ANOMALY_FEATURES = ["pm25", "pm10", "no2", "pm25_1h_mean", "pm25_6h_mean"]


class AnomalyDetector:
    """Isolation Forest-based anomaly detection for AQI data."""

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model: SklearnIF | None = None
        self.feature_columns = ANOMALY_FEATURES

    def train(self, df: pd.DataFrame) -> dict[str, Any]:
        """Train the Isolation Forest on historical data."""
        features = self._prepare_features(df)

        self.model = SklearnIF(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
            max_samples="auto",
        )
        self.model.fit(features)

        # Score training data
        scores = self.model.decision_function(features)
        predictions = self.model.predict(features)
        n_anomalies = (predictions == -1).sum()

        metrics = {
            "n_train_samples": len(features),
            "n_anomalies_detected": int(n_anomalies),
            "anomaly_rate": float(n_anomalies / len(features)),
            "mean_anomaly_score": float(scores.mean()),
            "contamination": self.contamination,
        }

        logger.info(
            f"Isolation Forest trained: {n_anomalies}/{len(features)} anomalies "
            f"({metrics['anomaly_rate']:.1%})"
        )
        return metrics

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score new readings for anomalies.

        Returns DataFrame with anomaly_score and is_anomaly columns added.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        result = df.copy()
        features = self._prepare_features(df)

        if features.empty:
            result["anomaly_score"] = np.nan
            result["is_anomaly"] = False
            return result

        scores = self.model.decision_function(features)
        predictions = self.model.predict(features)

        result["anomaly_score"] = scores
        result["is_anomaly"] = predictions == -1

        return result

    def detect_sensor_faults(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Detect sensor faults:
        - Reading = 0 for >3 consecutive hours
        - Reading > 999 (sensor malfunction)
        """
        faults = []

        if "pm25" not in df.columns:
            return faults

        # Check for stuck-at-zero (>3 hours)
        if len(df) >= 3:
            zero_streak = 0
            for i, row in df.iterrows():
                if row.get("pm25", -1) == 0:
                    zero_streak += 1
                    if zero_streak >= 3:
                        faults.append({
                            "station_id": row.get("station_id", "unknown"),
                            "timestamp_utc": str(row.get("timestamp_utc", "")),
                            "metric": "pm25",
                            "value": 0.0,
                            "anomaly_score": -1.0,
                            "alert_type": "sensor_fault_stuck_zero",
                        })
                else:
                    zero_streak = 0

        # Check for impossibly high readings
        high_readings = df[df["pm25"] > 999]
        for _, row in high_readings.iterrows():
            faults.append({
                "station_id": row.get("station_id", "unknown"),
                "timestamp_utc": str(row.get("timestamp_utc", "")),
                "metric": "pm25",
                "value": float(row["pm25"]),
                "anomaly_score": -1.0,
                "alert_type": "sensor_fault_impossible_value",
            })

        return faults

    def generate_alerts(self, scored_df: pd.DataFrame) -> list[dict[str, Any]]:
        """Generate alert records for detected anomalies."""
        alerts = []
        anomalies = scored_df[scored_df.get("is_anomaly", False) == True]

        for _, row in anomalies.iterrows():
            alert = {
                "station_id": row.get("station_id", "unknown"),
                "timestamp_utc": str(row.get("timestamp_utc", "")),
                "metric": "pm25",
                "value": float(row.get("pm25", 0)),
                "anomaly_score": float(row.get("anomaly_score", 0)),
                "alert_type": "statistical_anomaly",
            }
            alerts.append(alert)

        return alerts

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature matrix for Isolation Forest."""
        available = [f for f in self.feature_columns if f in df.columns]
        if not available:
            return pd.DataFrame()

        features = df[available].copy()
        features = features.fillna(0)
        return features

    def get_hyperparameters(self) -> dict[str, Any]:
        """Return hyperparameters for MLflow logging."""
        return {
            "model_type": "isolation_forest",
            "contamination": self.contamination,
            "n_estimators": 100,
            "features": ",".join(self.feature_columns),
        }
