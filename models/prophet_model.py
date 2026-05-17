"""
Prophet Model — seasonal baseline for PM2.5 forecasting.
Facebook Prophet captures yearly, weekly, and daily seasonality with Nepal-specific regressors.
"""

import logging
from typing import Any

import pandas as pd
import numpy as np
from prophet import Prophet

logger = logging.getLogger(__name__)


class ProphetAQModel:
    """Prophet-based PM2.5 forecasting model."""

    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True,
        changepoint_prior_scale: float = 0.05,
    ):
        self.config = {
            "yearly_seasonality": yearly_seasonality,
            "weekly_seasonality": weekly_seasonality,
            "daily_seasonality": daily_seasonality,
            "changepoint_prior_scale": changepoint_prior_scale,
        }
        self.model: Prophet | None = None
        self.regressors = [
            "temp_c", "humidity_pct", "wind_speed_kmh",
            "is_tihar", "is_dashain", "is_monsoon", "is_brick_kiln_season",
        ]

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data in Prophet format (ds, y, regressors)."""
        prophet_df = pd.DataFrame()
        prophet_df["ds"] = pd.to_datetime(df["timestamp_utc"])
        prophet_df["y"] = df["pm25"].values

        for reg in self.regressors:
            if reg in df.columns:
                prophet_df[reg] = df[reg].values.astype(float)
            else:
                prophet_df[reg] = 0.0

        # Drop rows with missing target
        prophet_df = prophet_df.dropna(subset=["y"])
        return prophet_df

    def train(self, train_df: pd.DataFrame) -> dict[str, Any]:
        """
        Train the Prophet model.

        Args:
            train_df: DataFrame with columns matching feature store schema.

        Returns:
            Dict of training metrics.
        """
        prophet_df = self.prepare_data(train_df)

        self.model = Prophet(
            yearly_seasonality=self.config["yearly_seasonality"],
            weekly_seasonality=self.config["weekly_seasonality"],
            daily_seasonality=self.config["daily_seasonality"],
            changepoint_prior_scale=self.config["changepoint_prior_scale"],
        )

        # Add regressors
        for reg in self.regressors:
            self.model.add_regressor(reg)

        self.model.fit(prophet_df)

        # Compute training metrics
        train_pred = self.model.predict(prophet_df)
        train_rmse = np.sqrt(np.mean((prophet_df["y"].values - train_pred["yhat"].values) ** 2))
        train_mae = np.mean(np.abs(prophet_df["y"].values - train_pred["yhat"].values))

        metrics = {
            "train_rmse": float(train_rmse),
            "train_mae": float(train_mae),
            "n_train_samples": len(prophet_df),
        }
        logger.info(f"Prophet training complete: RMSE={train_rmse:.2f}, MAE={train_mae:.2f}")
        return metrics

    def predict(self, future_df: pd.DataFrame, hours: int = 24) -> pd.DataFrame:
        """
        Generate PM2.5 forecast.

        Args:
            future_df: DataFrame with future timestamps and regressor values.
            hours: Number of hours to forecast.

        Returns:
            DataFrame with columns: ds, yhat, yhat_lower, yhat_upper.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        prophet_future = self.prepare_data(future_df) if not future_df.empty else self._make_future(hours)
        forecast = self.model.predict(prophet_future)

        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(hours)

    def _make_future(self, hours: int) -> pd.DataFrame:
        """Create future dataframe for prediction when no future data is available."""
        if self.model is None:
            raise RuntimeError("Model not trained.")

        future = self.model.make_future_dataframe(periods=hours, freq="h")
        # Fill regressors with last known values
        for reg in self.regressors:
            future[reg] = 0.0
        return future.tail(hours)

    def evaluate(self, val_df: pd.DataFrame) -> dict[str, Any]:
        """Evaluate model on validation data."""
        prophet_df = self.prepare_data(val_df)
        predictions = self.model.predict(prophet_df)

        y_true = prophet_df["y"].values
        y_pred = predictions["yhat"].values

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        metrics = {
            "val_rmse": float(rmse),
            "val_mae": float(mae),
            "val_mape": float(mape),
            "val_r2": float(r2),
            "n_val_samples": len(prophet_df),
        }
        logger.info(f"Prophet validation: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.1f}%")
        return metrics

    def get_hyperparameters(self) -> dict[str, Any]:
        """Return model hyperparameters for MLflow logging."""
        return {
            **self.config,
            "regressors": ",".join(self.regressors),
            "model_type": "prophet",
        }
