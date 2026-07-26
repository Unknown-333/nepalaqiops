"""
LSTM Model — event-spike capture for PM2.5 forecasting.
Captures non-linear temporal patterns and pollution spike events.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Feature columns used by the LSTM (order matters for input tensor)
LSTM_FEATURES = [
    "pm25", "pm10", "no2",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "pm25_1h_mean", "pm25_3h_mean", "pm25_6h_mean", "pm25_12h_mean", "pm25_24h_mean",
    "pm25_1h_std", "pm25_6h_std", "pm25_24h_std",
    "pm25_lag_1h", "pm25_lag_3h", "pm25_lag_6h", "pm25_lag_12h", "pm25_lag_24h",
    "temp_c", "humidity_pct", "wind_speed_kmh", "wind_dir_sin", "wind_dir_cos",
    "precip_mm", "pressure_hpa",
    "is_tihar", "is_dashain", "is_monsoon", "is_pre_monsoon", "is_brick_kiln_season",
]

SEQUENCE_LENGTH = 48  # 48 hours lookback
FORECAST_HORIZON = 24  # 24 hours ahead


class LSTMAQModel:
    """LSTM-based PM2.5 forecasting model."""

    def __init__(
        self,
        sequence_length: int = SEQUENCE_LENGTH,
        forecast_horizon: int = FORECAST_HORIZON,
        lstm_units_1: int = 128,
        lstm_units_2: int = 64,
        dense_units: int = 32,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        max_epochs: int = 100,
        patience: int = 10,
    ):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.lstm_units_1 = lstm_units_1
        self.lstm_units_2 = lstm_units_2
        self.dense_units = dense_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.model: Any = None
        self.scaler: Any = None
        self.feature_columns = LSTM_FEATURES
        self.history: Any = None

    def _build_model(self, n_features: int):
        """Build the LSTM architecture."""
        import tensorflow as tf
        from tensorflow.keras import layers, models

        model = models.Sequential([
            layers.LSTM(
                self.lstm_units_1,
                return_sequences=True,
                dropout=self.dropout,
                input_shape=(self.sequence_length, n_features),
            ),
            layers.LSTM(
                self.lstm_units_2,
                return_sequences=False,
                dropout=self.dropout,
            ),
            layers.Dense(self.dense_units, activation="relu"),
            layers.Dense(self.forecast_horizon),
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.Huber(),
            metrics=["mae"],
        )

        self.model = model
        logger.info(f"LSTM model built: {model.count_params()} parameters")
        return model

    def prepare_sequences(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare input sequences and target arrays.
        IMPORTANT: Chronological order preserved — NEVER shuffle time-series.
        """
        from sklearn.preprocessing import StandardScaler

        # Select and fill features
        available_features = [f for f in self.feature_columns if f in df.columns]
        data = df[available_features].copy()
        data = data.fillna(method="ffill").fillna(0)

        # Scale features
        if self.scaler is None:
            self.scaler = StandardScaler()
            scaled_data = self.scaler.fit_transform(data)
        else:
            scaled_data = self.scaler.transform(data)

        # Create sequences
        X, y = [], []

        for i in range(len(scaled_data) - self.sequence_length - self.forecast_horizon + 1):
            X.append(scaled_data[i:i + self.sequence_length])
            # Target: next 24 hours of PM2.5 (in original scale)
            y.append(data.iloc[i + self.sequence_length:i + self.sequence_length + self.forecast_horizon]["pm25"].values)

        return np.array(X), np.array(y)

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> dict[str, Any]:
        """
        Train the LSTM model.

        Args:
            train_df: Training data (chronologically ordered).
            val_df: Optional validation data.

        Returns:
            Dict of training metrics.
        """
        import tensorflow as tf

        X_train, y_train = self.prepare_sequences(train_df)

        if X_train.shape[0] == 0:
            raise ValueError("Not enough data to create sequences")

        n_features = X_train.shape[2]
        self._build_model(n_features)

        # Prepare validation data
        validation_data = None
        if val_df is not None and not val_df.empty:
            X_val, y_val = self.prepare_sequences(val_df)
            if X_val.shape[0] > 0:
                validation_data = (X_val, y_val)

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=self.patience,
                restore_best_weights=True,
                monitor="val_loss" if validation_data else "loss",
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=5, min_lr=1e-6,
                monitor="val_loss" if validation_data else "loss",
            ),
        ]

        # Train (chronological — no shuffle!)
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.max_epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            shuffle=False,  # CRITICAL: never shuffle time-series
            verbose=1,
        )

        # Compute metrics
        train_pred = self.model.predict(X_train, batch_size=self.batch_size)
        train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))

        metrics = {
            "train_rmse": float(train_rmse),
            "n_train_samples": len(X_train),
            "n_epochs_trained": len(self.history.history["loss"]),
            "final_lr": float(self.model.optimizer.learning_rate.numpy()),
        }

        if validation_data:
            val_pred = self.model.predict(X_val, batch_size=self.batch_size)
            val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
            metrics["val_rmse"] = float(val_rmse)

            # Per-horizon RMSE
            for h in range(self.forecast_horizon):
                h_rmse = np.sqrt(np.mean((y_val[:, h] - val_pred[:, h]) ** 2))
                metrics[f"val_rmse_h{h + 1}"] = float(h_rmse)

        logger.info(f"LSTM training complete: train_RMSE={train_rmse:.2f}")
        return metrics

    def predict(self, input_df: pd.DataFrame) -> np.ndarray:
        """
        Generate 24-hour forecast from last 48 hours of data.

        Args:
            input_df: DataFrame with at least sequence_length rows of features.

        Returns:
            Array of shape (24,) with PM2.5 predictions.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        available_features = [f for f in self.feature_columns if f in input_df.columns]
        data = input_df[available_features].tail(self.sequence_length).copy()
        data = data.fillna(method="ffill").fillna(0)

        if len(data) < self.sequence_length:
            # Pad with zeros if not enough data
            pad_rows = self.sequence_length - len(data)
            padding = pd.DataFrame(np.zeros((pad_rows, len(available_features))), columns=available_features)
            data = pd.concat([padding, data], ignore_index=True)

        scaled_input = self.scaler.transform(data)
        X = scaled_input.reshape(1, self.sequence_length, -1)

        prediction = self.model.predict(X, verbose=0)
        return prediction[0]  # Shape: (24,)

    def evaluate(self, test_df: pd.DataFrame) -> dict[str, Any]:
        """Evaluate model on test data."""
        X_test, y_test = self.prepare_sequences(test_df)

        if X_test.shape[0] == 0:
            return {"test_rmse": float("nan")}

        predictions = self.model.predict(X_test, batch_size=self.batch_size)

        rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
        mae = np.mean(np.abs(y_test - predictions))
        mape = np.mean(np.abs((y_test - predictions) / np.maximum(y_test, 1e-8))) * 100

        metrics = {
            "test_rmse": float(rmse),
            "test_mae": float(mae),
            "test_mape": float(mape),
        }

        # Per-horizon RMSE
        for h in range(self.forecast_horizon):
            h_rmse = np.sqrt(np.mean((y_test[:, h] - predictions[:, h]) ** 2))
            metrics[f"test_rmse_h{h + 1}"] = float(h_rmse)

        logger.info(f"LSTM test: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.1f}%")
        return metrics

    def get_hyperparameters(self) -> dict[str, Any]:
        """Return all hyperparameters for MLflow logging."""
        return {
            "model_type": "lstm",
            "sequence_length": self.sequence_length,
            "forecast_horizon": self.forecast_horizon,
            "lstm_units_1": self.lstm_units_1,
            "lstm_units_2": self.lstm_units_2,
            "dense_units": self.dense_units,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "loss_function": "huber",
            "optimizer": "adam",
            "n_features": len(self.feature_columns),
        }
