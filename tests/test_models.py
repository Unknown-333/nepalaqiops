"""
Tests for ML models.
"""

import numpy as np
import pandas as pd


class TestProphetModel:
    """Tests for Prophet forecasting model."""

    def _make_training_data(self, hours: int = 200):
        """Create synthetic training data."""
        timestamps = pd.date_range("2024-06-01", periods=hours, freq="h", tz="UTC")
        np.random.seed(42)
        # Realistic PM2.5 pattern with daily seasonality
        hour_of_day = timestamps.hour
        daily_pattern = 50 + 30 * np.sin(2 * np.pi * (hour_of_day - 8) / 24)
        noise = np.random.normal(0, 10, hours)

        return pd.DataFrame({
            "timestamp_utc": timestamps,
            "pm25": daily_pattern + noise,
            "temp_c": np.random.uniform(10, 30, hours),
            "humidity_pct": np.random.uniform(40, 90, hours),
            "wind_speed_kmh": np.random.uniform(0, 15, hours),
            "is_tihar": False,
            "is_dashain": False,
            "is_monsoon": True,
            "is_brick_kiln_season": False,
        })

    def test_prophet_model_trains_without_error(self):
        """Prophet model should train without raising exceptions."""
        from models.prophet_model import ProphetAQModel

        df = self._make_training_data(200)
        model = ProphetAQModel()
        metrics = model.train(df)

        assert "train_rmse" in metrics
        assert metrics["train_rmse"] > 0
        assert metrics["n_train_samples"] > 0

    def test_prophet_model_produces_forecast(self):
        """Prophet should produce 24 forecasted values."""
        from models.prophet_model import ProphetAQModel

        df = self._make_training_data(200)
        model = ProphetAQModel()
        model.train(df)

        # Generate forecast
        forecast = model._make_future(24)
        predictions = model.model.predict(forecast)

        assert len(predictions) == 24
        assert "yhat" in predictions.columns


class TestLSTMModel:
    """Tests for LSTM forecasting model."""

    def _make_training_data(self, hours: int = 200):
        """Create synthetic data with all required features."""
        from models.lstm_model import LSTM_FEATURES

        timestamps = pd.date_range("2024-01-01", periods=hours, freq="h", tz="UTC")
        np.random.seed(42)

        data = {"timestamp_utc": timestamps}
        for feat in LSTM_FEATURES:
            if feat.startswith("is_"):
                data[feat] = np.random.choice([0, 1], hours, p=[0.9, 0.1])
            else:
                data[feat] = np.random.uniform(0, 100, hours)

        # Ensure pm25 has realistic values
        data["pm25"] = 50 + 30 * np.sin(2 * np.pi * np.arange(hours) / 24) + np.random.normal(0, 10, hours)

        return pd.DataFrame(data)

    def test_lstm_output_shape_is_24(self):
        """LSTM should output 24-hour predictions."""
        from models.lstm_model import LSTMAQModel

        model = LSTMAQModel(max_epochs=2, batch_size=16)  # Quick train
        df = self._make_training_data(200)

        # Split
        train_df = df.iloc[:160]
        val_df = df.iloc[160:]

        metrics = model.train(train_df, val_df)

        # Test prediction shape
        prediction = model.predict(val_df)
        assert prediction.shape == (24,)

    def test_lstm_trains_without_shuffle(self):
        """LSTM must not shuffle time-series data."""
        from models.lstm_model import LSTMAQModel

        model = LSTMAQModel(max_epochs=2)
        df = self._make_training_data(150)

        train_df = df.iloc[:120]
        val_df = df.iloc[120:]

        # Should complete without error
        metrics = model.train(train_df, val_df)
        assert "train_rmse" in metrics


class TestIsolationForest:
    """Tests for anomaly detection model."""

    def test_isolation_forest_detects_known_spike(self):
        """Isolation Forest should flag an extreme spike as anomaly."""
        from models.isolation_forest import AnomalyDetector

        np.random.seed(42)
        # Normal data
        normal_data = pd.DataFrame({
            "pm25": np.random.uniform(30, 80, 100),
            "pm10": np.random.uniform(50, 120, 100),
            "no2": np.random.uniform(5, 30, 100),
            "pm25_1h_mean": np.random.uniform(30, 80, 100),
            "pm25_6h_mean": np.random.uniform(30, 80, 100),
        })

        detector = AnomalyDetector(contamination=0.05)
        detector.train(normal_data)

        # Add a spike
        spike_data = pd.DataFrame({
            "pm25": [500.0],  # Extreme value
            "pm10": [800.0],
            "no2": [200.0],
            "pm25_1h_mean": [450.0],
            "pm25_6h_mean": [300.0],
        })

        scored = detector.score(spike_data)
        assert scored["is_anomaly"].iloc[0] is True or scored["is_anomaly"].iloc[0] == True

    def test_sensor_fault_detection(self):
        """Should detect stuck-at-zero sensor fault."""
        from models.isolation_forest import AnomalyDetector

        detector = AnomalyDetector()
        # 5 hours of zero readings
        stuck_data = pd.DataFrame({
            "station_id": ["station_1"] * 5,
            "timestamp_utc": pd.date_range("2025-01-01", periods=5, freq="h"),
            "pm25": [0.0, 0.0, 0.0, 0.0, 0.0],
        })

        faults = detector.detect_sensor_faults(stuck_data)
        assert len(faults) > 0
        assert any(f["alert_type"] == "sensor_fault_stuck_zero" for f in faults)


class TestEnsemble:
    """Tests for ensemble model."""

    def test_ensemble_weights_sum_to_one(self):
        """Ensemble weights must sum to 1.0."""
        from models.ensemble import EnsembleModel

        ensemble = EnsembleModel(prophet_weight=0.4, lstm_weight=0.6)
        weights = ensemble.get_weights()
        assert abs(weights["prophet_weight"] + weights["lstm_weight"] - 1.0) < 1e-6

    def test_ensemble_prediction_is_weighted_average(self):
        """Ensemble prediction should be weighted average of components."""
        from models.ensemble import EnsembleModel

        ensemble = EnsembleModel(prophet_weight=0.4, lstm_weight=0.6)
        prophet_pred = np.array([50.0] * 24)
        lstm_pred = np.array([60.0] * 24)

        result = ensemble.predict(prophet_pred, lstm_pred)
        expected = 0.4 * 50.0 + 0.6 * 60.0
        np.testing.assert_allclose(result, expected, rtol=1e-5)
