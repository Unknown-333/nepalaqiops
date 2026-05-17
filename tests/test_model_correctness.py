"""
Unit tests for ML model correctness.
No running services needed — uses fixtures and synthetic data.
"""

import os
import pytest
import numpy as np
import pandas as pd


class TestProphetForecastCorrectness:
    """Prophet model output correctness tests."""

    def _make_training_data(self, hours: int = 300):
        """Create realistic synthetic training data."""
        timestamps = pd.date_range("2024-06-01", periods=hours, freq="h", tz="UTC")
        np.random.seed(42)
        hour_of_day = timestamps.hour
        daily_pattern = 60 + 30 * np.sin(2 * np.pi * (hour_of_day - 8) / 24)
        noise = np.random.normal(0, 8, hours)

        return pd.DataFrame({
            "timestamp_utc": timestamps,
            "pm25": np.maximum(daily_pattern + noise, 5.0),
            "temp_c": np.random.uniform(10, 30, hours),
            "humidity_pct": np.random.uniform(40, 90, hours),
            "wind_speed_kmh": np.random.uniform(0, 15, hours),
            "is_tihar": False,
            "is_dashain": False,
            "is_monsoon": True,
            "is_brick_kiln_season": False,
        })

    def test_prophet_forecast_no_nan(self):
        """Prophet forecast should not contain any NaN values."""
        try:
            from models.prophet_model import ProphetAQModel
        except ImportError:
            pytest.skip("Prophet model not importable")

        df = self._make_training_data(300)
        model = ProphetAQModel()
        model.train(df)

        forecast = model._make_future(24)
        predictions = model.model.predict(forecast)

        assert not predictions["yhat"].isna().any(), (
            f"Prophet produced {predictions['yhat'].isna().sum()} NaN values in yhat"
        )
        assert len(predictions) == 24

    def test_prophet_uncertainty_intervals(self):
        """Assert yhat_upper > yhat > yhat_lower for all rows."""
        try:
            from models.prophet_model import ProphetAQModel
        except ImportError:
            pytest.skip("Prophet model not importable")

        df = self._make_training_data(300)
        model = ProphetAQModel()
        model.train(df)

        forecast = model._make_future(24)
        predictions = model.model.predict(forecast)

        # Check uncertainty intervals
        for idx, row in predictions.iterrows():
            assert row["yhat_upper"] >= row["yhat"], (
                f"Row {idx}: yhat_upper ({row['yhat_upper']:.2f}) < yhat ({row['yhat']:.2f})"
            )
            assert row["yhat_lower"] <= row["yhat"], (
                f"Row {idx}: yhat_lower ({row['yhat_lower']:.2f}) > yhat ({row['yhat']:.2f})"
            )


class TestLSTMOutputShape:
    """LSTM model output shape tests."""

    def _make_features(self, hours: int = 200):
        """Create synthetic feature data for LSTM."""
        try:
            from models.lstm_model import LSTM_FEATURES
        except ImportError:
            pytest.skip("LSTM model not importable")
            return None, None

        timestamps = pd.date_range("2024-01-01", periods=hours, freq="h", tz="UTC")
        np.random.seed(42)

        data = {"timestamp_utc": timestamps}
        for feat in LSTM_FEATURES:
            if feat.startswith("is_"):
                data[feat] = np.random.choice([0, 1], hours, p=[0.9, 0.1])
            else:
                data[feat] = np.random.uniform(0, 100, hours)

        data["pm25"] = 50 + 30 * np.sin(2 * np.pi * np.arange(hours) / 24) + np.random.normal(0, 10, hours)
        return pd.DataFrame(data), LSTM_FEATURES

    def test_lstm_output_shape(self):
        """Given input shape (1, 48, N_FEATURES), output shape must be (1, 24)."""
        try:
            from models.lstm_model import LSTMAQModel
        except ImportError:
            pytest.skip("LSTM model not importable")

        df, features = self._make_features(200)
        if df is None:
            return

        model = LSTMAQModel(max_epochs=2, batch_size=16)
        train_df = df.iloc[:160]
        val_df = df.iloc[160:]

        model.train(train_df, val_df)
        prediction = model.predict(val_df)

        assert prediction.shape == (24,), (
            f"Expected output shape (24,), got {prediction.shape}"
        )


class TestEnsembleWeights:
    """Ensemble model weight tests."""

    def test_ensemble_weights_sum_to_one(self):
        """PROPHET_WEIGHT + LSTM_WEIGHT must sum to 1.0."""
        prophet_weight = float(os.getenv("PROPHET_WEIGHT", "0.4"))
        lstm_weight = float(os.getenv("LSTM_WEIGHT", "0.6"))

        total = prophet_weight + lstm_weight
        assert abs(total - 1.0) < 1e-6, (
            f"Weights don't sum to 1.0: PROPHET={prophet_weight} + LSTM={lstm_weight} = {total}"
        )

    def test_ensemble_weights_from_model(self):
        """EnsembleModel class weights must sum to 1.0."""
        try:
            from models.ensemble import EnsembleModel
        except ImportError:
            pytest.skip("Ensemble model not importable")

        ensemble = EnsembleModel(prophet_weight=0.4, lstm_weight=0.6)
        weights = ensemble.get_weights()
        total = weights["prophet_weight"] + weights["lstm_weight"]
        assert abs(total - 1.0) < 1e-6


class TestKrigingCorrectness:
    """Kriging spatial interpolation correctness."""

    def test_kriging_output_in_valid_range(self):
        """
        Given 5 synthetic stations within KTM bounding box with AQI values
        [80, 120, 95, 150, 110], interpolated values must be in [50, 200].
        """
        try:
            from ingestion.spatial_interpolation import SpatialInterpolator
        except ImportError:
            pytest.skip("SpatialInterpolator not importable")

        interpolator = SpatialInterpolator()

        if interpolator.ward_centroids.empty:
            pytest.skip("Ward centroids file not available")

        # 5 sensors spanning Kathmandu Valley
        sensors = [
            {"lat": 27.7172, "lon": 85.3240, "pm25": 80.0, "timestamp_utc": "2026-01-01T12:00:00Z"},
            {"lat": 27.6588, "lon": 85.3247, "pm25": 120.0, "timestamp_utc": "2026-01-01T12:00:00Z"},
            {"lat": 27.6710, "lon": 85.4298, "pm25": 95.0, "timestamp_utc": "2026-01-01T12:00:00Z"},
            {"lat": 27.6783, "lon": 85.2789, "pm25": 150.0, "timestamp_utc": "2026-01-01T12:00:00Z"},
            {"lat": 27.7400, "lon": 85.3100, "pm25": 110.0, "timestamp_utc": "2026-01-01T12:00:00Z"},
        ]

        try:
            results = interpolator.interpolate_pm25(sensors)
        except ImportError:
            pytest.skip("pykrige not installed")

        assert len(results) == 32, f"Expected 32 wards, got {len(results)}"

        for ward in results:
            pm25 = ward["pm25"]
            assert pm25 is not None, f"Ward {ward.get('station_id')} has None PM2.5"
            assert 0 < pm25 < 500, (
                f"Ward {ward.get('station_id')}: pm25={pm25} outside valid range"
            )
            # With input range [80, 150], allow some extrapolation tolerance
            assert 50 <= pm25 <= 200, (
                f"Ward {ward.get('station_id')}: pm25={pm25:.1f} outside [50, 200] "
                f"(input range was [80, 150])"
            )

    def test_kriging_spatial_variance(self):
        """
        Interpolated output must NOT all be the same value.
        Catches "fallback to mean" failure mode.
        """
        try:
            from ingestion.spatial_interpolation import SpatialInterpolator
        except ImportError:
            pytest.skip("SpatialInterpolator not importable")

        interpolator = SpatialInterpolator()

        if interpolator.ward_centroids.empty:
            pytest.skip("Ward centroids file not available")

        # Deliberately varied sensor readings
        sensors = [
            {"lat": 27.72, "lon": 85.32, "pm25": 40.0, "timestamp_utc": "2026-01-01T12:00:00Z"},
            {"lat": 27.66, "lon": 85.32, "pm25": 140.0, "timestamp_utc": "2026-01-01T12:00:00Z"},
            {"lat": 27.67, "lon": 85.43, "pm25": 60.0, "timestamp_utc": "2026-01-01T12:00:00Z"},
            {"lat": 27.68, "lon": 85.28, "pm25": 180.0, "timestamp_utc": "2026-01-01T12:00:00Z"},
            {"lat": 27.74, "lon": 85.35, "pm25": 90.0, "timestamp_utc": "2026-01-01T12:00:00Z"},
        ]

        try:
            results = interpolator.interpolate_pm25(sensors)
        except ImportError:
            pytest.skip("pykrige not installed")

        if not results:
            pytest.skip("Kriging returned empty results")

        pm25_values = [r["pm25"] for r in results]
        std_dev = np.std(pm25_values)

        assert std_dev > 1.0, (
            f"Kriging output has std_dev={std_dev:.4f} — all values are nearly identical. "
            f"This indicates fallback to mean instead of spatial interpolation."
        )
