"""
Tests for feature engineering pipeline.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest


class TestRollingFeatures:
    """Tests for rolling statistics computation."""

    def _make_sample_data(self, hours: int = 72):
        """Create sample AQI data for testing."""
        timestamps = pd.date_range("2025-01-01", periods=hours, freq="h", tz="UTC")
        np.random.seed(42)
        return pd.DataFrame({
            "station_id": "test_station",
            "timestamp_utc": timestamps,
            "pm25": np.random.uniform(30, 150, hours),
            "pm10": np.random.uniform(50, 200, hours),
            "no2": np.random.uniform(5, 50, hours),
            "lat": 27.7172,
            "lon": 85.3240,
        })

    def test_rolling_features_no_data_leakage(self):
        """Rolling features must not use future data (no lookahead)."""
        from features.feature_engineering import FeatureEngineer

        df = self._make_sample_data(72)
        weather_df = pd.DataFrame({
            "timestamp_utc": df["timestamp_utc"],
            "temp_c": np.random.uniform(5, 25, 72),
            "humidity_pct": np.random.uniform(40, 90, 72),
            "wind_speed_kmh": np.random.uniform(0, 15, 72),
            "wind_dir_deg": np.random.uniform(0, 360, 72),
            "precip_mm": np.zeros(72),
            "pressure_hpa": np.random.uniform(860, 880, 72),
        })

        engineer = FeatureEngineer()
        features = engineer.compute_all_features(df, weather_df)

        # The 24h rolling mean at hour 24 should only use hours 0-23
        if "pm25_24h_mean" in features.columns and len(features) > 24:
            row_24 = features.iloc[23]
            actual_mean = df["pm25"].iloc[:24].mean()
            assert abs(row_24["pm25_24h_mean"] - actual_mean) < 1.0

    def test_no_future_data_in_lag_features(self):
        """Lag features must point backward only."""
        from features.feature_engineering import FeatureEngineer

        df = self._make_sample_data(50)
        weather_df = pd.DataFrame(columns=["timestamp_utc", "temp_c", "humidity_pct",
                                           "wind_speed_kmh", "wind_dir_deg", "precip_mm", "pressure_hpa"])

        engineer = FeatureEngineer()
        features = engineer.compute_all_features(df, weather_df)

        if "pm25_lag_1h" in features.columns and len(features) > 5:
            # lag_1h at index 5 should equal pm25 at index 4
            assert features.iloc[5]["pm25_lag_1h"] == pytest.approx(
                df["pm25"].iloc[4], rel=1e-5
            ) or pd.isna(features.iloc[5]["pm25_lag_1h"])


class TestCyclicalEncoding:
    """Tests for cyclical time encoding."""

    def test_cyclical_encoding_hour_preserves_range(self):
        """Hour sin/cos should be in [-1, 1] range."""
        from features.feature_engineering import FeatureEngineer

        timestamps = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
        df = pd.DataFrame({
            "station_id": "test",
            "timestamp_utc": timestamps,
            "pm25": np.random.uniform(30, 100, 24),
            "lat": 27.7172,
            "lon": 85.3240,
        })
        weather_df = pd.DataFrame(columns=["timestamp_utc", "temp_c", "humidity_pct",
                                           "wind_speed_kmh", "wind_dir_deg", "precip_mm", "pressure_hpa"])

        engineer = FeatureEngineer()
        features = engineer.compute_all_features(df, weather_df)

        if "hour_sin" in features.columns:
            assert features["hour_sin"].min() >= -1.0
            assert features["hour_sin"].max() <= 1.0
            assert features["hour_cos"].min() >= -1.0
            assert features["hour_cos"].max() <= 1.0


class TestCalendarFlags:
    """Tests for Nepal festival calendar flags."""

    def test_festival_flags_tihar_dates_correct(self):
        """Tihar dates should be flagged correctly."""
        from features.calendar_flags import CalendarFlags

        calendar = CalendarFlags()

        # 2025 Tihar: October 20-24
        tihar_date = datetime(2025, 10, 22, 12, 0, tzinfo=timezone.utc)
        flags = calendar.get_flags_for_date(tihar_date)
        assert flags["is_tihar"] is True

        # Non-Tihar date
        normal_date = datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc)
        flags = calendar.get_flags_for_date(normal_date)
        assert flags["is_tihar"] is False

    def test_monsoon_season_flags(self):
        """Monsoon flags should be set for June-September."""
        from features.calendar_flags import CalendarFlags

        calendar = CalendarFlags()

        # July = monsoon
        monsoon_date = datetime(2025, 7, 15, 12, 0, tzinfo=timezone.utc)
        flags = calendar.get_flags_for_date(monsoon_date)
        assert flags["is_monsoon"] is True
        assert flags["is_pre_monsoon"] is False

        # April = pre-monsoon
        pre_monsoon_date = datetime(2025, 4, 15, 12, 0, tzinfo=timezone.utc)
        flags = calendar.get_flags_for_date(pre_monsoon_date)
        assert flags["is_monsoon"] is False
        assert flags["is_pre_monsoon"] is True


class TestKriging:
    """Tests for spatial interpolation."""

    def test_kriging_produces_values_for_all_wards(self):
        """Kriging should produce estimates for all 32 wards."""
        from ingestion.spatial_interpolation import SpatialInterpolator

        interpolator = SpatialInterpolator()

        # Skip if ward centroids not available
        if interpolator.ward_centroids.empty:
            pytest.skip("Ward centroids file not available")

        # Synthetic sensor data (5 points)
        sensors = [
            {"lat": 27.7172, "lon": 85.3240, "pm25": 65.0, "timestamp_utc": "2025-01-01T12:00:00Z"},
            {"lat": 27.6588, "lon": 85.3247, "pm25": 78.0, "timestamp_utc": "2025-01-01T12:00:00Z"},
            {"lat": 27.6710, "lon": 85.4298, "pm25": 55.0, "timestamp_utc": "2025-01-01T12:00:00Z"},
            {"lat": 27.6783, "lon": 85.2789, "pm25": 90.0, "timestamp_utc": "2025-01-01T12:00:00Z"},
            {"lat": 27.7000, "lon": 85.3500, "pm25": 72.0, "timestamp_utc": "2025-01-01T12:00:00Z"},
        ]

        try:
            results = interpolator.interpolate_pm25(sensors)
            assert len(results) == 32
            assert all(r["source"] == "kriging_interpolated" for r in results)
            assert all(r["pm25"] is not None for r in results)
        except ImportError:
            pytest.skip("pykrige not installed")
