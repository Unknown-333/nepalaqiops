"""
Tests for FastAPI serving layer.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from serving.main import app
    # Mock model store
    mock_store = MagicMock()
    mock_store.champion_info = {"name": "prophet_v1"}
    mock_store.challenger_info = {"name": "lstm_v2"}
    mock_store.last_loaded = None
    mock_store.predict = AsyncMock(return_value={
        "predictions": [50.0 + i for i in range(24)],
        "lower": [40.0 + i for i in range(24)],
        "upper": [60.0 + i for i in range(24)],
        "model_used": "ensemble",
    })
    mock_store.load_models = AsyncMock()
    app.state.model_store = mock_store
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_endpoint_returns_200(self, client):
        """Health check should return 200 with model info."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "champion_model" in data
        assert "challenger_model" in data
        assert "last_retrain" in data


class TestForecastEndpoint:
    """Tests for /forecast endpoints."""

    def test_forecast_endpoint_returns_24_hours(self, client):
        """Forecast should return 24 hourly predictions by default."""
        response = client.get("/forecast/aqicn_kathmandu?hours=24")
        assert response.status_code == 200
        data = response.json()
        assert len(data["forecasts"]) == 24
        assert data["station_id"] == "aqicn_kathmandu"
        assert "model_used" in data

        # Each forecast entry should have required fields
        entry = data["forecasts"][0]
        assert "hour" in entry
        assert "pm25_predicted" in entry
        assert "aqi_category" in entry
        assert "confidence_lower" in entry
        assert "confidence_upper" in entry

    def test_challenger_routing_via_header(self, client):
        """X-Model-Version: challenger should route to challenger model."""
        response = client.get(
            "/forecast/aqicn_kathmandu",
            headers={"X-Model-Version": "challenger"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model_version"] == "challenger"

    @patch("serving.routers.forecast.redis")
    def test_heatmap_returns_valid_geojson(self, mock_redis_module, client):
        """Heatmap should return valid GeoJSON FeatureCollection."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        mock_redis_module.Redis.return_value = mock_redis

        response = client.get("/forecast/heatmap")
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "FeatureCollection"
        assert "features" in data
        assert len(data["features"]) == 32  # 32 wards


class TestMetricsEndpoint:
    """Tests for Prometheus metrics endpoint."""

    def test_metrics_endpoint_exposes_prometheus_metrics(self, client):
        """Metrics endpoint should return Prometheus format."""
        response = client.get("/metrics")
        assert response.status_code == 200
        content = response.text
        # Should contain at least the custom metric names
        assert "nepalaqiops" in content or "python_info" in content
