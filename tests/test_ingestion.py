"""
Tests for data ingestion clients.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone


class TestOpenAQClient:
    """Tests for OpenAQ API v3 client."""

    @patch("ingestion.openaq_client.requests.Session.get")
    def test_openaq_client_returns_nepal_stations(self, mock_get):
        """Test that client returns Nepal station locations."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "id": 1,
                    "name": "US Embassy Kathmandu",
                    "coordinates": {"latitude": 27.7172, "longitude": 85.3240},
                    "sensors": [{"id": 101, "parameter": {"name": "pm25"}}],
                },
                {
                    "id": 2,
                    "name": "Ratnapark",
                    "coordinates": {"latitude": 27.7050, "longitude": 85.3150},
                    "sensors": [{"id": 102, "parameter": {"name": "pm10"}}],
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        from ingestion.openaq_client import OpenAQClient
        client = OpenAQClient(api_key="test_key")
        locations = client.get_kathmandu_locations()

        assert len(locations) == 2
        assert locations[0]["name"] == "US Embassy Kathmandu"
        assert locations[0]["coordinates"]["latitude"] == 27.7172

    @patch("ingestion.openaq_client.requests.Session.get")
    def test_openaq_rate_limit_retry(self, mock_get):
        """Test exponential backoff on rate limit."""
        # First call: 429, second call: success
        rate_limited = MagicMock()
        rate_limited.status_code = 429
        rate_limited.headers = {"Retry-After": "1"}

        success = MagicMock()
        success.status_code = 200
        success.json.return_value = {"results": []}
        success.raise_for_status = MagicMock()

        mock_get.side_effect = [rate_limited, success]

        from ingestion.openaq_client import OpenAQClient
        client = OpenAQClient(api_key="test_key")
        result = client.get_nepal_locations()

        assert result == []
        assert mock_get.call_count == 2


class TestWeatherClient:
    """Tests for Open-Meteo weather client."""

    @patch("ingestion.weather_client.requests.Session.get")
    def test_weather_client_returns_expected_columns(self, mock_get):
        """Test that weather client returns all expected columns."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "hourly": {
                "time": ["2025-01-01T00:00", "2025-01-01T01:00"],
                "temperature_2m": [5.2, 4.8],
                "relative_humidity_2m": [80, 82],
                "wind_speed_10m": [3.5, 4.0],
                "wind_direction_10m": [180, 190],
                "precipitation": [0.0, 0.1],
                "surface_pressure": [870, 871],
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        from ingestion.weather_client import WeatherClient
        client = WeatherClient()
        records = client.get_current_forecast()

        assert len(records) == 2
        expected_keys = {"lat", "lon", "timestamp_utc", "temp_c", "humidity_pct",
                         "wind_speed_kmh", "wind_dir_deg", "precip_mm", "pressure_hpa"}
        assert set(records[0].keys()) == expected_keys
        assert records[0]["temp_c"] == 5.2
        assert records[0]["humidity_pct"] == 80


class TestKafkaProducer:
    """Tests for Kafka producer."""

    @patch.dict("os.environ", {"KAFKA_ENABLED": "false"})
    def test_kafka_producer_sends_correct_schema(self):
        """Test Kafka bypass mode logs correct schema."""
        from ingestion.kafka_producer import KafkaAQIProducer

        producer = KafkaAQIProducer()
        records = [{
            "station_id": "test_1",
            "source": "openaq",
            "lat": 27.7172,
            "lon": 85.3240,
            "timestamp_utc": "2025-01-01T00:00:00Z",
            "pm25": 45.0,
            "pm10": 80.0,
            "no2": None,
            "o3": None,
            "co": None,
            "aqi_us": 125,
        }]

        sent = producer.send_aqi_readings(records)
        assert sent == 1


class TestAQICNClient:
    """Tests for AQICN/WAQI client."""

    @patch("ingestion.aqicn_client.requests.Session.get")
    def test_aqicn_client_parses_aqi_correctly(self, mock_get):
        """Test that AQICN response is parsed into correct schema."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "ok",
            "data": {
                "aqi": 156,
                "iaqi": {
                    "pm25": {"v": 78.5},
                    "pm10": {"v": 120.0},
                    "no2": {"v": 15.0},
                    "o3": {"v": 8.0},
                    "co": {"v": 3.2},
                },
                "time": {"iso": "2025-01-01T12:00:00+05:45"},
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        from ingestion.aqicn_client import AQICNClient
        client = AQICNClient(token="test_token")
        readings = client.fetch_all_cities()

        # Should get at least one city
        assert len(readings) >= 1
        reading = readings[0]
        assert reading["source"] == "aqicn"
        assert reading["pm25"] == 78.5
        assert reading["aqi_us"] == 156
        assert reading["lat"] is not None
        assert reading["lon"] is not None
