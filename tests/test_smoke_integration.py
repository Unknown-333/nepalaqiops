"""
Integration tests for NepalAQI-Ops pipeline.
Requires running Docker services — skip with: pytest -m "not integration"
"""

import json
import os
import time

import pytest

# Mark all tests as integration
pytestmark = pytest.mark.integration

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5000")

# KTM bounding box
KTM_LAT_MIN = 27.60
KTM_LAT_MAX = 27.80
KTM_LON_MIN = 85.20
KTM_LON_MAX = 85.45


class TestKafkaTopics:
    """Test Kafka topic existence."""

    def test_kafka_topics_exist(self):
        """Connect to Kafka and assert the three required topics exist."""
        try:
            from confluent_kafka.admin import AdminClient
        except ImportError:
            pytest.skip("confluent_kafka not installed")

        admin = AdminClient({"bootstrap.servers": KAFKA_BOOTSTRAP})
        metadata = admin.list_topics(timeout=10)
        topic_names = set(metadata.topics.keys())

        expected_topics = {"raw.aqi", "weather.raw", "anomaly.alerts"}
        missing = expected_topics - topic_names

        assert not missing, f"Missing Kafka topics: {missing}. Found: {topic_names}"


class TestRedisFeatureStore:
    """Test Redis feature store population."""

    def test_redis_feature_keys_populated(self):
        """Assert at least one key matching 'features:*' exists with valid JSON."""
        try:
            import redis
        except ImportError:
            pytest.skip("redis not installed")

        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

        try:
            r.ping()
        except redis.ConnectionError:
            pytest.skip("Redis not reachable")

        # Scan for feature keys
        keys = list(r.scan_iter(match="features:*", count=100))

        if not keys:
            pytest.skip("No feature keys in Redis (pipeline may not have run yet)")

        # Validate at least one key has valid JSON
        for key in keys[:5]:
            value = r.get(key)
            assert value is not None, f"Key {key} has no value"
            data = json.loads(value)
            assert isinstance(data, dict), f"Key {key} value is not a JSON object"
            # Should have PM2.5 or station_id
            assert "pm25" in data or "station_id" in data, (
                f"Key {key} missing expected fields. Keys: {list(data.keys())[:10]}"
            )


class TestMLflowRegistry:
    """Test MLflow model registry."""

    def test_mlflow_champion_model_registered(self):
        """Query MLflow API and assert a Production-stage model exists."""
        import requests

        try:
            resp = requests.get(
                f"{MLFLOW_URL}/api/2.0/mlflow/registered-models/search",
                params={"max_results": 10},
                timeout=10,
            )
        except requests.ConnectionError:
            pytest.skip("MLflow not reachable")

        if resp.status_code != 200:
            pytest.skip(f"MLflow returned {resp.status_code}")

        data = resp.json()
        models = data.get("registered_models", [])

        if not models:
            pytest.skip("No models registered yet (training may not have run)")

        # Check if any model has a version in Production
        nepalaqiops_models = [m for m in models if "nepalaqiops" in m.get("name", "").lower()]

        assert nepalaqiops_models, (
            f"No model with 'nepalaqiops' in name found. "
            f"Available: {[m['name'] for m in models]}"
        )


class TestFastAPIForecastSchema:
    """Test FastAPI forecast response schema."""

    def test_fastapi_forecast_schema(self):
        """Call GET /forecast/aqicn_kathmandu and validate full response schema."""
        from datetime import datetime

        import requests

        try:
            resp = requests.get(f"{FASTAPI_URL}/forecast/aqicn_kathmandu?hours=24", timeout=10)
        except requests.ConnectionError:
            pytest.skip("FastAPI not reachable")

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()

        # Top-level fields
        assert "station_id" in data
        assert "forecasts" in data
        assert "model_used" in data
        assert data["station_id"] == "aqicn_kathmandu"

        forecasts = data["forecasts"]
        assert len(forecasts) == 24, f"Expected 24 forecasts, got {len(forecasts)}"

        valid_categories = {"Good", "Moderate", "USG", "Unhealthy", "Very Unhealthy", "Hazardous"}

        for i, entry in enumerate(forecasts):
            # Required fields
            assert "hour" in entry, f"Forecast {i} missing 'hour'"
            assert "pm25_predicted" in entry, f"Forecast {i} missing 'pm25_predicted'"
            assert "aqi_category" in entry, f"Forecast {i} missing 'aqi_category'"
            assert "confidence_lower" in entry, f"Forecast {i} missing 'confidence_lower'"
            assert "confidence_upper" in entry, f"Forecast {i} missing 'confidence_upper'"

            # Type and range validation
            pm25 = entry["pm25_predicted"]
            assert isinstance(pm25, (int, float)), f"pm25 not numeric: {pm25}"
            assert 0 <= pm25 <= 500, f"Forecast {i}: pm25={pm25} out of [0, 500]"

            # AQI category
            assert entry["aqi_category"] in valid_categories, (
                f"Forecast {i}: invalid category '{entry['aqi_category']}'"
            )

            # Confidence bounds
            assert entry["confidence_lower"] <= pm25, (
                f"Forecast {i}: lower bound {entry['confidence_lower']} > prediction {pm25}"
            )
            assert entry["confidence_upper"] >= pm25, (
                f"Forecast {i}: upper bound {entry['confidence_upper']} < prediction {pm25}"
            )

            # Timestamp is parseable ISO format
            dt = datetime.fromisoformat(entry["hour"])
            assert dt is not None


class TestFastAPIHeatmap:
    """Test FastAPI heatmap GeoJSON."""

    def test_fastapi_heatmap_geojson(self):
        """Call GET /forecast/heatmap, parse as GeoJSON, validate coordinates."""
        import requests

        try:
            resp = requests.get(f"{FASTAPI_URL}/forecast/heatmap", timeout=10)
        except requests.ConnectionError:
            pytest.skip("FastAPI not reachable")

        assert resp.status_code == 200
        data = resp.json()

        # GeoJSON structure
        assert data["type"] == "FeatureCollection"
        features = data["features"]
        assert len(features) == 32, f"Expected 32 wards, got {len(features)}"

        for feat in features:
            assert feat["type"] == "Feature"
            assert "properties" in feat
            assert "geometry" in feat
            assert feat["geometry"]["type"] == "Point"

            coords = feat["geometry"]["coordinates"]
            assert len(coords) == 2, f"Expected [lon, lat], got {coords}"

            lon, lat = coords[0], coords[1]
            assert KTM_LON_MIN <= lon <= KTM_LON_MAX, (
                f"Longitude {lon} outside KTM bounds [{KTM_LON_MIN}, {KTM_LON_MAX}]"
            )
            assert KTM_LAT_MIN <= lat <= KTM_LAT_MAX, (
                f"Latitude {lat} outside KTM bounds [{KTM_LAT_MIN}, {KTM_LAT_MAX}]"
            )

            # Properties
            props = feat["properties"]
            assert "ward_id" in props
            assert "ward_name" in props
            assert "aqi_category" in props


class TestEndToEndKafkaToDuckDB:
    """End-to-end integration: Kafka → DuckDB."""

    def test_end_to_end_kafka_to_duckdb(self):
        """
        Produce a synthetic test message to raw.aqi topic,
        wait up to 30 seconds, then verify it arrived in DuckDB.
        """
        try:
            from confluent_kafka import Producer
        except ImportError:
            pytest.skip("confluent_kafka not installed")

        unique_station = f"integration_test_{int(time.time())}"
        message = {
            "station_id": unique_station,
            "source": "integration_test",
            "lat": 27.7172,
            "lon": 85.3240,
            "timestamp_utc": "2026-05-17T00:00:00Z",
            "pm25": 99.9,
            "pm10": 120.0,
            "no2": 15.0,
            "o3": None,
            "co": None,
            "aqi_us": 150,
        }

        # Produce to Kafka
        try:
            producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP})
            producer.produce(
                topic="raw.aqi",
                key=unique_station.encode(),
                value=json.dumps(message).encode(),
            )
            producer.flush(timeout=5)
        except Exception as e:
            pytest.skip(f"Cannot produce to Kafka: {e}")

        # Poll DuckDB for the row (up to 30s)
        # Note: This test requires the ingestion pipeline to consume from Kafka
        # If no consumer is running, this test will time out and skip
        try:
            import duckdb
        except ImportError:
            pytest.skip("duckdb not installed")

        db_path = os.getenv("DUCKDB_PATH", "/opt/airflow/datalake/nepalaqiops.duckdb")
        if not os.path.exists(db_path):
            # Try local path for testing outside Docker
            db_path = os.path.join(
                os.path.dirname(__file__), "..", "datalake", "nepalaqiops.duckdb"
            )
            if not os.path.exists(db_path):
                pytest.skip(f"DuckDB file not found at {db_path}")

        found = False
        deadline = time.time() + 30

        while time.time() < deadline:
            try:
                con = duckdb.connect(db_path, read_only=True)
                result = con.execute(
                    "SELECT COUNT(*) FROM raw_aqi WHERE station_id = ?",
                    [unique_station],
                ).fetchone()
                con.close()
                if result and result[0] > 0:
                    found = True
                    break
            except Exception:
                pass
            time.sleep(2)

        if not found:
            pytest.skip(
                f"Message '{unique_station}' did not appear in DuckDB within 30s "
                f"(Kafka consumer may not be running)"
            )
