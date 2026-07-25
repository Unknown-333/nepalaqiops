"""
Kafka Producer — pushes AQI and weather readings to Kafka topics.
Includes KAFKA_ENABLED bypass for local development.
"""

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

KAFKA_ENABLED = os.getenv("KAFKA_ENABLED", "true").lower() == "true"
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")

# Topic names
TOPIC_RAW_AQI = "raw.aqi"
TOPIC_WEATHER_RAW = "weather.raw"
TOPIC_ANOMALY_ALERTS = "anomaly.alerts"


class KafkaAQIProducer:
    """Kafka producer for AQI and weather data streams."""

    def __init__(self):
        self._producer = None
        if KAFKA_ENABLED:
            self._init_producer()

    def _init_producer(self):
        """Initialize Kafka producer with retry."""
        try:
            from confluent_kafka import Producer
            self._producer = Producer({
                "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
                "client.id": "nepalaqiops-ingestion",
                "acks": "all",
                "retries": 3,
                "retry.backoff.ms": 1000,
            })
            logger.info(f"Kafka producer connected to {KAFKA_BOOTSTRAP_SERVERS}")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            self._producer = None

    def _delivery_report(self, err, msg):
        """Callback for message delivery confirmation."""
        if err:
            logger.error(f"Message delivery failed: {err}")
        else:
            logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}]")

    def produce(self, topic: str, records: list[dict[str, Any]]) -> int:
        """
        Produce records to a Kafka topic.
        If KAFKA_ENABLED=false, logs records instead.
        Returns count of records sent.
        """
        if not KAFKA_ENABLED or self._producer is None:
            logger.info(
                f"[KAFKA BYPASS] Would send {len(records)} records to topic '{topic}'"
            )
            return len(records)

        sent = 0
        for record in records:
            try:
                key = record.get("station_id", "unknown")
                value = json.dumps(record, default=str).encode("utf-8")
                self._producer.produce(
                    topic=topic,
                    key=key.encode("utf-8"),
                    value=value,
                    callback=self._delivery_report,
                )
                sent += 1
            except Exception as e:
                logger.error(f"Failed to produce record: {e}")

        # Flush to ensure all messages are sent
        self._producer.flush(timeout=10)
        logger.info(f"Produced {sent}/{len(records)} records to '{topic}'")
        return sent

    def send_aqi_readings(self, readings: list[dict[str, Any]]) -> int:
        """Send AQI readings to the raw.aqi topic."""
        return self.produce(TOPIC_RAW_AQI, readings)

    def send_weather_readings(self, readings: list[dict[str, Any]]) -> int:
        """Send weather readings to the weather.raw topic."""
        return self.produce(TOPIC_WEATHER_RAW, readings)

    def send_anomaly_alert(self, alert: dict[str, Any]) -> int:
        """Send a single anomaly alert to the anomaly.alerts topic."""
        return self.produce(TOPIC_ANOMALY_ALERTS, [alert])

    def close(self):
        """Flush remaining messages and close producer."""
        if self._producer:
            self._producer.flush(timeout=30)
            logger.info("Kafka producer closed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    producer = KafkaAQIProducer()

    # Test with sample data
    sample_aqi = [{
        "station_id": "test_station",
        "source": "test",
        "lat": 27.7172,
        "lon": 85.3240,
        "timestamp_utc": "2025-01-01T00:00:00Z",
        "pm25": 45.0,
        "pm10": 80.0,
        "no2": 12.0,
        "o3": None,
        "co": None,
        "aqi_us": 125,
    }]
    producer.send_aqi_readings(sample_aqi)
    producer.close()
