"""
Airflow DAG: AQI Data Ingestion — hourly fetch from OpenAQ v3 and AQICN.
"""

import sys

sys.path.insert(0, "/opt/airflow")

from datetime import datetime, timedelta

from airflow.operators.python import PythonOperator

from airflow import DAG

default_args = {
    "owner": "nepalaqiops",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "ingest_aqi_dag",
    default_args=default_args,
    description="Hourly AQI data ingestion from OpenAQ v3 and AQICN",
    schedule_interval="@hourly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["ingestion", "aqi"],
)


def fetch_openaq_stations(**kwargs):
    """Fetch air quality readings from OpenAQ v3 API."""
    from ingestion.kafka_producer import KafkaAQIProducer
    from ingestion.openaq_client import OpenAQClient

    client = OpenAQClient()
    readings = client.fetch_all_kathmandu_readings()

    producer = KafkaAQIProducer()
    sent = producer.send_aqi_readings(readings)
    producer.close()

    kwargs["ti"].xcom_push(key="openaq_count", value=sent)
    return sent


def fetch_aqicn_cities(**kwargs):
    """Fetch AQI readings from AQICN/WAQI API."""
    from ingestion.aqicn_client import AQICNClient
    from ingestion.kafka_producer import KafkaAQIProducer

    client = AQICNClient()
    readings = client.fetch_all_cities()

    producer = KafkaAQIProducer()
    sent = producer.send_aqi_readings(readings)
    producer.close()

    kwargs["ti"].xcom_push(key="aqicn_count", value=sent)
    return sent


def persist_to_datalake(**kwargs):
    """Consume readings and persist to DuckDB data lake."""
    from ingestion.aqicn_client import AQICNClient
    from ingestion.openaq_client import OpenAQClient
    from storage.lake import DataLake

    lake = DataLake()

    # Re-fetch and persist (in non-Kafka mode, direct persistence)
    openaq_client = OpenAQClient()
    openaq_readings = openaq_client.fetch_all_kathmandu_readings()
    lake.insert_aqi_readings(openaq_readings)

    aqicn_client = AQICNClient()
    aqicn_readings = aqicn_client.fetch_all_cities()
    lake.insert_aqi_readings(aqicn_readings)

    total = len(openaq_readings) + len(aqicn_readings)
    kwargs["ti"].xcom_push(key="persisted_count", value=total)
    return total


def run_anomaly_detection(**kwargs):
    """Score new readings with Isolation Forest, emit anomaly alerts."""
    import os
    import pickle

    from ingestion.kafka_producer import KafkaAQIProducer
    from models.isolation_forest import AnomalyDetector
    from storage.lake import DataLake

    lake = DataLake()

    # Get latest hour of data
    recent_data = lake.query("""
        SELECT * FROM raw_aqi
        WHERE timestamp_utc >= NOW() - INTERVAL 2 HOUR
        ORDER BY timestamp_utc DESC
    """)

    if recent_data.empty:
        return 0

    # Try to load trained model, otherwise use default
    detector = AnomalyDetector()
    model_path = "/opt/airflow/datalake/models/isolation_forest.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            detector = pickle.load(f)
    else:
        # Train on available data
        detector.train(recent_data)

    # Score and generate alerts
    scored = detector.score(recent_data)
    alerts = detector.generate_alerts(scored)

    # Also check for sensor faults
    faults = detector.detect_sensor_faults(recent_data)
    alerts.extend(faults)

    # Publish alerts to Kafka
    if alerts:
        producer = KafkaAQIProducer()
        producer.produce("anomaly.alerts", alerts)
        producer.close()

    return len(alerts)


# Task definitions
t1 = PythonOperator(
    task_id="fetch_openaq_stations",
    python_callable=fetch_openaq_stations,
    dag=dag,
)

t2 = PythonOperator(
    task_id="fetch_aqicn_cities",
    python_callable=fetch_aqicn_cities,
    dag=dag,
)

t3 = PythonOperator(
    task_id="persist_to_datalake",
    python_callable=persist_to_datalake,
    dag=dag,
)

t4 = PythonOperator(
    task_id="run_anomaly_detection",
    python_callable=run_anomaly_detection,
    dag=dag,
)

# DAG structure
[t1, t2] >> t3 >> t4
