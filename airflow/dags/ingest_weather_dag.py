"""
Airflow DAG: Weather Data Ingestion — hourly fetch from Open-Meteo.
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
    "ingest_weather_dag",
    default_args=default_args,
    description="Hourly weather data ingestion from Open-Meteo",
    schedule_interval="@hourly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["ingestion", "weather"],
)


def fetch_openmeteo(**kwargs):
    """Fetch weather data from Open-Meteo API."""
    from ingestion.kafka_producer import KafkaAQIProducer
    from ingestion.weather_client import WeatherClient

    client = WeatherClient()
    readings = client.get_current_forecast(forecast_days=1)

    producer = KafkaAQIProducer()
    sent = producer.send_weather_readings(readings)
    producer.close()

    kwargs["ti"].xcom_push(key="weather_count", value=sent)
    return sent


def persist_weather(**kwargs):
    """Persist weather data to DuckDB data lake as Parquet."""
    from ingestion.weather_client import WeatherClient
    from storage.lake import DataLake

    lake = DataLake()
    client = WeatherClient()
    readings = client.get_current_forecast(forecast_days=1)
    count = lake.insert_weather_readings(readings)

    kwargs["ti"].xcom_push(key="persisted_weather_count", value=count)
    return count


# Task definitions
t1 = PythonOperator(
    task_id="fetch_openmeteo",
    python_callable=fetch_openmeteo,
    dag=dag,
)

t2 = PythonOperator(
    task_id="persist_weather",
    python_callable=persist_weather,
    dag=dag,
)

t1 >> t2
