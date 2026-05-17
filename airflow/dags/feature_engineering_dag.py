"""
Airflow DAG: Feature Engineering — computes rolling, lag, calendar, and spatial features.
Waits for ingestion DAG to complete before writing to DuckDB (avoids single-writer lock).
"""

import sys
sys.path.insert(0, "/opt/airflow")

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor

default_args = {
    "owner": "nepalaqiops",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "feature_engineering_dag",
    default_args=default_args,
    description="Hourly feature computation after ingestion",
    schedule_interval="@hourly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["features", "engineering"],
)


def compute_rolling_features(**kwargs):
    """Compute rolling averages, lag features, and time encodings."""
    from storage.lake import DataLake
    from features.feature_engineering import FeatureEngineer

    lake = DataLake()
    engineer = FeatureEngineer()

    # Get recent AQI data (last 7 days for rolling calculations)
    aqi_data = lake.query("""
        SELECT * FROM raw_aqi
        WHERE timestamp_utc >= NOW() - INTERVAL 7 DAY
        ORDER BY station_id, timestamp_utc
    """)

    weather_data = lake.query("""
        SELECT * FROM raw_weather
        WHERE timestamp_utc >= NOW() - INTERVAL 7 DAY
        ORDER BY timestamp_utc
    """)

    if aqi_data.empty:
        return 0

    features_df = engineer.compute_all_features(aqi_data, weather_data)
    kwargs["ti"].xcom_push(key="n_features_computed", value=len(features_df))
    return len(features_df)


def compute_calendar_flags(**kwargs):
    """Compute Nepal festival and seasonal flags."""
    from features.calendar_flags import CalendarFlags

    calendar = CalendarFlags()
    # Verify calendar data is loaded
    n_entries = len(calendar.festivals_df)
    kwargs["ti"].xcom_push(key="n_calendar_entries", value=n_entries)
    return n_entries


def compute_spatial_kriging(**kwargs):
    """Run Kriging interpolation for wards without sensors."""
    from storage.lake import DataLake
    from ingestion.spatial_interpolation import SpatialInterpolator

    lake = DataLake()
    interpolator = SpatialInterpolator()

    # Get latest readings from real sensors
    latest_readings = lake.query("""
        SELECT station_id, lat, lon, pm25, timestamp_utc
        FROM raw_aqi
        WHERE source != 'kriging_interpolated'
          AND pm25 IS NOT NULL
          AND timestamp_utc >= NOW() - INTERVAL 2 HOUR
        ORDER BY timestamp_utc DESC
    """)

    if latest_readings.empty:
        return 0

    # Convert to list of dicts for interpolation
    sensor_data = latest_readings.to_dict("records")
    interpolated = interpolator.interpolate_pm25(sensor_data)

    # Store interpolated values
    if interpolated:
        lake.insert_aqi_readings(interpolated)

    kwargs["ti"].xcom_push(key="n_wards_interpolated", value=len(interpolated))
    return len(interpolated)


def write_feature_store(**kwargs):
    """Write computed features to Redis for online serving."""
    import os
    import json
    import redis

    from storage.lake import DataLake

    lake = DataLake()
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "redis"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        decode_responses=True,
    )

    # Get latest features
    features = lake.query("""
        SELECT * FROM features
        WHERE computed_at >= NOW() - INTERVAL 2 HOUR
        ORDER BY timestamp_utc DESC
    """)

    if features.empty:
        return 0

    # Write to Redis (keyed by station_id)
    written = 0
    for station_id, group in features.groupby("station_id"):
        latest = group.iloc[0].to_dict()
        # Convert timestamps to strings
        for k, v in latest.items():
            if hasattr(v, "isoformat"):
                latest[k] = v.isoformat()
            elif isinstance(v, float) and (v != v):  # NaN check
                latest[k] = None

        redis_key = f"features:{station_id}:latest"
        redis_client.set(redis_key, json.dumps(latest, default=str), ex=7200)
        written += 1

    kwargs["ti"].xcom_push(key="n_stations_cached", value=written)
    return written


# Task definitions
wait_for_ingestion = ExternalTaskSensor(
    task_id="wait_for_ingestion",
    external_dag_id="ingest_aqi_dag",
    external_task_id="persist_to_datalake",
    execution_delta=timedelta(0),  # Same schedule interval
    timeout=600,
    poke_interval=30,
    mode="reschedule",
    dag=dag,
)

t1 = PythonOperator(
    task_id="compute_rolling_features",
    python_callable=compute_rolling_features,
    dag=dag,
)

t2 = PythonOperator(
    task_id="compute_calendar_flags",
    python_callable=compute_calendar_flags,
    dag=dag,
)

t3 = PythonOperator(
    task_id="compute_spatial_kriging",
    python_callable=compute_spatial_kriging,
    dag=dag,
)

t4 = PythonOperator(
    task_id="write_feature_store",
    python_callable=write_feature_store,
    dag=dag,
)

wait_for_ingestion >> [t1, t2, t3] >> t4
