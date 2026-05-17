"""
Data Lake Abstraction — DuckDB + Parquet storage layer.
Provides ACID-like transactions on Parquet files without Spark overhead.
"""

import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

DATALAKE_PATH = os.getenv("DATALAKE_PATH", "/opt/airflow/datalake")


class DataLake:
    """DuckDB + Parquet data lake for NepalAQI-Ops."""

    def __init__(self, base_path: str = DATALAKE_PATH):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._db_path = str(self.base_path / "nepalaqiops.duckdb")
        self._init_database()

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get a DuckDB connection."""
        return duckdb.connect(self._db_path)

    def _init_database(self):
        """Initialize DuckDB database and create tables."""
        con = self._get_connection()
        try:
            # Raw AQI readings table
            con.execute("""
                CREATE TABLE IF NOT EXISTS raw_aqi (
                    station_id VARCHAR,
                    source VARCHAR,
                    lat DOUBLE,
                    lon DOUBLE,
                    timestamp_utc TIMESTAMP,
                    pm25 DOUBLE,
                    pm10 DOUBLE,
                    no2 DOUBLE,
                    o3 DOUBLE,
                    co DOUBLE,
                    aqi_us INTEGER,
                    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Raw weather readings table
            con.execute("""
                CREATE TABLE IF NOT EXISTS raw_weather (
                    lat DOUBLE,
                    lon DOUBLE,
                    timestamp_utc TIMESTAMP,
                    temp_c DOUBLE,
                    humidity_pct DOUBLE,
                    wind_speed_kmh DOUBLE,
                    wind_dir_deg DOUBLE,
                    precip_mm DOUBLE,
                    pressure_hpa DOUBLE,
                    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Feature store table
            con.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    station_id VARCHAR,
                    timestamp_utc TIMESTAMP,
                    hour_of_day INTEGER,
                    day_of_week INTEGER,
                    month INTEGER,
                    is_weekend BOOLEAN,
                    hour_sin DOUBLE,
                    hour_cos DOUBLE,
                    month_sin DOUBLE,
                    month_cos DOUBLE,
                    pm25_1h_mean DOUBLE,
                    pm25_3h_mean DOUBLE,
                    pm25_6h_mean DOUBLE,
                    pm25_12h_mean DOUBLE,
                    pm25_24h_mean DOUBLE,
                    pm25_1h_std DOUBLE,
                    pm25_6h_std DOUBLE,
                    pm25_24h_std DOUBLE,
                    pm25_lag_1h DOUBLE,
                    pm25_lag_3h DOUBLE,
                    pm25_lag_6h DOUBLE,
                    pm25_lag_12h DOUBLE,
                    pm25_lag_24h DOUBLE,
                    pm25_lag_48h DOUBLE,
                    pm25_lag_168h DOUBLE,
                    temp_c DOUBLE,
                    humidity_pct DOUBLE,
                    wind_speed_kmh DOUBLE,
                    wind_dir_sin DOUBLE,
                    wind_dir_cos DOUBLE,
                    precip_mm DOUBLE,
                    pressure_hpa DOUBLE,
                    precip_6h_cumulative DOUBLE,
                    is_tihar BOOLEAN,
                    is_dashain BOOLEAN,
                    is_indra_jatra BOOLEAN,
                    is_monsoon BOOLEAN,
                    is_pre_monsoon BOOLEAN,
                    is_brick_kiln_season BOOLEAN,
                    is_public_holiday BOOLEAN,
                    aqi_us_category INTEGER,
                    station_distance_to_city_center_km DOUBLE,
                    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            logger.info(f"DuckDB initialized at {self._db_path}")
        finally:
            con.close()

    def insert_aqi_readings(self, readings: list[dict[str, Any]]) -> int:
        """Insert raw AQI readings into the data lake."""
        if not readings:
            return 0

        df = pd.DataFrame(readings)
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df["ingested_at"] = datetime.now(timezone.utc)

        con = self._get_connection()
        try:
            con.execute("INSERT INTO raw_aqi SELECT * FROM df")
            count = len(df)
            logger.info(f"Inserted {count} AQI readings into data lake")
            return count
        finally:
            con.close()

    def insert_weather_readings(self, readings: list[dict[str, Any]]) -> int:
        """Insert raw weather readings into the data lake."""
        if not readings:
            return 0

        df = pd.DataFrame(readings)
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df["ingested_at"] = datetime.now(timezone.utc)

        con = self._get_connection()
        try:
            con.execute("INSERT INTO raw_weather SELECT * FROM df")
            count = len(df)
            logger.info(f"Inserted {count} weather readings into data lake")
            return count
        finally:
            con.close()

    def insert_features(self, features_df: pd.DataFrame) -> int:
        """Insert computed features into the data lake."""
        if features_df.empty:
            return 0

        df = features_df.copy()
        df["computed_at"] = datetime.now(timezone.utc)

        con = self._get_connection()
        try:
            con.execute("INSERT INTO features SELECT * FROM df")
            count = len(df)
            logger.info(f"Inserted {count} feature rows into data lake")
            return count
        finally:
            con.close()

    def query(self, sql: str) -> pd.DataFrame:
        """Execute a SQL query and return results as DataFrame."""
        con = self._get_connection()
        try:
            return con.execute(sql).fetchdf()
        finally:
            con.close()

    def get_aqi_data(
        self,
        station_id: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Retrieve AQI data with optional filters."""
        conditions = []
        if station_id:
            conditions.append(f"station_id = '{station_id}'")
        if start_date:
            conditions.append(f"timestamp_utc >= '{start_date.isoformat()}'")
        if end_date:
            conditions.append(f"timestamp_utc <= '{end_date.isoformat()}'")

        where_clause = " AND ".join(conditions)
        sql = "SELECT * FROM raw_aqi"
        if where_clause:
            sql += f" WHERE {where_clause}"
        sql += " ORDER BY timestamp_utc"

        return self.query(sql)

    def get_weather_data(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Retrieve weather data with optional filters."""
        conditions = []
        if start_date:
            conditions.append(f"timestamp_utc >= '{start_date.isoformat()}'")
        if end_date:
            conditions.append(f"timestamp_utc <= '{end_date.isoformat()}'")

        where_clause = " AND ".join(conditions)
        sql = "SELECT * FROM raw_weather"
        if where_clause:
            sql += f" WHERE {where_clause}"
        sql += " ORDER BY timestamp_utc"

        return self.query(sql)

    def get_features(
        self,
        station_id: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Retrieve feature data for model training."""
        conditions = []
        if station_id:
            conditions.append(f"station_id = '{station_id}'")
        if start_date:
            conditions.append(f"timestamp_utc >= '{start_date.isoformat()}'")
        if end_date:
            conditions.append(f"timestamp_utc <= '{end_date.isoformat()}'")

        where_clause = " AND ".join(conditions)
        sql = "SELECT * FROM features"
        if where_clause:
            sql += f" WHERE {where_clause}"
        sql += " ORDER BY timestamp_utc"

        return self.query(sql)

    def export_to_parquet(self, table: str, output_path: str | None = None) -> str:
        """Export a table to Parquet file for external processing."""
        if output_path is None:
            output_path = str(self.base_path / f"{table}.parquet")

        con = self._get_connection()
        try:
            con.execute(f"COPY {table} TO '{output_path}' (FORMAT PARQUET)")
            logger.info(f"Exported {table} to {output_path}")
            return output_path
        finally:
            con.close()

    def get_row_count(self, table: str) -> int:
        """Get the row count for a table."""
        con = self._get_connection()
        try:
            result = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            return result[0] if result else 0
        finally:
            con.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    lake = DataLake(base_path="./test_datalake")
    print(f"AQI rows: {lake.get_row_count('raw_aqi')}")
    print(f"Weather rows: {lake.get_row_count('raw_weather')}")
    print(f"Feature rows: {lake.get_row_count('features')}")
