"""
Data Quality Schemas — Pandera validation for ingestion and training pipelines.
Catches API schema changes, NaN floods, and physically impossible values.
"""

import pandera as pa
from pandera import Column, Check, DataFrameSchema
import numpy as np


# =============================================================================
# INGESTION LAYER SCHEMAS
# =============================================================================

AQI_READING_SCHEMA = DataFrameSchema(
    columns={
        "station_id": Column(str, nullable=False, checks=[
            Check.str_length(min_value=1, max_value=100),
        ]),
        "source": Column(str, nullable=False, checks=[
            Check.isin(["openaq", "aqicn", "kriging_interpolated", "integration_test"]),
        ]),
        "lat": Column(float, nullable=False, checks=[
            Check.in_range(20.0, 35.0),  # Nepal latitude bounds
        ]),
        "lon": Column(float, nullable=False, checks=[
            Check.in_range(80.0, 90.0),  # Nepal longitude bounds
        ]),
        "timestamp_utc": Column(checks=[
            Check(lambda s: s.notna().all(), error="timestamp cannot be null"),
        ]),
        "pm25": Column(float, nullable=True, checks=[
            Check.in_range(0.0, 999.9),  # Physical bounds
        ]),
        "pm10": Column(float, nullable=True, checks=[
            Check.in_range(0.0, 2000.0),
        ]),
        "no2": Column(float, nullable=True, checks=[
            Check.in_range(0.0, 500.0),
        ]),
        "o3": Column(float, nullable=True, checks=[
            Check.in_range(0.0, 600.0),
        ]),
        "co": Column(float, nullable=True, checks=[
            Check.in_range(0.0, 100.0),
        ]),
        "aqi_us": Column(int, nullable=True, checks=[
            Check.in_range(0, 500),
        ]),
    },
    checks=[
        # At least pm25 OR pm10 must be present (not all null)
        Check(
            lambda df: df[["pm25", "pm10"]].notna().any(axis=1).mean() > 0.5,
            error="More than 50% of rows have neither PM2.5 nor PM10 — data source likely broken",
        ),
    ],
    coerce=True,
)


WEATHER_READING_SCHEMA = DataFrameSchema(
    columns={
        "lat": Column(float, nullable=False, checks=[Check.in_range(20.0, 35.0)]),
        "lon": Column(float, nullable=False, checks=[Check.in_range(80.0, 90.0)]),
        "timestamp_utc": Column(checks=[
            Check(lambda s: s.notna().all(), error="timestamp cannot be null"),
        ]),
        "temp_c": Column(float, nullable=True, checks=[
            Check.in_range(-20.0, 50.0),  # Kathmandu: -5 to 40°C realistic
        ]),
        "humidity_pct": Column(float, nullable=True, checks=[
            Check.in_range(0.0, 100.0),
        ]),
        "wind_speed_kmh": Column(float, nullable=True, checks=[
            Check.in_range(0.0, 200.0),
        ]),
        "wind_dir_deg": Column(float, nullable=True, checks=[
            Check.in_range(0.0, 360.0),
        ]),
        "precip_mm": Column(float, nullable=True, checks=[
            Check.in_range(0.0, 500.0),
        ]),
        "pressure_hpa": Column(float, nullable=True, checks=[
            Check.in_range(700.0, 1100.0),  # Kathmandu at ~1400m elevation
        ]),
    },
    coerce=True,
)


# =============================================================================
# TRAINING PIPELINE SCHEMA
# =============================================================================

TRAINING_FEATURES_SCHEMA = DataFrameSchema(
    columns={
        "station_id": Column(str, nullable=False),
        "timestamp_utc": Column(nullable=False),
        "pm25_24h_mean": Column(float, nullable=True, checks=[
            Check.in_range(0.0, 999.9),
        ]),
        "pm25_lag_1h": Column(float, nullable=True, checks=[
            Check.in_range(0.0, 999.9),
        ]),
        "temp_c": Column(float, nullable=True, checks=[
            Check.in_range(-20.0, 50.0),
        ]),
        "humidity_pct": Column(float, nullable=True, checks=[
            Check.in_range(0.0, 100.0),
        ]),
    },
    checks=[
        # NaN flood gate: reject batches where >30% of critical columns are NaN
        Check(
            lambda df: df[["pm25_24h_mean", "pm25_lag_1h"]].notna().mean().min() > 0.7,
            error="NaN flood detected: >30% of PM2.5 features are null. "
                  "Possible API outage or schema change.",
        ),
        # Frozen value detection: if pm25_lag_1h has zero variance, data is likely stuck
        Check(
            lambda df: df["pm25_lag_1h"].std() > 0.01 if len(df) > 10 else True,
            error="Frozen sensor detected: PM2.5 lag has zero variance (stuck reading).",
        ),
    ],
    coerce=True,
)


def validate_aqi_batch(df, raise_on_fail: bool = True):
    """
    Validate an AQI reading DataFrame before ingestion.
    Returns (validated_df, errors) tuple.
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        validated = AQI_READING_SCHEMA.validate(df, lazy=True)
        return validated, None
    except pa.errors.SchemaErrors as e:
        logger.error(f"Data quality validation failed: {e.failure_cases.shape[0]} failures")
        logger.error(f"First failures:\n{e.failure_cases.head()}")
        if raise_on_fail:
            raise
        return df, e


def validate_training_features(df, raise_on_fail: bool = True):
    """
    Validate features DataFrame before ML training.
    Catches NaN floods, schema drift, and frozen sensors.
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        validated = TRAINING_FEATURES_SCHEMA.validate(df, lazy=True)
        return validated, None
    except pa.errors.SchemaErrors as e:
        logger.error(f"Training data validation failed: {e.failure_cases.shape[0]} failures")
        if raise_on_fail:
            raise
        return df, e
