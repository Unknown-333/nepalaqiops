"""
Feature Engineering — computes rolling, lag, cyclical, and derived features.
"""

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from features.calendar_flags import CalendarFlags

logger = logging.getLogger(__name__)

# Kathmandu city center for distance calculation
KTM_CENTER_LAT = 27.7172
KTM_CENTER_LON = 85.3240


class FeatureEngineer:
    """Computes all features for the AQI forecasting pipeline."""

    def __init__(self):
        self.calendar = CalendarFlags()

    def compute_all_features(
        self,
        aqi_df: pd.DataFrame,
        weather_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute all features from raw AQI and weather data.

        Args:
            aqi_df: Raw AQI data with columns: station_id, timestamp_utc, pm25, etc.
            weather_df: Weather data with columns: timestamp_utc, temp_c, etc.

        Returns:
            DataFrame with all engineered features, one row per station per hour.
        """
        if aqi_df.empty:
            logger.warning("Empty AQI dataframe, skipping feature engineering")
            return pd.DataFrame()

        # Ensure datetime type
        aqi_df = aqi_df.copy()
        aqi_df["timestamp_utc"] = pd.to_datetime(aqi_df["timestamp_utc"], utc=True)
        aqi_df = aqi_df.sort_values(["station_id", "timestamp_utc"])

        # Process per station
        results = []
        for station_id, group in aqi_df.groupby("station_id"):
            station_features = self._compute_station_features(group, weather_df)
            if not station_features.empty:
                results.append(station_features)

        if not results:
            return pd.DataFrame()

        features_df = pd.concat(results, ignore_index=True)

        # Add calendar flags
        features_df = self.calendar.add_flags_to_dataframe(features_df)

        return features_df

    def _compute_station_features(
        self,
        station_df: pd.DataFrame,
        weather_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute features for a single station."""
        df = station_df.copy()
        df = df.set_index("timestamp_utc").sort_index()

        # Deduplicate hourly (keep first reading per hour)
        df = df[~df.index.duplicated(keep="first")]

        # ===== TIME FEATURES =====
        df["hour_of_day"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month
        df["is_weekend"] = df.index.dayofweek >= 5

        # Cyclical encoding
        df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # ===== ROLLING STATISTICS (on PM2.5) =====
        if "pm25" in df.columns:
            df["pm25_1h_mean"] = df["pm25"].rolling("1h", min_periods=1).mean()
            df["pm25_3h_mean"] = df["pm25"].rolling("3h", min_periods=1).mean()
            df["pm25_6h_mean"] = df["pm25"].rolling("6h", min_periods=1).mean()
            df["pm25_12h_mean"] = df["pm25"].rolling("12h", min_periods=1).mean()
            df["pm25_24h_mean"] = df["pm25"].rolling("24h", min_periods=1).mean()
            df["pm25_1h_std"] = df["pm25"].rolling("1h", min_periods=2).std()
            df["pm25_6h_std"] = df["pm25"].rolling("6h", min_periods=2).std()
            df["pm25_24h_std"] = df["pm25"].rolling("24h", min_periods=2).std()

            # ===== LAG FEATURES =====
            df["pm25_lag_1h"] = df["pm25"].shift(1)
            df["pm25_lag_3h"] = df["pm25"].shift(3)
            df["pm25_lag_6h"] = df["pm25"].shift(6)
            df["pm25_lag_12h"] = df["pm25"].shift(12)
            df["pm25_lag_24h"] = df["pm25"].shift(24)
            df["pm25_lag_48h"] = df["pm25"].shift(48)
            df["pm25_lag_168h"] = df["pm25"].shift(168)  # 1 week ago

        # ===== WEATHER FEATURES =====
        df = self._merge_weather(df, weather_df)

        # ===== DERIVED FEATURES =====
        if "pm25" in df.columns:
            df["aqi_us_category"] = df["pm25"].apply(self._pm25_to_aqi_category)

        # Station distance to city center
        if "lat" in df.columns and "lon" in df.columns:
            lat = df["lat"].iloc[0] if not df["lat"].isna().all() else KTM_CENTER_LAT
            lon = df["lon"].iloc[0] if not df["lon"].isna().all() else KTM_CENTER_LON
            df["station_distance_to_city_center_km"] = self._haversine(
                lat, lon, KTM_CENTER_LAT, KTM_CENTER_LON
            )
        else:
            df["station_distance_to_city_center_km"] = 0.0

        # Reset index and ensure station_id column
        df = df.reset_index()
        if "station_id" not in df.columns:
            df["station_id"] = station_df["station_id"].iloc[0]

        return df

    def _merge_weather(self, df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        """Merge weather features using nearest hour matching."""
        if weather_df.empty:
            df["temp_c"] = np.nan
            df["humidity_pct"] = np.nan
            df["wind_speed_kmh"] = np.nan
            df["wind_dir_sin"] = np.nan
            df["wind_dir_cos"] = np.nan
            df["precip_mm"] = np.nan
            df["pressure_hpa"] = np.nan
            df["precip_6h_cumulative"] = np.nan
            return df

        weather = weather_df.copy()
        weather["timestamp_utc"] = pd.to_datetime(weather["timestamp_utc"], utc=True)
        weather = weather.set_index("timestamp_utc").sort_index()
        weather = weather[~weather.index.duplicated(keep="first")]

        # Encode wind direction as sin/cos
        if "wind_dir_deg" in weather.columns:
            weather["wind_dir_sin"] = np.sin(np.radians(weather["wind_dir_deg"]))
            weather["wind_dir_cos"] = np.cos(np.radians(weather["wind_dir_deg"]))
        else:
            weather["wind_dir_sin"] = np.nan
            weather["wind_dir_cos"] = np.nan

        # Cumulative precipitation
        if "precip_mm" in weather.columns:
            weather["precip_6h_cumulative"] = weather["precip_mm"].rolling("6h", min_periods=1).sum()
        else:
            weather["precip_6h_cumulative"] = np.nan

        # Merge on nearest timestamp (asof merge)
        weather_cols = [
            "temp_c", "humidity_pct", "wind_speed_kmh",
            "wind_dir_sin", "wind_dir_cos", "precip_mm",
            "pressure_hpa", "precip_6h_cumulative",
        ]
        available_cols = [c for c in weather_cols if c in weather.columns]

        for col in weather_cols:
            if col in weather.columns:
                # Reindex weather to match AQI timestamps
                df[col] = weather[col].reindex(df.index, method="nearest", tolerance="1h")
            else:
                df[col] = np.nan

        return df

    @staticmethod
    def _pm25_to_aqi_category(pm25: float | None) -> int | None:
        """Convert PM2.5 value to US AQI category (0-5)."""
        if pm25 is None or pd.isna(pm25):
            return None
        if pm25 <= 12.0:
            return 0  # Good
        elif pm25 <= 35.4:
            return 1  # Moderate
        elif pm25 <= 55.4:
            return 2  # Unhealthy for Sensitive Groups
        elif pm25 <= 150.4:
            return 3  # Unhealthy
        elif pm25 <= 250.4:
            return 4  # Very Unhealthy
        else:
            return 5  # Hazardous

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two coordinates in km."""
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c
