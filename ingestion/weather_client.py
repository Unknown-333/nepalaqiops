"""
Open-Meteo Weather Client — fetches weather data for Kathmandu Valley.
Completely free, no API key required.
"""

import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.open-meteo.com/v1"
HISTORY_URL = "https://archive-api.open-meteo.com/v1/archive"

# Kathmandu default coordinates
KTM_LAT = float(os.getenv("KTM_CENTER_LAT", "27.7172"))
KTM_LON = float(os.getenv("KTM_CENTER_LON", "85.3240"))

HOURLY_PARAMS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "surface_pressure",
]


class WeatherClient:
    """Client for Open-Meteo API — weather data for Kathmandu."""

    def __init__(self, lat: float = KTM_LAT, lon: float = KTM_LON):
        self.lat = lat
        self.lon = lon
        self.session = requests.Session()

    def get_current_forecast(self, forecast_days: int = 2) -> list[dict[str, Any]]:
        """Fetch current forecast (hourly) for next N days."""
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "hourly": ",".join(HOURLY_PARAMS),
            "forecast_days": forecast_days,
            "timezone": "UTC",
        }
        response = self.session.get(f"{BASE_URL}/forecast", params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        return self._parse_hourly(data)

    def get_historical(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch historical weather data (for model training)."""
        if end_date is None:
            end_date = datetime.now(timezone.utc) - timedelta(days=1)
        if start_date is None:
            start_date = end_date - timedelta(days=7)

        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "hourly": ",".join(HOURLY_PARAMS),
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "timezone": "UTC",
        }
        response = self.session.get(HISTORY_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        return self._parse_hourly(data)

    def _parse_hourly(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Parse Open-Meteo hourly response into normalized records."""
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])

        records = []
        for i, timestamp in enumerate(times):
            record = {
                "lat": self.lat,
                "lon": self.lon,
                "timestamp_utc": timestamp if timestamp.endswith("Z") else f"{timestamp}Z",
                "temp_c": hourly.get("temperature_2m", [None])[i] if i < len(hourly.get("temperature_2m", [])) else None,
                "humidity_pct": hourly.get("relative_humidity_2m", [None])[i] if i < len(hourly.get("relative_humidity_2m", [])) else None,
                "wind_speed_kmh": hourly.get("wind_speed_10m", [None])[i] if i < len(hourly.get("wind_speed_10m", [])) else None,
                "wind_dir_deg": hourly.get("wind_direction_10m", [None])[i] if i < len(hourly.get("wind_direction_10m", [])) else None,
                "precip_mm": hourly.get("precipitation", [None])[i] if i < len(hourly.get("precipitation", [])) else None,
                "pressure_hpa": hourly.get("surface_pressure", [None])[i] if i < len(hourly.get("surface_pressure", [])) else None,
            }
            records.append(record)

        logger.info(f"Parsed {len(records)} hourly weather records")
        return records


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = WeatherClient()

    print("=== Current Forecast ===")
    forecast = client.get_current_forecast(forecast_days=1)
    for r in forecast[:3]:
        print(f"  {r['timestamp_utc']}: {r['temp_c']}°C, {r['humidity_pct']}% RH, wind {r['wind_speed_kmh']} km/h")

    print(f"\n  ... ({len(forecast)} total hourly records)")
