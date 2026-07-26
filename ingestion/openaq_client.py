"""
OpenAQ API v3 Client — fetches air quality data for Nepal/Kathmandu Valley.
API v1/v2 were retired Jan 31, 2025. This uses exclusively v3 endpoints.
"""

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.openaq.org/v3"

# Kathmandu Valley bounding box
KTM_LAT_MIN = float(os.getenv("KTM_LAT_MIN", "27.60"))
KTM_LAT_MAX = float(os.getenv("KTM_LAT_MAX", "27.80"))
KTM_LON_MIN = float(os.getenv("KTM_LON_MIN", "85.20"))
KTM_LON_MAX = float(os.getenv("KTM_LON_MAX", "85.45"))


class OpenAQClient:
    """Client for OpenAQ API v3 with retry logic and rate limit handling."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("OPENAQ_API_KEY") or ""
        self.session = requests.Session()
        self.session.headers.update({
            "X-API-Key": self.api_key,
            "Accept": "application/json",
        })
        self._max_retries = 3
        self._base_delay = 1.0

    def _request(self, endpoint: str, params: dict | None = None) -> dict[str, Any]:
        """Make a request with exponential backoff retry."""
        url = f"{BASE_URL}{endpoint}"
        for attempt in range(self._max_retries):
            try:
                response = self.session.get(url, params=params, timeout=30)

                # Honour rate limit headers
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    logger.warning(f"Rate limited. Waiting {retry_after}s...")
                    time.sleep(retry_after)
                    continue

                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                delay = self._base_delay * (2 ** attempt)
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self._max_retries}): {e}. "
                    f"Retrying in {delay}s..."
                )
                if attempt < self._max_retries - 1:
                    time.sleep(delay)
                else:
                    raise

        return {}

    def get_nepal_locations(self, limit: int = 100) -> list[dict[str, Any]]:
        """Discover all air quality monitoring stations in Nepal."""
        # Nepal country_id in OpenAQ v3 — use ISO code lookup
        data = self._request("/locations", params={
            "countries_id": 164,  # Nepal country ID in OpenAQ
            "limit": limit,
        })
        return data.get("results", [])

    def get_kathmandu_locations(self) -> list[dict[str, Any]]:
        """Filter Nepal locations to Kathmandu Valley bounding box."""
        locations = self.get_nepal_locations()
        ktm_locations = []
        for loc in locations:
            coords = loc.get("coordinates", {})
            lat = coords.get("latitude", 0)
            lon = coords.get("longitude", 0)
            if (KTM_LAT_MIN <= lat <= KTM_LAT_MAX and
                    KTM_LON_MIN <= lon <= KTM_LON_MAX):
                ktm_locations.append(loc)
        return ktm_locations

    def get_sensor_measurements(
        self,
        sensor_id: int,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch hourly measurements for a specific sensor."""
        if date_from is None:
            date_from = datetime.now(timezone.utc) - timedelta(hours=24)
        if date_to is None:
            date_to = datetime.now(timezone.utc)

        params = {
            "date_from": date_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "date_to": date_to.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        data = self._request(
            f"/sensors/{sensor_id}/measurements/hourly",
            params=params,
        )
        return data.get("results", [])

    def get_latest_measurements(self, location_id: int) -> list[dict[str, Any]]:
        """Get the latest measurements for a location."""
        data = self._request(f"/locations/{location_id}/latest")
        return data.get("results", [])

    def fetch_all_kathmandu_readings(self) -> list[dict[str, Any]]:
        """
        Fetch latest readings from all Kathmandu Valley stations.
        Returns normalized records ready for Kafka production.
        """
        locations = self.get_kathmandu_locations()
        readings = []

        for loc in locations:
            location_id = loc.get("id")
            coords = loc.get("coordinates", {})
            sensors = loc.get("sensors", [])

            for sensor in sensors:
                sensor_id = sensor.get("id")
                parameter = sensor.get("parameter", {}).get("name", "").lower()

                try:
                    measurements = self.get_sensor_measurements(sensor_id)
                except Exception as e:
                    logger.error(f"Failed to fetch sensor {sensor_id}: {e}")
                    continue

                for m in measurements:
                    reading = {
                        "station_id": str(location_id),
                        "source": "openaq",
                        "lat": coords.get("latitude"),
                        "lon": coords.get("longitude"),
                        "timestamp_utc": m.get("period", {}).get("datetimeFrom", {}).get("utc"),
                        "pm25": m.get("value") if parameter == "pm25" else None,
                        "pm10": m.get("value") if parameter == "pm10" else None,
                        "no2": m.get("value") if parameter == "no2" else None,
                        "o3": m.get("value") if parameter == "o3" else None,
                        "co": m.get("value") if parameter == "co" else None,
                        "aqi_us": None,  # OpenAQ provides raw values, not AQI
                    }
                    readings.append(reading)

        logger.info(f"Fetched {len(readings)} readings from {len(locations)} Kathmandu stations")
        return readings


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = OpenAQClient()
    locations = client.get_kathmandu_locations()
    print(f"Found {len(locations)} stations in Kathmandu Valley:")
    for loc in locations:
        name = loc.get("name", "Unknown")
        coords = loc.get("coordinates", {})
        print(f"  - {name} ({coords.get('latitude')}, {coords.get('longitude')})")
