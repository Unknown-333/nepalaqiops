"""
AQICN/WAQI API Client — fetches official Nepal DoE air quality readings.
Free token from: https://aqicn.org/data-platform/token/
"""

import os
import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.waqi.info"

# Target cities in Kathmandu Valley
KATHMANDU_CITIES = ["kathmandu", "patan", "bhaktapur", "kirtipur"]

# Lat/lon for each city (for record enrichment)
CITY_COORDS = {
    "kathmandu": {"lat": 27.7172, "lon": 85.3240},
    "patan": {"lat": 27.6588, "lon": 85.3247},
    "bhaktapur": {"lat": 27.6710, "lon": 85.4298},
    "kirtipur": {"lat": 27.6783, "lon": 85.2789},
}


class AQICNClient:
    """Client for AQICN/WAQI API — official Nepal DoE air quality data."""

    def __init__(self, token: str | None = None):
        self.token = token or os.getenv("AQICN_TOKEN", "")
        self.session = requests.Session()

    def get_city_feed(self, city: str) -> dict[str, Any] | None:
        """Fetch current AQI feed for a specific city."""
        url = f"{BASE_URL}/feed/{city}/"
        params = {"token": self.token}

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "ok":
                logger.warning(f"AQICN returned non-ok status for {city}: {data.get('data')}")
                return None

            return data.get("data")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch AQICN data for {city}: {e}")
            return None

    def parse_feed(self, city: str, feed_data: dict[str, Any]) -> dict[str, Any]:
        """Parse AQICN feed response into normalized AQI record."""
        iaqi = feed_data.get("iaqi", {})
        time_info = feed_data.get("time", {})
        coords = CITY_COORDS.get(city, {"lat": 0, "lon": 0})

        return {
            "station_id": f"aqicn_{city}",
            "source": "aqicn",
            "lat": coords["lat"],
            "lon": coords["lon"],
            "timestamp_utc": time_info.get("iso", ""),
            "pm25": iaqi.get("pm25", {}).get("v"),
            "pm10": iaqi.get("pm10", {}).get("v"),
            "no2": iaqi.get("no2", {}).get("v"),
            "o3": iaqi.get("o3", {}).get("v"),
            "co": iaqi.get("co", {}).get("v"),
            "aqi_us": feed_data.get("aqi"),
        }

    def fetch_all_cities(self) -> list[dict[str, Any]]:
        """Fetch readings from all Kathmandu Valley cities."""
        readings = []
        for city in KATHMANDU_CITIES:
            feed = self.get_city_feed(city)
            if feed:
                record = self.parse_feed(city, feed)
                readings.append(record)
                logger.info(f"AQICN {city}: AQI={record['aqi_us']}, PM2.5={record['pm25']}")
            else:
                logger.warning(f"No data from AQICN for {city}")

        return readings


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = AQICNClient()
    readings = client.fetch_all_cities()
    print(f"\nFetched {len(readings)} city readings:")
    for r in readings:
        print(f"  {r['station_id']}: AQI={r['aqi_us']}, PM2.5={r['pm25']} µg/m³")
