"""
AQICN/WAQI API Client — fetches official Nepal DoE air quality readings.
Free token from: https://aqicn.org/data-platform/token/

API docs: https://aqicn.org/json-api/doc/
Feed endpoints:
  - /feed/@{station_id}/  (most reliable, uses numeric station ID)
  - /feed/{city}/         (by city name, sometimes unreliable)
  - /search/?keyword=...  (discover stations by keyword)
"""

import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.waqi.info"

# Known AQICN station IDs for Kathmandu Valley.
# These are more reliable than city name lookups.
# Discovered via: https://api.waqi.info/search/?keyword=kathmandu&token=...
# Note: @8646, @11367, @12350 confirmed working via feed endpoint.
# Search shows @9468, @10495, @14868, @14866, @13592 but their feed returns "can not connect".
KATHMANDU_STATIONS = {
    "@8646": {"name": "kathmandu_ratnapark", "lat": 27.7030, "lon": 85.3135},
    "@11367": {"name": "kathmandu_us_embassy", "lat": 27.7385, "lon": 85.3165},
    "@12350": {"name": "patan_pulchowk", "lat": 27.6588, "lon": 85.3247},
}

# Fallback: city name lookups (less reliable, may return "can not connect")
KATHMANDU_CITY_NAMES = ["kathmandu", "patan", "bhaktapur"]

# Lat/lon for city name fallbacks
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

    def get_feed(self, feed_id: str) -> dict[str, Any] | None:
        """
        Fetch AQI feed by station ID (@nnn) or city name.
        Station IDs are more reliable than city names.
        """
        url = f"{BASE_URL}/feed/{feed_id}/"
        params = {"token": self.token}

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "ok":
                logger.warning(f"AQICN non-ok for {feed_id}: {data.get('data')}")
                return None

            return data.get("data")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch AQICN feed {feed_id}: {e}")
            return None

    def search_stations(self, keyword: str = "kathmandu") -> list[dict[str, Any]]:
        """Search for stations by keyword. Useful for discovering station IDs."""
        url = f"{BASE_URL}/search/"
        params = {"keyword": keyword, "token": self.token}

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "ok":
                return []

            return data.get("data", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"AQICN search failed for '{keyword}': {e}")
            return []

    def parse_feed(self, station_name: str, feed_data: dict[str, Any], coords: dict | None = None) -> dict[str, Any]:
        """Parse AQICN feed response into normalized AQI record."""
        iaqi = feed_data.get("iaqi", {})
        time_info = feed_data.get("time", {})

        # Extract coords from response if not provided
        if coords is None:
            city_geo = feed_data.get("city", {}).get("geo", [0, 0])
            coords = {"lat": city_geo[0] if len(city_geo) > 0 else 0,
                      "lon": city_geo[1] if len(city_geo) > 1 else 0}

        return {
            "station_id": f"aqicn_{station_name}",
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
        """
        Fetch readings from all Kathmandu Valley stations.
        Strategy: try known station IDs first (reliable), fall back to city names.
        """
        readings = []

        # Primary: fetch by known station IDs (most reliable)
        for feed_id, station_info in KATHMANDU_STATIONS.items():
            feed = self.get_feed(feed_id)
            if feed:
                record = self.parse_feed(
                    station_info["name"],
                    feed,
                    {"lat": station_info["lat"], "lon": station_info["lon"]},
                )
                readings.append(record)
                logger.info(
                    f"AQICN {station_info['name']}: AQI={record['aqi_us']}, PM2.5={record['pm25']}"
                )

        # Fallback: try city names if station IDs returned nothing
        if not readings:
            logger.warning("No data from station IDs, trying city name lookups...")
            for city in KATHMANDU_CITY_NAMES:
                feed = self.get_feed(city)
                if feed:
                    coords = CITY_COORDS.get(city)
                    record = self.parse_feed(city, feed, coords)
                    readings.append(record)
                    logger.info(f"AQICN {city}: AQI={record['aqi_us']}, PM2.5={record['pm25']}")

        if not readings:
            logger.warning("No AQICN data from any source for Kathmandu Valley")

        return readings


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = AQICNClient()

    # Discovery: show available stations
    print("Searching for Kathmandu stations...")
    results = client.search_stations("kathmandu")
    for r in results:
        uid = r.get("uid")
        name = r.get("station", {}).get("name", "?")
        aqi = r.get("aqi", "?")
        print(f"  Station @{uid}: {name} (AQI={aqi})")

    print("\nFetching all readings...")
    readings = client.fetch_all_cities()
    print(f"Fetched {len(readings)} readings:")
    for r in readings:
        print(f"  {r['station_id']}: AQI={r['aqi_us']}, PM2.5={r['pm25']} µg/m³")
