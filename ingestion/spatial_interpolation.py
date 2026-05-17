"""
Spatial Interpolation — Ordinary Kriging for ward-level PM2.5 estimates.
Uses pykrige to interpolate from sparse sensor data to Kathmandu ward centroids.
"""

import os
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WARD_CENTROIDS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "features", "ward_centroids.csv"
)


class SpatialInterpolator:
    """Ordinary Kriging interpolation for AQI coverage gaps."""

    def __init__(self):
        self.ward_centroids = self._load_ward_centroids()

    def _load_ward_centroids(self) -> pd.DataFrame:
        """Load Kathmandu ward centroid coordinates."""
        try:
            df = pd.read_csv(WARD_CENTROIDS_PATH)
            logger.info(f"Loaded {len(df)} ward centroids")
            return df
        except FileNotFoundError:
            logger.warning(f"Ward centroids file not found at {WARD_CENTROIDS_PATH}")
            return pd.DataFrame(columns=["ward_id", "ward_name", "lat", "lon"])

    def interpolate_pm25(
        self,
        sensor_readings: list[dict[str, Any]],
        variogram_model: str = "spherical",
    ) -> list[dict[str, Any]]:
        """
        Perform Ordinary Kriging to estimate PM2.5 at all ward centroids.

        Args:
            sensor_readings: List of dicts with lat, lon, pm25 values
            variogram_model: Kriging variogram model (spherical, exponential, gaussian)

        Returns:
            List of interpolated records tagged with source='kriging_interpolated'
        """
        # Filter to readings that have valid PM2.5
        valid = [r for r in sensor_readings if r.get("pm25") is not None]
        if len(valid) < 3:
            logger.warning("Need at least 3 sensor readings for Kriging. Skipping.")
            return []

        lats = np.array([r["lat"] for r in valid])
        lons = np.array([r["lon"] for r in valid])
        pm25_values = np.array([r["pm25"] for r in valid])

        try:
            from pykrige.ok import OrdinaryKriging

            ok = OrdinaryKriging(
                lons,
                lats,
                pm25_values,
                variogram_model=variogram_model,
                verbose=False,
                enable_plotting=False,
            )

            # Get target ward centroids
            if self.ward_centroids.empty:
                logger.warning("No ward centroids available for interpolation")
                return []

            target_lons = self.ward_centroids["lon"].values
            target_lats = self.ward_centroids["lat"].values

            # Execute kriging
            z_pred, ss_pred = ok.execute(
                "points",
                target_lons,
                target_lats,
            )

            # Build result records
            timestamp = valid[0].get("timestamp_utc", "")
            results = []
            for i, row in self.ward_centroids.iterrows():
                results.append({
                    "station_id": f"ward_{row['ward_id']}",
                    "source": "kriging_interpolated",
                    "lat": row["lat"],
                    "lon": row["lon"],
                    "timestamp_utc": timestamp,
                    "pm25": float(z_pred[i]) if not np.isnan(z_pred[i]) else None,
                    "pm10": None,
                    "no2": None,
                    "o3": None,
                    "co": None,
                    "aqi_us": None,
                    "kriging_variance": float(ss_pred[i]),
                    "ward_name": row.get("ward_name", ""),
                })

            logger.info(
                f"Kriging interpolation complete: {len(results)} ward estimates "
                f"from {len(valid)} sensor readings"
            )
            return results

        except ImportError:
            logger.error("pykrige not installed. Run: pip install pykrige")
            return []
        except Exception as e:
            logger.error(f"Kriging interpolation failed: {e}")
            return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with synthetic sensor data
    test_sensors = [
        {"lat": 27.7172, "lon": 85.3240, "pm25": 65.0, "timestamp_utc": "2025-01-01T12:00:00Z"},
        {"lat": 27.6588, "lon": 85.3247, "pm25": 78.0, "timestamp_utc": "2025-01-01T12:00:00Z"},
        {"lat": 27.6710, "lon": 85.4298, "pm25": 55.0, "timestamp_utc": "2025-01-01T12:00:00Z"},
        {"lat": 27.6783, "lon": 85.2789, "pm25": 90.0, "timestamp_utc": "2025-01-01T12:00:00Z"},
        {"lat": 27.7000, "lon": 85.3500, "pm25": 72.0, "timestamp_utc": "2025-01-01T12:00:00Z"},
    ]

    interpolator = SpatialInterpolator()
    results = interpolator.interpolate_pm25(test_sensors)
    print(f"Interpolated {len(results)} ward estimates")
    for r in results[:5]:
        print(f"  {r['station_id']} ({r['ward_name']}): PM2.5={r['pm25']:.1f}")
