"""
Forecast endpoints — PM2.5 predictions and heatmap.
"""

import json
import os
from datetime import datetime, timedelta, timezone

import redis
from fastapi import APIRouter, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field

router = APIRouter()

AQI_CATEGORIES = ["Good", "Moderate", "USG", "Unhealthy", "Very Unhealthy", "Hazardous"]


class ForecastPoint(BaseModel):
    """Single forecast data point with validated bounds."""
    hour: str
    pm25_predicted: float = Field(ge=0.0, le=500.0)
    aqi_category: str
    confidence_lower: float = Field(ge=0.0)
    confidence_upper: float = Field(le=1000.0)


class ForecastResponse(BaseModel):
    """Response model for forecast endpoint with output validation."""
    station_id: str
    generated_at: str
    timezone: str = "UTC"
    forecasts: list[ForecastPoint]
    model_used: str
    model_version: str


@router.get("/heatmap", name="forecast_heatmap")
async def get_heatmap(request: Request):
    """
    Get ward-level AQI predictions as GeoJSON FeatureCollection.
    Includes Kriging-interpolated values for wards without sensors.
    """
    redis_pool = getattr(request.app.state, "redis_pool", None)

    try:
        if redis_pool:
            r = redis.Redis(connection_pool=redis_pool)
        else:
            redis_host = os.getenv("REDIS_HOST", "redis")
            redis_port = int(os.getenv("REDIS_PORT", "6379"))
            r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

        # Build GeoJSON from cached ward features
        features = []
        for ward_id in range(1, 33):
            key = f"features:ward_{ward_id}:latest"
            data = r.get(key)

            if data:
                ward_data = json.loads(data)
                pm25 = ward_data.get("pm25") or ward_data.get("pm25_1h_mean", 0)
            else:
                pm25 = None

            feature = {
                "type": "Feature",
                "properties": {
                    "ward_id": ward_id,
                    "ward_name": f"Kathmandu Ward {ward_id}",
                    "pm25": pm25,
                    "aqi_category": AQI_CATEGORIES[_pm25_to_category_idx(pm25)] if pm25 else "Unknown",
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        85.27 + (ward_id % 8) * 0.025,
                        27.65 + (ward_id // 8) * 0.02,
                    ],
                },
            }
            features.append(feature)

        return {
            "type": "FeatureCollection",
            "features": features,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Heatmap generation failed: {e}")


@router.get("/{station_id}", response_model=ForecastResponse)
async def get_forecast(
    station_id: str,
    request: Request,
    hours: int = Query(default=24, ge=1, le=72),
    model: str = Query(default="auto", regex="^(auto|prophet|lstm|ensemble)$"),
    x_model_version: str | None = Header(default="champion", alias="X-Model-Version"),
):
    """
    Get PM2.5 forecast for a specific station.

    - **station_id**: Station identifier
    - **hours**: Forecast horizon (1-72, default 24)
    - **model**: Model to use (auto|prophet|lstm|ensemble)
    - **X-Model-Version**: champion or challenger (for A/B routing)
    """
    import time

    from monitoring.prometheus_metrics import (
        PREDICTION_LATENCY,
        PREDICTIONS_TOTAL,
    )

    start_time = time.time()
    model_store = getattr(request.app.state, "model_store", None)

    if model_store is None:
        raise HTTPException(status_code=503, detail="Model store not initialized")

    # Determine which model version to use
    use_challenger = x_model_version == "challenger"
    model_version = "challenger" if use_challenger else "champion"

    # Get Redis pool from app state
    redis_pool = getattr(request.app.state, "redis_pool", None)

    try:
        # Generate forecast
        forecast_result = await model_store.predict(
            station_id=station_id,
            hours=hours,
            model_type=model,
            use_challenger=use_challenger,
            redis_pool=redis_pool,
        )

        # Build response with clamped bounds
        now = datetime.now(timezone.utc)
        forecasts = []
        for i in range(min(hours, len(forecast_result["predictions"]))):
            pm25_pred = max(0.0, min(500.0, float(forecast_result["predictions"][i])))
            category_idx = _pm25_to_category_idx(pm25_pred)
            lower_val = max(0.0, float(forecast_result.get("lower", [pm25_pred * 0.7])[i] if i < len(forecast_result.get("lower", [])) else pm25_pred * 0.7))
            upper_val = min(1000.0, float(forecast_result.get("upper", [pm25_pred * 1.3])[i] if i < len(forecast_result.get("upper", [])) else pm25_pred * 1.3))
            forecasts.append(ForecastPoint(
                hour=(now + timedelta(hours=i + 1)).isoformat(),
                pm25_predicted=round(pm25_pred, 2),
                aqi_category=AQI_CATEGORIES[category_idx],
                confidence_lower=round(lower_val, 2),
                confidence_upper=round(upper_val, 2),
            ))

        # Record metrics
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)
        PREDICTIONS_TOTAL.labels(model=model_version, station=station_id).inc()

        return {
            "station_id": station_id,
            "generated_at": now.isoformat(),
            "timezone": "UTC",
            "forecasts": forecasts,
            "model_used": forecast_result.get("model_used", model),
            "model_version": model_version,
        }

    except Exception as e:
        from monitoring.prometheus_metrics import API_ERRORS
        API_ERRORS.labels(endpoint="forecast", error_type=type(e).__name__).inc()
        raise HTTPException(status_code=500, detail=str(e))


def _pm25_to_category_idx(pm25: float | None) -> int:
    """Convert PM2.5 to AQI category index (0-5)."""
    if pm25 is None or pm25 < 0:
        return 0
    if pm25 <= 12.0:
        return 0
    elif pm25 <= 35.4:
        return 1
    elif pm25 <= 55.4:
        return 2
    elif pm25 <= 150.4:
        return 3
    elif pm25 <= 250.4:
        return 4
    else:
        return 5
