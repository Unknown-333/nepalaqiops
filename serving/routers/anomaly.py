"""
Anomaly endpoints — latest anomaly events from Kafka.
"""

from datetime import datetime, timezone
from typing import cast

from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


@router.get("/latest")
async def get_latest_anomalies(request: Request, limit: int = 50):
    """
    Return the last N anomaly events from the anomaly.alerts topic.
    """
    import json
    import os

    import redis

    redis_host = os.getenv("REDIS_HOST", "redis")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))

    try:
        r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

        # Get anomaly events from Redis list (populated by Kafka consumer)
        raw_events = cast("list[str]", r.lrange("anomaly:events", 0, limit - 1))

        events = []
        for raw in raw_events:
            try:
                event = json.loads(raw)
                events.append(event)
            except json.JSONDecodeError:
                continue

        return {
            "count": len(events),
            "events": events,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve anomalies: {e}")
