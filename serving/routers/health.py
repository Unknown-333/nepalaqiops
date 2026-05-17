"""
Health endpoint — system status and Prometheus scrape target.
"""

import os
from datetime import datetime, timezone

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health_check(request: Request):
    """System health check — returns model and service status."""
    model_store = getattr(request.app.state, "model_store", None)

    champion_model = "none"
    challenger_model = "none"
    last_retrain = "unknown"

    if model_store:
        champion_model = model_store.champion_info.get("name", "none")
        challenger_model = model_store.challenger_info.get("name", "none")
        last_retrain = model_store.last_loaded.isoformat() if model_store.last_loaded else "unknown"

    return {
        "status": "ok",
        "champion_model": champion_model,
        "challenger_model": challenger_model,
        "last_retrain": last_retrain,
        "kafka_lag": 0,  # Updated by Kafka consumer in production
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/retrain")
async def trigger_retrain(request: Request):
    """
    Trigger model retrain via Airflow REST API.
    Secured with API key header.
    """
    import httpx

    api_key = request.headers.get("X-API-Key", "")
    expected_key = os.getenv("API_SECRET_KEY", "")
    if not expected_key or api_key != expected_key:
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="Invalid API key")

    airflow_url = "http://airflow-webserver:8080/api/v1/dags/train_evaluate_dag/dagRuns"
    airflow_user = os.getenv("AIRFLOW_ADMIN_USER", "admin")
    airflow_pass = os.getenv("AIRFLOW_ADMIN_PASSWORD", "admin")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            airflow_url,
            json={"conf": {"triggered_by": "api"}},
            auth=(airflow_user, airflow_pass),
            timeout=10,
        )

    if response.status_code in (200, 201):
        data = response.json()
        return {"dag_run_id": data.get("dag_run_id", "unknown"), "status": "triggered"}
    else:
        return {"status": "failed", "detail": response.text}
