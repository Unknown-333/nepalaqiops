"""
Prometheus Metrics — custom metrics for NepalAQI-Ops observability.
"""

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware

# ===== Custom Metrics =====

PREDICTION_LATENCY = Histogram(
    "nepalaqiops_prediction_latency_seconds",
    "Prediction request latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

PREDICTIONS_TOTAL = Counter(
    "nepalaqiops_predictions_total",
    "Total number of predictions served",
    ["model", "station"],
)

API_ERRORS = Counter(
    "nepalaqiops_api_errors_total",
    "Total API errors",
    ["endpoint", "error_type"],
)

DRIFT_SCORE = Gauge(
    "nepalaqiops_drift_score",
    "Feature drift PSI score",
    ["model", "feature"],
)

MODEL_RMSE = Gauge(
    "nepalaqiops_model_rmse",
    "Model RMSE metric",
    ["model", "horizon_hours"],
)

KAFKA_CONSUMER_LAG = Gauge(
    "nepalaqiops_kafka_consumer_lag",
    "Kafka consumer lag in messages",
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware to track request latency and errors."""

    async def dispatch(self, request: Request, call_next):
        response = None
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            endpoint = request.url.path
            API_ERRORS.labels(endpoint=endpoint, error_type=type(e).__name__).inc()
            raise
        finally:
            if response and response.status_code >= 400:
                endpoint = request.url.path
                API_ERRORS.labels(
                    endpoint=endpoint,
                    error_type=f"http_{response.status_code}",
                ).inc()


def setup_metrics(app: FastAPI):
    """Configure Prometheus metrics endpoint and middleware."""

    app.add_middleware(PrometheusMiddleware)

    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        """Prometheus scrape endpoint."""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )
