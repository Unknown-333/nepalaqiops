"""
NepalAQI-Ops FastAPI Application — serves predictions and model metrics.
"""

import os
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from serving.routers import forecast, anomaly, health
from monitoring.prometheus_metrics import setup_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NepalAQI-Ops API",
    description="Air Quality Intelligence & Forecasting for Kathmandu Valley",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Prometheus metrics middleware
setup_metrics(app)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(forecast.router, prefix="/forecast", tags=["Forecast"])
app.include_router(anomaly.router, prefix="/anomalies", tags=["Anomalies"])


@app.on_event("startup")
async def startup_event():
    """Load models and initialize connection pools on startup."""
    import redis

    # Initialize Redis connection pool (shared across all requests)
    redis_pool = redis.ConnectionPool(
        host=os.getenv("REDIS_HOST", "redis"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        decode_responses=True,
        max_connections=20,
    )
    app.state.redis_pool = redis_pool

    from serving.model_loader import ModelStore
    app.state.model_store = ModelStore()
    await app.state.model_store.load_models()
    logger.info("NepalAQI-Ops API started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("NepalAQI-Ops API shutting down")
