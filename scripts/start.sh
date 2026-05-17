#!/usr/bin/env bash
# NepalAQI-Ops — Full Stack Startup Script
# Usage: ./scripts/start.sh
set -e

echo "=== NepalAQI-Ops Starting ==="

# Check prerequisites
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed. Install from https://docs.docker.com/get-docker/"
    exit 1
fi

if [ ! -f .env ]; then
    echo "ERROR: .env file not found. Run: cp .env.example .env"
    echo "Then edit .env to add your AQICN_TOKEN and OPENAQ_API_KEY"
    exit 1
fi

# Start all services
echo "[1/5] Building and starting containers..."
docker compose up --build -d

# Wait for health
echo "[2/5] Waiting for services to become healthy (~60s)..."
sleep 10
docker compose ps --format "table {{.Name}}\t{{.Status}}" | head -20

# Fix permissions
echo "[3/5] Fixing datalake permissions..."
docker compose exec --user root airflow-scheduler chmod 777 /opt/airflow/datalake 2>/dev/null || true

# Unpause DAGs
echo "[4/5] Unpausing Airflow DAGs..."
docker compose exec airflow-scheduler airflow dags unpause ingest_aqi_dag 2>/dev/null | tail -3
docker compose exec airflow-scheduler airflow dags unpause drift_monitor_dag 2>/dev/null | tail -3

# Run first ingestion
TODAY=$(date -u +%Y-%m-%d)
echo "[5/5] Running first data ingestion ($TODAY)..."
docker compose exec airflow-scheduler airflow tasks test ingest_aqi_dag persist_to_datalake "$TODAY" 2>&1 | grep -E "SUCCESS|FAILED|Inserted|readings"

echo ""
echo "=== NepalAQI-Ops Ready ==="
echo ""
echo "  Forecast API:   http://localhost:8000/forecast/aqicn_kathmandu_ratnapark"
echo "  Health Check:   http://localhost:8000/health"
echo "  Airflow UI:     http://localhost:8080  (admin/admin)"
echo "  Streamlit:      http://localhost:8501"
echo "  MLflow:         http://localhost:5000"
echo "  Grafana:        http://localhost:3000  (admin/admin)"
echo "  MinIO Console:  http://localhost:9001  (minioadmin/minioadmin123)"
echo ""
echo "To stop:  docker compose down"
echo "To reset: docker compose down -v"
