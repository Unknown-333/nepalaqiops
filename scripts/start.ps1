# NepalAQI-Ops — Full Stack Startup Script (Windows PowerShell)
# Usage: .\scripts\start.ps1

$ErrorActionPreference = "Continue"

Write-Host "=== NepalAQI-Ops Starting ===" -ForegroundColor Cyan

# Check prerequisites
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Docker is not installed. Install from https://docs.docker.com/get-docker/" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path .env)) {
    Write-Host "ERROR: .env file not found. Run: Copy-Item .env.example .env" -ForegroundColor Red
    Write-Host "Then edit .env to add your AQICN_TOKEN and OPENAQ_API_KEY"
    exit 1
}

# Start all services
Write-Host "[1/5] Building and starting containers..." -ForegroundColor Yellow
docker compose up --build -d
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: docker compose up failed" -ForegroundColor Red; exit 1 }

# Wait for health
Write-Host "[2/5] Waiting for services to become healthy (~60s)..." -ForegroundColor Yellow
Start-Sleep -Seconds 15
docker compose ps

# Fix permissions
Write-Host "[3/5] Fixing datalake permissions..." -ForegroundColor Yellow
docker compose exec --user root airflow-scheduler chmod 777 /opt/airflow/datalake 2>$null

# Unpause DAGs
Write-Host "[4/5] Unpausing Airflow DAGs..." -ForegroundColor Yellow
docker compose exec airflow-scheduler airflow dags unpause ingest_aqi_dag 2>$null | Select-String "is_paused"
docker compose exec airflow-scheduler airflow dags unpause drift_monitor_dag 2>$null | Select-String "is_paused"

# Run first ingestion
$today = (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd")
Write-Host "[5/5] Running first data ingestion ($today)..." -ForegroundColor Yellow
docker compose exec airflow-scheduler airflow tasks test ingest_aqi_dag persist_to_datalake $today 2>&1 | Select-String "SUCCESS|FAILED|Inserted|readings"

Write-Host ""
Write-Host "=== NepalAQI-Ops Ready ===" -ForegroundColor Green
Write-Host ""
Write-Host "  Forecast API:   http://localhost:8000/forecast/aqicn_kathmandu_ratnapark"
Write-Host "  Health Check:   http://localhost:8000/health"
Write-Host "  Airflow UI:     http://localhost:8080  (admin/admin)"
Write-Host "  Streamlit:      http://localhost:8501"
Write-Host "  MLflow:         http://localhost:5000"
Write-Host "  Grafana:        http://localhost:3000  (admin/admin)"
Write-Host "  MinIO Console:  http://localhost:9001  (minioadmin/minioadmin123)"
Write-Host ""
Write-Host "To stop:  docker compose down"
Write-Host "To reset: docker compose down -v"
