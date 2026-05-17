# NepalAQI-Ops — System Architecture

## Overview

NepalAQI-Ops is a production-grade MLOps pipeline for real-time PM2.5 air quality forecasting in Kathmandu Valley. The system ingests data from multiple sources, engineers features, trains ensemble models, detects drift, and serves predictions through a low-latency API.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                                      │
│   AQICN API  ─┐    OpenWeather API  ─┐    Nepal Gov Stations  ─┐        │
└───────────────┼────────────────────────┼─────────────────────────┼────────┘
                │                        │                         │
                ▼                        ▼                         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION (Airflow 2.9)                            │
│                                                                          │
│   ┌─────────────┐  ┌─────────────────┐  ┌───────────────────┐          │
│   │ ingest_aqi  │→ │ feature_eng     │→ │ train_evaluate    │          │
│   │ ingest_wx   │→ │ (ExternalSensor)│  │ (Prophet+LSTM)    │          │
│   └─────────────┘  └─────────────────┘  └───────────────────┘          │
│                                              ↑                           │
│   ┌─────────────────────────────────────────┐│                           │
│   │ drift_monitor (PSI + RMSE, festival-aware)│                          │
│   └──────────────────────────────────────────┘                           │
└──────────────────────────────────────────────────────────────────────────┘
                │                        │
                ▼                        ▼
┌──────────────────────┐    ┌──────────────────────────────────────────────┐
│   STORAGE LAYER      │    │              MODEL REGISTRY                   │
│                      │    │                                              │
│  DuckDB (OLAP)       │    │  MLflow 2.13                                 │
│  ├─ raw_aqi          │    │  ├─ Champion (Production)                    │
│  ├─ raw_weather      │    │  ├─ Challenger (Staging)                     │
│  └─ features         │    │  └─ Experiment tracking                      │
│                      │    │                                              │
│  Indexes:            │    └──────────────────────────────────────────────┘
│  ├─ station+ts       │                     │
│  ├─ timestamp_utc    │                     ▼
│  └─ (lock retry x5) │    ┌──────────────────────────────────────────────┐
└──────────────────────┘    │          SERVING LAYER (FastAPI)              │
                            │                                              │
                            │  /forecast/{station_id}  (Pydantic validated)│
                            │  /forecast/heatmap       (GeoJSON)           │
                            │  /anomalies/latest       (Isolation Forest)  │
                            │  /health                                     │
                            │  /metrics                (Prometheus)         │
                            │                                              │
                            │  Features:                                   │
                            │  ├─ Redis ConnectionPool (shared)            │
                            │  ├─ FallbackCache (Redis outage resilience)  │
                            │  ├─ Champion/Challenger A/B routing           │
                            │  └─ run_in_threadpool (non-blocking)          │
                            └──────────────────────────────────────────────┘
                                             │
                                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       MONITORING                                          │
│                                                                          │
│  Prometheus + Grafana  │  Evidently (drift reports)  │  Telegram alerts  │
└──────────────────────────────────────────────────────────────────────────┘
```

## Component Deep-Dive

### 1. Data Ingestion (`airflow/dags/ingest_*.py`)

| DAG | Source | Schedule | Output |
|-----|--------|----------|--------|
| `ingest_aqi_dag` | AQICN API | `*/30 * * * *` | `raw_aqi` table |
| `ingest_weather_dag` | OpenWeatherMap | `0 * * * *` | `raw_weather` table |

**Resilience:** Both DAGs have retry=3 with 5-min backoff. API failures are logged but don't crash the pipeline.

### 2. Feature Engineering (`airflow/dags/feature_engineering_dag.py`)

- **Dependency:** `ExternalTaskSensor` waits for both ingestion DAGs
- **Output:** 40+ engineered features written to `features` table and Redis
- **Key Features:**
  - Temporal: hour_sin/cos, month_sin/cos, is_weekend
  - Lag: 1h, 3h, 6h, 12h, 24h, 48h, 168h (weekly)
  - Rolling stats: mean/std over multiple windows
  - Weather: wind direction decomposed (sin/cos), cumulative precipitation
  - Calendar: Nepal festivals (Tihar, Dashain, Indra Jatra), brick kiln season, monsoon

### 3. Training & Evaluation (`airflow/dags/train_evaluate_dag.py`)

**Ensemble Architecture:**
- **Prophet** (weight 0.4): Captures seasonality, trend, holidays
- **LSTM** (weight 0.6): Captures complex nonlinear temporal patterns
- Combined via weighted average

**Promotion Logic:**
1. Train on latest features → MLflow Staging
2. Evaluate against Production champion
3. If RMSE improves by >5%, promote to Production
4. Otherwise archive challenger

### 4. Drift Detection (`airflow/dags/drift_monitor_dag.py`)

| Metric | Threshold | Action |
|--------|-----------|--------|
| PSI (feature drift) | 0.25 | Emergency retrain |
| RMSE degradation | >15% | Emergency retrain |

**Festival-Aware Suppression:**
During Tihar (firecrackers), Dashain (bonfires), and brick kiln season (Jan-Apr), PSI thresholds are multiplied by 2x to prevent false retrain triggers on known, expected pollution events.

### 5. Serving Layer (`serving/`)

**FastAPI** on port 8000 with:
- Pydantic response models (PM2.5 clamped to [0, 500])
- Redis connection pooling (shared across endpoints)
- FallbackCache for Redis/MLflow outage resilience
- Champion/Challenger A/B routing via `X-Model-Version` header
- Prometheus metrics instrumentation

### 6. Storage (`storage/lake.py`)

**DuckDB** — embedded OLAP database, single-writer architecture.

**Lock Contention Handling:**
- Exponential backoff with jitter (max 5 retries, up to 30s delay)
- Catches `duckdb.IOException` and `TransactionException`
- All insert + query methods wrapped with retry logic

**Indexes:** Composite (station_id, timestamp_utc) on all tables for fast range scans.

## Data Quality Gates (`serving/data_quality.py`)

Pandera schemas enforce:
- PM2.5 physical bounds: [0, 999.9] µg/m³
- Lat/lon Nepal bounds: [26.3, 30.5] × [80.0, 88.2]
- NaN flood gate: reject batches with >30% null values
- Frozen sensor detection: std(pm25) must be > 0

## Infrastructure (Docker Compose)

15+ services on `nepalaqiops` bridge network:

| Service | Image | Purpose |
|---------|-------|---------|
| fastapi | Custom | API serving |
| airflow-* | apache/airflow:2.9.1 | Orchestration (webserver, scheduler, worker) |
| mlflow | Custom | Model registry + experiment tracking |
| redis | redis:7-alpine | Feature store (online) |
| postgres | postgres:16 | Airflow metadata |
| zookeeper | confluentinc/cp-zookeeper:7.6.0 | Kafka coordination |
| kafka | confluentinc/cp-kafka:7.6.0 | Event streaming |
| grafana | grafana/grafana:latest | Dashboards |
| prometheus | prom/prometheus:latest | Metrics collection |

All services have:
- Health checks with startup grace periods
- Memory limits (128MB-2GB depending on role)
- Restart policies (`unless-stopped`)
- Named volumes for persistence

## Security Considerations

- No secrets in code (`.env.example` template only)
- API rate limiting via FastAPI middleware
- Input validation on all endpoints (Pydantic + Query constraints)
- SQL injection prevention (parameterized queries in DuckDB)
- Docker network isolation (services communicate only via named network)

## Failure Modes & Recovery

| Failure | Impact | Automatic Recovery |
|---------|--------|--------------------|
| Redis down | Predictions use FallbackCache | FallbackCache serves stale (10min TTL) |
| DuckDB locked | Write contention | Exponential backoff retry (5 attempts) |
| MLflow unreachable | No model loading | Uses last cached model in memory |
| Ingestion API down | Stale data | Airflow retries 3x, then alerts |
| Drift detected | Model degradation | Auto-triggers retrain DAG |
| Festival pollution spike | False drift alarm | Festival-aware threshold relaxation |
