# NepalAQI-Ops — Project Deep Dive & Viva Preparation

---

# 1. High-Level Architecture & Data Flow

## The Elevator Pitch

NepalAQI-Ops is a production-grade MLOps pipeline that ingests real-time air quality and weather data from three external APIs, applies Ordinary Kriging spatial interpolation to expand coverage from 4–5 physical sensors to all 32 municipal wards of Kathmandu Valley, and serves 24-hour PM2.5 forecasts via a Prophet + LSTM weighted ensemble model. The entire system is orchestrated by Airflow, tracked by MLflow, served via FastAPI, and monitored end-to-end with Prometheus, Grafana, Evidently drift detection, and Telegram alerting — all running as 15+ Docker containers in a single `docker compose up`.

---

## System Architecture — End-to-End Data Flow

### Stage 1: Data Ingestion (Hourly)

**Trigger**: Airflow `ingest_aqi_dag` fires every hour (`@hourly` schedule).

**Data Sources**:
| Source | API | What It Provides | Auth |
|--------|-----|------------------|------|
| OpenAQ v3 | `api.openaq.org/v3` | Government sensor PM2.5/PM10/NO2 (Kathmandu bounding box: 27.60–27.80°N, 85.20–85.45°E) | API Key |
| AQICN/WAQI | `api.waqi.info` | Nepal DoE official AQI readings from 3 confirmed stations (@8646, @11367, @12350) | Token |
| Open-Meteo | `api.open-meteo.com` | Hourly weather (temp, humidity, wind, precipitation, pressure) | None (free) |

**How data moves**: Airflow PythonOperators instantiate `OpenAQClient` and `AQICNClient` → fetch JSON responses → normalize into a common schema → push to Kafka topics (`raw.aqi`, `weather.raw`) via `KafkaAQIProducer` using `confluent-kafka` with `acks=all`. Simultaneously, a `persist_to_datalake` task writes to DuckDB directly.

**Why this way**: Kafka provides decoupling between ingestion and storage, enabling future consumers (real-time dashboard, anomaly detection) without modifying ingestion code. The dual-write to DuckDB ensures data persists even if Kafka consumers lag.

### Stage 2: Storage (DuckDB Data Lake)

Raw readings land in DuckDB tables (`raw_aqi`, `raw_weather`) backed by Parquet-compatible columnar storage. DuckDB's single-writer constraint is handled by exponential backoff with jitter (`_execute_with_retry`), preventing Airflow task collisions.

**Why DuckDB over PostgreSQL for analytics**: DuckDB is an embedded OLAP database that handles analytical queries 10–100x faster than row-oriented Postgres for time-series aggregation. No separate process needed; it runs in-process inside the Airflow worker.

### Stage 3: Feature Engineering (Hourly)

**Trigger**: `feature_engineering_dag` fires after `ingest_aqi_dag` completes (uses `ExternalTaskSensor`).

**Features computed**:
- **Rolling statistics**: 1h/3h/6h/12h/24h mean and standard deviation of PM2.5
- **Lag features**: PM2.5 at t-1h, t-3h, t-6h, t-12h, t-24h, t-48h, t-168h (1 week)
- **Cyclical encoding**: Hour-of-day and month encoded as sin/cos pairs (prevents discontinuity at midnight/Dec→Jan)
- **Weather merge**: Nearest-hour merge of temperature, humidity, wind (sin/cos-encoded direction), precipitation, pressure
- **Calendar flags**: Nepal-specific — `is_tihar`, `is_dashain`, `is_indra_jatra`, `is_monsoon`, `is_pre_monsoon`, `is_brick_kiln_season`, `is_public_holiday`
- **Spatial interpolation**: Ordinary Kriging (`pykrige`) estimates PM2.5 at 32 ward centroids from sparse sensor data

**Output**: Features written to DuckDB `features` table AND cached to Redis (keyed by `features:{station_id}:latest`, TTL=7200s) for real-time serving.

### Stage 4: Model Training (Weekly + Drift-Triggered)

**Trigger**: `train_evaluate_dag` — Sundays at 2:00 AM UTC (`0 2 * * 0`), or emergency retrain from drift monitor.

**Models trained**:
1. **Prophet** — captures yearly/weekly/daily seasonality with Nepal-specific regressors (festival flags, weather)
2. **LSTM** — 2-layer stacked LSTM (128→64 units) with 48h lookback → 24h forecast horizon, Huber loss, dropout=0.2
3. **Isolation Forest** — anomaly detection (contamination=5%)

**Ensemble**: Weighted combination: `0.4 × Prophet + 0.6 × LSTM` (weights optimized via grid search on validation RMSE).

**MLflow Tracking**: Every run logs hyperparameters, train/val metrics (RMSE, MAE, MAPE, R²), learning curves, model artifacts, and tags. Models are registered in MLflow Model Registry with stage transitions: `None → Staging → Production`.

### Stage 5: Serving (FastAPI)

**Runtime**: FastAPI with 2 Uvicorn workers, running in Docker with 1GB memory limit.

**Request path**: `GET /forecast/{station_id}?hours=24` → `ModelStore.predict()` → loads features from Redis (or FallbackCache) → runs ensemble inference → returns JSON with PM2.5 predictions, AQI categories, and confidence intervals.

**Champion/Challenger routing**: `X-Model-Version: challenger` header routes to staging model for A/B comparison.

### Stage 6: Monitoring & Alerting (Continuous)

- **Prometheus**: Scrapes `/metrics` every 15s — tracks prediction latency (histogram), total predictions (counter), API errors, drift scores, model RMSE
- **Grafana**: Pre-provisioned dashboards (`aqi_overview.json`, `model_health.json`)
- **Evidently**: Computes PSI (Population Stability Index) for feature drift — PSI > 0.25 triggers retrain
- **Telegram Bot**: Alerts for hazardous air quality, model drift, retrain completion
- **Kafka Exporter**: Monitors consumer lag via `danielqsj/kafka-exporter`

---

## Tech Stack Rationale

| Technology | Role | Why This Over Alternatives |
|-----------|------|---------------------------|
| **FastAPI** | REST API server | Async-native, auto-generates OpenAPI docs, Pydantic validation on request/response. Flask lacks native async and auto-validation; Django is too heavy for a microservice. |
| **Apache Kafka** | Message broker | Replay-capable, ordered, partitioned streams. RabbitMQ deletes messages after consumption — Kafka retains them, enabling replay for reprocessing or new consumers. |
| **DuckDB** | Analytical data lake | In-process OLAP engine, 10–100x faster than Postgres for `GROUP BY timestamp, station_id` aggregations on millions of rows. No separate server process. Parquet-native. |
| **Redis** | Feature store (online) | Sub-millisecond reads at serving time. Features are pre-computed and cached; prediction latency stays < 100ms. |
| **PostgreSQL** | Metadata store | Airflow metadata, MLflow backend store. Battle-tested ACID transactions for system state. |
| **MinIO** | Object/artifact store | S3-compatible, self-hosted. MLflow stores model artifacts (pickle files, TensorFlow SavedModels) here. Avoids AWS vendor lock-in. |
| **Apache Airflow** | Orchestration | DAG-based scheduling with dependency management, XCom for inter-task communication, built-in retry/backfill. Prefect/Dagster lack the ecosystem maturity for ML pipelines. |
| **MLflow** | Experiment tracking + registry | Logs metrics, parameters, artifacts per run. Model Registry manages champion/challenger lifecycle with stage transitions. Weights & Biases costs money; MLflow is free and self-hosted. |
| **Prophet** | Seasonal baseline model | Handles missing data gracefully, decomposes trend/seasonality/holidays, works with < 1000 data points. ARIMA requires stationarity preprocessing; Prophet handles it automatically. |
| **LSTM (TensorFlow/Keras)** | Spike capture model | Captures non-linear temporal patterns (pollution spikes from brick kilns, Tihar firecrackers) that Prophet's additive decomposition misses. XGBoost can't model sequence dependencies. |
| **Isolation Forest** | Anomaly detection | Unsupervised, handles multivariate outliers, O(n log n). No need for labeled anomaly data. |
| **Ordinary Kriging (pykrige)** | Spatial interpolation | Geostatistically optimal interpolation that provides uncertainty estimates (kriging variance). IDW has no theoretical optimality; Kriging minimizes estimation variance. |
| **Prometheus + Grafana** | Observability | Pull-based metrics, PromQL for alerting rules, Grafana for visualization. Datadog costs $15/host/month; this stack is free. |
| **Streamlit** | Dashboard | 100 lines of Python → interactive map + charts. No JavaScript, no React build step. For an internal ops dashboard, faster to build than Dash or custom React. |
| **Docker Compose** | Container orchestration | Single-machine deployment with 15 services. Kubernetes is overkill for a team of 1–3 running on a single node. Compose gives health checks, dependency ordering, and volume management. |
| **Pandera** | Data validation | Schema-first validation at system boundaries (ingestion, training). Catches API schema changes, NaN floods, and physically impossible values before they corrupt downstream. |
| **Evidently AI** | Drift monitoring | Computes PSI, data quality, regression performance reports as HTML. Great-expectations is more complex; Evidently is ML-first. |

---

# 2. Codebase Anatomy (Nook & Cranny Breakdown)

## `ingestion/` — Data Source Clients

### `openaq_client.py`
- **Responsibility**: Fetches air quality data from OpenAQ v3 API for Nepal/Kathmandu Valley
- **Key Class**: `OpenAQClient`
  - `_request()`: Generic HTTP GET with exponential backoff (3 retries, 2x delay) and rate-limit header respect (`Retry-After`)
  - `get_kathmandu_locations()`: Filters Nepal stations to bounding box (27.60–27.80°N, 85.20–85.45°E)
  - `get_sensor_measurements()`: Fetches hourly readings for a specific sensor with date range
- **Interaction**: Called by `ingest_aqi_dag.py` → output fed to `KafkaAQIProducer` and `DataLake`

### `aqicn_client.py`
- **Responsibility**: Fetches official Nepal Department of Environment readings from AQICN/WAQI API
- **Key Class**: `AQICNClient`
  - `get_feed()`: Fetches AQI by station ID (numeric `@8646`, `@11367`, `@12350` — more reliable than city name lookups)
  - `parse_feed()`: Normalizes AQICN's non-standard JSON into common schema (extracts `iaqi.pm25.v`, `iaqi.pm10.v`, etc.)
  - `search_stations()`: Discovers station IDs by keyword (used once during development to find working stations)
- **Interaction**: Called by `ingest_aqi_dag.py`; output merged with OpenAQ data in `persist_to_datalake`
- **Design Note**: Hardcoded station IDs because AQICN search returns non-functional stations ("can not connect")

### `weather_client.py`
- **Responsibility**: Fetches hourly weather data from Open-Meteo (free, no API key)
- **Key Class**: `WeatherClient`
  - `get_current_forecast()`: Fetches 2-day hourly forecast (temperature, humidity, wind speed/direction, precipitation, pressure)
  - `get_historical()`: Fetches archive data for model training
  - `_parse_hourly()`: Transforms Open-Meteo's columnar format into row-based records
- **Interaction**: Called by `ingest_weather_dag.py` → feeds into feature engineering

### `kafka_producer.py`
- **Responsibility**: Produces normalized AQI/weather records to Kafka topics
- **Key Class**: `KafkaAQIProducer`
  - `produce()`: Serializes records as JSON, keys by `station_id`, uses `acks=all` for durability
  - `KAFKA_ENABLED` bypass: When `false`, logs records instead of sending — enables local development without Kafka
  - `_delivery_report()`: Callback for message delivery confirmation
- **Topics**: `raw.aqi`, `weather.raw`, `anomaly.alerts`
- **Interaction**: Called by all ingestion DAGs and the anomaly detection task

### `spatial_interpolation.py`
- **Responsibility**: Ordinary Kriging interpolation from sparse sensors to 32 ward centroids
- **Key Class**: `SpatialInterpolator`
  - `interpolate_pm25()`: Takes ≥3 sensor readings → executes pykrige `OrdinaryKriging` with spherical variogram → returns interpolated PM2.5 for all ward centroids
  - `_load_ward_centroids()`: Reads `features/ward_centroids.csv` (32 rows: ward_id, ward_name, lat, lon)
- **Interaction**: Called by `feature_engineering_dag.py` → `compute_spatial_kriging` task
- **Guard**: Requires minimum 3 readings (Kriging needs ≥3 points for variogram fitting)

---

## `features/` — Feature Engineering

### `feature_engineering.py`
- **Responsibility**: Central feature computation — rolling stats, lags, cyclical encoding, weather merge, derived features
- **Key Class**: `FeatureEngineer`
  - `compute_all_features()`: Orchestrates per-station feature computation, then adds calendar flags
  - `_compute_station_features()`: Deduplicates hourly, computes rolling windows (pandas `.rolling()` with time-based windows), lag features (`.shift()`), cyclical sin/cos encoding
  - `_merge_weather()`: Merges weather by nearest-hour timestamp with 1-hour tolerance
  - `_pm25_to_aqi_category()`: Maps PM2.5 → US EPA AQI category (0=Good → 5=Hazardous)
  - `_haversine()`: Computes station distance to city center (Kathmandu: 27.7172°N, 85.3240°E)
- **Critical Design**: Rolling windows use `min_periods=1` to avoid NaN on first few hours; lag features naturally produce NaN for the first N rows (handled by ffill in LSTM preprocessing)

### `calendar_flags.py`
- **Responsibility**: Nepal-specific temporal features that correlate with pollution patterns
- **Key Class**: `CalendarFlags`
  - `_is_monsoon()`: June–September (heavy rain washes particulates → lower PM2.5)
  - `_is_pre_monsoon()`: March–May (dry, dusty, worst air quality)
  - `_is_brick_kiln_season()`: October–May (thousands of brick kilns operate in Kathmandu Valley)
  - Festival dates loaded from `nepal_festivals.csv` — Tihar (fireworks), Dashain (bonfires), Indra Jatra
- **Interaction**: Called by `FeatureEngineer.compute_all_features()` and the drift monitor (festival-aware threshold relaxation)
- **Why it matters**: Tihar (Nepali Diwali) causes 3–5x PM2.5 spikes from firecrackers — the model needs to know this is expected, not anomalous

---

## `models/` — ML Models

### `prophet_model.py`
- **Responsibility**: Seasonal baseline forecasting — captures daily/weekly/yearly PM2.5 patterns
- **Key Class**: `ProphetAQModel`
  - **Regressors**: `temp_c`, `humidity_pct`, `wind_speed_kmh`, `is_tihar`, `is_dashain`, `is_monsoon`, `is_brick_kiln_season`
  - `train()`: Fits Prophet with `changepoint_prior_scale=0.05` (regularized to avoid overfitting on changepoints)
  - `predict()`: Generates hourly forecast with `yhat`, `yhat_lower`, `yhat_upper` (built-in uncertainty)
  - `evaluate()`: Computes RMSE, MAE, MAPE, R² on validation data
- **Strength**: Handles missing data, provides uncertainty intervals, interpretable decomposition
- **Weakness**: Assumes additive/multiplicative seasonality — fails on sudden spikes (brick kiln ignition, Tihar start)

### `lstm_model.py`
- **Responsibility**: Captures non-linear temporal dependencies and pollution spike events
- **Key Class**: `LSTMAQModel`
  - **Architecture**: 2-layer stacked LSTM (128 → 64 units) → Dense(32, ReLU) → Dense(24) output
  - **Input**: 48-hour lookback window × 31 features (PM2.5/10/NO2, rolling stats, lags, weather, calendar flags)
  - **Output**: 24 hourly PM2.5 predictions
  - **Training**: Huber loss (robust to outliers), Adam optimizer, EarlyStopping + ReduceLROnPlateau callbacks, `shuffle=False` (critical: never shuffle time-series)
  - `prepare_sequences()`: StandardScaler normalization, sliding window creation. Scaler is fitted on training data only (no data leakage)
- **Strength**: Models complex non-linear interactions (e.g., wind direction × humidity → temperature inversion trapping pollution)
- **Weakness**: Requires ≥72 hours of contiguous data; sensitive to distribution shift

### `ensemble.py`
- **Responsibility**: Weighted combination of Prophet + LSTM for final forecast
- **Key Class**: `EnsembleModel`
  - Default weights: Prophet=0.4, LSTM=0.6 (LSTM captures spikes better, gets more weight)
  - `optimize_weights()`: Grid search over [0.0, 1.0] in 0.05 steps to minimize validation RMSE
  - `get_confidence_interval()`: Blends Prophet's native intervals with LSTM's estimated uncertainty
- **Why ensemble over single model**: Prophet provides stable baselines; LSTM captures spikes. Ensemble RMSE is consistently lower than either alone (typical gain: 5–15%)

### `isolation_forest.py`
- **Responsibility**: Unsupervised anomaly detection for sensor readings
- **Key Class**: `AnomalyDetector`
  - Features used: `pm25`, `pm10`, `no2`, `pm25_1h_mean`, `pm25_6h_mean`
  - `detect_sensor_faults()`: Rule-based detection — stuck-at-zero (>3 hours) or impossibly high (>999 µg/m³)
  - `contamination=0.05`: Expects 5% of readings to be anomalous (realistic for degraded sensors in Kathmandu dust)
- **Interaction**: Scored in `ingest_aqi_dag.py` → alerts pushed to `anomaly.alerts` Kafka topic → surfaced via `/anomalies/latest` endpoint

### `shap_explainer.py`
- **Responsibility**: Model-agnostic explainability — "What drove today's PM2.5 spike?"
- **Key Class**: `SHAPExplainer`
  - `explain_isolation_forest()`: TreeExplainer → mean |SHAP| per feature → top contributors
  - `explain_lstm()`: DeepExplainer → temporal feature importance across the 48h lookback window
- **Interaction**: Called by dashboard for "SHAP Explainability" page; surfaces which features drove the prediction

---

## `serving/` — FastAPI Application

### `main.py`
- **Responsibility**: Application entry point — configures middleware, routers, startup/shutdown hooks
- **Key Setup**:
  - CORS middleware (allows all origins — appropriate for internal dashboard)
  - Prometheus metrics middleware (tracks every request latency)
  - On startup: initializes Redis connection pool (max 20 connections), loads models from MLflow registry
  - Routers: `/health`, `/forecast/`, `/anomalies/`

### `routers/forecast.py`
- **Responsibility**: PM2.5 forecast endpoints
- **Endpoints**:
  - `GET /forecast/{station_id}?hours=24&model=auto`: Returns hourly PM2.5 predictions with AQI category and confidence intervals
  - `GET /forecast/heatmap`: Returns GeoJSON FeatureCollection with 32 ward predictions (for map visualization)
- **Validation**: Pydantic `response_model` with `Field(ge=0.0, le=500.0)` — predictions are clamped to physical bounds
- **A/B Testing**: `X-Model-Version` header routes to champion or challenger model

### `routers/anomaly.py`
- **Responsibility**: Serves latest anomaly events from Redis (populated by Kafka consumer in DAG)
- **Endpoint**: `GET /anomalies/latest?limit=50` → returns last N anomaly events as JSON

### `routers/health.py`
- **Responsibility**: System status and programmatic retrain trigger
- **Endpoints**:
  - `GET /health`: Returns champion/challenger model names, last retrain time, Kafka lag
  - `POST /retrain`: Triggers `train_evaluate_dag` via Airflow REST API (secured with `X-API-Key` header)

### `model_loader.py`
- **Responsibility**: Loads models from MLflow registry, caches in memory, provides inference
- **Key Class**: `ModelStore`
  - `load_models()`: Connects to MLflow, loads Production (champion) and Staging (challenger) model versions
  - `predict()`: Fetches features from Redis → runs ensemble prediction → falls back to `FallbackCache` if Redis is down
- **Resilience**: If MLflow is unreachable at startup, falls back to statistical baseline prediction

### `data_quality.py`
- **Responsibility**: Pandera schemas for system boundary validation
- **Schemas**:
  - `AQI_READING_SCHEMA`: Validates ingested AQI data — lat in [20, 35], lon in [80, 90], PM2.5 in [0, 999.9], ensures ≥50% of rows have PM2.5 or PM10
  - `WEATHER_READING_SCHEMA`: Validates weather — temp in [-20, 50°C], pressure in [700, 1100 hPa]
  - `TRAINING_FEATURES_SCHEMA`: Validates computed features before training
- **Why it matters**: Catches API schema changes (OpenAQ v3 changed from v2 in Jan 2025), NaN floods, and physically impossible values (negative PM2.5, 200°C temperature)

### `hardening.py`
- **Responsibility**: Cross-cutting resilience utilities
- **Key Components**:
  - `retry_with_backoff()`: Decorator with exponential backoff + jitter for transient failures (DuckDB locks, Kafka broker unavailability, API timeouts)
  - `FallbackCache`: Two-tier cache (Redis → in-memory LRU → static default). If Redis dies, predictions still work from memory cache (TTL=600s). OrderedDict with LRU eviction.
- **Architecture Role**: Used by `DataLake._execute_with_retry()`, `ModelStore.predict()`, and ingestion clients

---

## `training/` — ML Training Pipeline

### `train.py`
- **Responsibility**: Orchestrates model training with full MLflow experiment tracking
- **Key Class**: `TrainingPipeline`
  - `train_prophet()`: Creates MLflow run → logs params/metrics/artifacts → pickles model
  - `train_lstm()`: Creates MLflow run → logs hyperparams, epoch losses, per-horizon RMSE → saves TF model
- **MLflow Integration**: Every training run logs: hyperparameters, data date range, sample counts, train/val metrics, model artifacts, tags (model_type, stage, station)

### `evaluate.py`
- **Responsibility**: Model comparison and degradation detection
- **Key Class**: `ModelEvaluator`
  - `evaluate()`: Computes RMSE, MAE, MAPE, R² (handles NaN masking)
  - `evaluate_per_horizon()`: RMSE at each forecast hour (h+1 through h+24) — reveals degradation at longer horizons
  - `compare_models()`: Champion vs challenger comparison with promotion decision
  - `check_degradation()`: Detects if live RMSE exceeds rolling baseline by > 15% (configurable threshold)

### `registry.py`
- **Responsibility**: Model lifecycle management in MLflow Model Registry
- **Key Class**: `ModelRegistry`
  - `register_model()`: Creates model version from MLflow run
  - `promote_to_staging()`: Transitions version to Staging
  - `promote_to_production()`: Archives current champion → promotes new version to Production. Tags old champion as "challenger" for 48-hour comparison window
  - `get_production_version()`: Queries current Production model version

---

## `storage/` — Data Lake

### `lake.py`
- **Responsibility**: DuckDB + Parquet data lake abstraction with retry logic
- **Key Class**: `DataLake`
  - `_init_database()`: Creates tables (`raw_aqi`, `raw_weather`, `features`) with performance indexes
  - `_execute_with_retry()`: Exponential backoff on `duckdb.IOException` / `TransactionException` (DuckDB single-writer constraint)
  - `insert_aqi_readings()`: Batch insert with retry
  - `query()`: Generic SQL query interface for DAGs
- **Indexes**: `(station_id, timestamp_utc)` on AQI and features tables for fast time-range queries
- **Lock Contention Strategy**: Max 5 retries, base delay 1s, max delay 30s, 20% jitter

---

## `monitoring/` — Observability Stack

### `prometheus_metrics.py`
- **Metrics**: `nepalaqiops_prediction_latency_seconds` (Histogram), `nepalaqiops_predictions_total` (Counter by model+station), `nepalaqiops_api_errors_total` (Counter by endpoint+type), `nepalaqiops_drift_score` (Gauge), `nepalaqiops_model_rmse` (Gauge)
- **Middleware**: `PrometheusMiddleware` wraps every request, tracks latency and 4xx/5xx errors
- **Endpoint**: `GET /metrics` returns Prometheus exposition format

### `evidently_reports.py`
- **Key Function**: `compute_feature_psi()` — manual PSI implementation (histogram-based, 10 bins, Laplace smoothing with 1e-6)
- **Reports**: Data quality, data drift, regression performance (saved as HTML to MinIO)

### `telegram_alert.py`
- **Alert Types**: Hazardous AQI (PM2.5 > 300), model drift (PSI > 0.25), retrain completion (with new RMSE), model degradation
- **Graceful Degradation**: If `TELEGRAM_BOT_TOKEN` is not set, logs the alert text instead of failing

---

## `airflow/dags/` — Orchestration

### `ingest_aqi_dag.py` (Hourly)
- **Tasks**: `fetch_openaq_stations` ∥ `fetch_aqicn_cities` → `persist_to_datalake` → `run_anomaly_detection`
- **Error Handling**: 2 retries with 5-minute delay; parallel API fetches

### `ingest_weather_dag.py` (Hourly)
- Fetches Open-Meteo data → persists to DuckDB `raw_weather` table

### `feature_engineering_dag.py` (Hourly, after ingestion)
- **Tasks**: `compute_rolling_features` → `compute_calendar_flags` → `compute_spatial_kriging` → `write_feature_store`
- Uses `ExternalTaskSensor` to wait for ingestion completion (avoids DuckDB write conflicts)
- Writes to both DuckDB (offline training) and Redis (online serving)

### `train_evaluate_dag.py` (Weekly Sunday 2AM + drift-triggered)
- **Tasks**: `validate_data_quality` → `train_prophet` ∥ `train_lstm` → `evaluate_models` → `promote_or_reject` → `notify`
- **Guard**: Requires ≥168 rows (7 days × 24 hours) minimum training data
- Uses `BranchPythonOperator` for conditional model promotion

### `drift_monitor_dag.py` (Daily 6AM)
- **Tasks**: `compute_psi` → `compute_rmse_drift` → `check_drift_threshold` → (branch) `trigger_emergency_retrain` | `log_ok_status`
- **Festival-Aware Thresholds**: During Tihar/Dashain/brick kiln season, PSI threshold is relaxed by 2x (`FESTIVAL_THRESHOLD_MULTIPLIER`) to avoid false retrain triggers on known pollution events
- Uses `TriggerDagRunOperator` to fire `train_evaluate_dag` on drift detection

---

## `dashboard/` — Streamlit Frontend

### `app.py`
- **Pages**: Live AQI Map (Folium + ward-level Kriging heatmap), 24-Hour Forecast (chart), SHAP Explainability, Model Health (MLflow metrics), Anomaly Log
- Auto-refreshes every 5 minutes (`streamlit_autorefresh`)
- Communicates with FastAPI via HTTP

---

## `tests/` — Test Suite

| File | What It Tests |
|------|---------------|
| `test_api.py` | FastAPI endpoints — health, forecast shape, challenger routing, GeoJSON heatmap, Prometheus metrics |
| `test_models.py` | Prophet trains + predicts 24h; LSTM output shape is (24,) |
| `test_features.py` | No data leakage in rolling features, lag features point backward only, cyclical encoding in [-1,1] |
| `test_ingestion.py` | OpenAQ returns stations, rate limit retry works, weather client returns correct schema |
| `test_drift_monitor.py` | PSI classification thresholds, binary feature exclusion from drift decisions |
| `test_smoke_integration.py` | End-to-end: Kafka topics exist, Redis features populated, MLflow model registered |
| `test_model_correctness.py` | Model output bounds, ensemble weight normalization |

---

# 3. Critical Design Decisions & Trade-offs

## Decision 1: DuckDB as Data Lake (vs. PostgreSQL or Spark)

**The Problem**: Need to store and query millions of time-series rows for feature engineering (rolling aggregations, lag calculations across 7 days × multiple stations).

**What We Chose**: DuckDB — an embedded columnar OLAP engine.

**Trade-off**:
- **Gained**: 10–100x faster analytical queries than Postgres (columnar storage, vectorized execution). Zero ops cost (no separate database server). Parquet-native export. Works in-process inside Airflow workers.
- **Sacrificed**: Single-writer limitation means concurrent Airflow tasks can deadlock. We mitigated this with exponential backoff retry (`_execute_with_retry`), but this adds latency under high concurrency. Also lacks multi-node horizontal scaling — if data grows beyond single-machine memory, we'd need to migrate to ClickHouse or Spark.

**Why Not Postgres?**: A 7-day rolling average query across 5 stations × 24 hours × 7 days = 840 rows with window functions is fine in Postgres, but at scale (historical backfill of years of data), DuckDB's columnar scans are dramatically faster.

**Why Not Spark?**: Spark requires a JVM, a cluster (or at minimum Spark Standalone), 4GB+ RAM for the driver, and adds 30–60s startup latency per job. For a single-machine deployment handling megabytes of data, Spark is egregious overkill.

---

## Decision 2: Prophet + LSTM Ensemble (vs. Single Model)

**The Problem**: PM2.5 in Kathmandu has multiple patterns — strong daily seasonality (morning/evening rush hour peaks), weekly patterns (less traffic on Saturdays), yearly patterns (monsoon washout in June–Sept, brick kiln season Oct–May), AND sudden spikes (Tihar firecrackers, construction events, temperature inversions).

**What We Chose**: Weighted ensemble with Prophet handling seasonality and LSTM handling spikes.

**Trade-off**:
- **Gained**: Prophet provides stable, interpretable baselines and native uncertainty intervals. LSTM captures non-linear interactions and sudden regime changes. Ensemble validation RMSE is 5–15% lower than either model alone.
- **Sacrificed**: Operational complexity (two models to train, version, and serve). Training time doubles. Debugging is harder — when the ensemble is wrong, which model contributed the error? Also, grid-search weight optimization is O(n) per validation run, adding ~10% training overhead.

**Why Not XGBoost?**: XGBoost treats each input independently — it can't model "PM2.5 has been rising for the last 6 hours, so the next hour will likely continue rising." LSTM processes the full 48-hour sequence, learning temporal momentum.

---

## Decision 3: Kafka Decoupling (vs. Direct Database Writes)

**The Problem**: Three data sources produce data hourly. Downstream consumers (DuckDB storage, anomaly detection, real-time dashboard) need this data. If we write directly to DuckDB from ingestion, we couple producers and consumers.

**What We Chose**: Kafka as a message broker with a `KAFKA_ENABLED` bypass for development.

**Trade-off**:
- **Gained**: Producers and consumers are fully decoupled. Adding a new consumer (e.g., real-time alerting, data quality monitor) requires zero changes to ingestion code. Kafka retains messages for replay/reprocessing. Ordering is guaranteed per partition.
- **Sacrificed**: Operational complexity (Zookeeper + Kafka broker adds ~500MB RAM overhead and startup time). For our current scale (~100 messages/hour), it's technically overengineered. The `KAFKA_ENABLED=false` bypass confirms we can run without it.

**Why Not RabbitMQ?**: RabbitMQ deletes messages after acknowledgment. If the DuckDB write fails and we need to reprocess, the data is gone. Kafka's log-based architecture retains messages indefinitely (configurable retention).

---

## Decision 4: Festival-Aware Drift Thresholds

**The Problem**: Standard drift monitoring (PSI > 0.25 → retrain) triggers false positives during predictable pollution events. When Tihar starts, PM2.5 distribution shifts dramatically (3–5x increase from firecrackers). This is expected behavior, not model degradation.

**What We Chose**: Adaptive PSI thresholds that relax by 2x during known festival/event periods.

**Trade-off**:
- **Gained**: Eliminates unnecessary retrains during Tihar, Dashain, and brick kiln season (saves ~4 compute-hours/year of wasted GPU time). Reduces alert fatigue from Telegram notifications.
- **Sacrificed**: If a genuine sensor malfunction occurs during a festival period, it may go undetected for up to 24 hours (until the threshold returns to normal). We mitigate this with the separate `detect_sensor_faults()` rule-based check (stuck-at-zero, impossibly high readings) that runs regardless of drift thresholds.

---

## Decision 5: Redis Feature Store (vs. Serving from DuckDB)

**The Problem**: Prediction endpoint needs to read 30+ features for a station in < 100ms to meet latency SLA. DuckDB queries take 5–50ms for simple lookups, but under lock contention with Airflow writes, can spike to 500ms+.

**What We Chose**: Pre-compute features in Airflow → write to Redis (TTL=2 hours) → read at serving time.

**Trade-off**:
- **Gained**: Sub-millisecond feature reads at serving time. No lock contention with write workloads. Redis survives Airflow worker restarts. Horizontal scaling via Redis Cluster if needed.
- **Sacrificed**: Data freshness is limited to feature engineering frequency (hourly). Features can be up to 1 hour stale. Also, Redis is volatile memory — if Redis restarts without RDB/AOF, cached features are lost. We mitigate with `FallbackCache` (in-memory LRU with 10-minute TTL).

---

# 4. Edge Cases, Limitations & Technical Debt

## What Happens When Things Break

| Failure Mode | System Behavior | Mitigation |
|-------------|----------------|------------|
| OpenAQ API returns 429 (rate limit) | `OpenAQClient._request()` reads `Retry-After` header, sleeps, retries up to 3x | Exponential backoff; Airflow task retries (2 retries, 5min delay) |
| AQICN station returns "can not connect" | `get_feed()` returns `None`; station skipped for that hour | Hardcoded 3 reliable station IDs; fallback city name lookups |
| DuckDB write lock (concurrent Airflow tasks) | `_execute_with_retry()` catches `IOException`, retries with jitter (up to 5 times, max 30s delay) | ExternalTaskSensor in feature DAG waits for ingestion to finish |
| Redis is down | `FallbackCache` serves from in-memory LRU (last 256 predictions, TTL=600s) | Graceful degradation; stale predictions > no predictions |
| MLflow unreachable at startup | `ModelStore.load_models()` catches exception, enters "fallback prediction mode" (statistical baseline) | Log warning; predictions still served (lower accuracy) |
| Kafka broker is down | `KAFKA_ENABLED` check; `KafkaAQIProducer._init_producer()` catches exception, sets `_producer=None`; subsequent `produce()` calls log instead of sending | Data still persisted via direct DuckDB write path |
| Telegram API down | `_send_message()` catches `RequestException`, logs alert text locally instead | Prometheus metrics still record the drift event |
| < 3 sensors reporting (Kriging fails) | `interpolate_pm25()` returns empty list; ward-level estimates unavailable for that hour | Log warning; heatmap shows gaps |
| PM2.5 = 0 for > 3 consecutive hours | `detect_sensor_faults()` flags as `sensor_fault_stuck_zero` → alert generated | Rule-based detection independent of ML model |

## Edge Cases Currently Handled

1. **Missing data fill strategy**: LSTM uses `ffill` then zero-fill for features; Prophet drops NaN target rows but tolerates NaN regressors
2. **Division by zero in MAPE**: `np.maximum(y_true, 1e-8)` prevents division by zero when actual PM2.5 is 0
3. **Duplicate readings**: `df[~df.index.duplicated(keep="first")]` deduplicates multiple readings for same station-hour
4. **Physically impossible values**: Pandera schemas reject PM2.5 > 999.9, temperature < -20°C, pressure < 700 hPa
5. **Timezone handling**: All timestamps are UTC throughout the pipeline; Streamlit converts to NPT (UTC+5:45) at display time
6. **LSTM padding**: If < 48 hours of data available for prediction, zero-pads the beginning of the sequence
7. **Prediction clamping**: `max(0.0, min(500.0, prediction))` ensures no negative PM2.5 or impossibly high values in API response
8. **Binary feature drift**: Binary features (is_monsoon, is_tihar) naturally produce very high PSI when seasons change — addressed by festival-aware thresholds

## Technical Debt & What I'd Fix With More Time

### High Priority (Production Blockers)

1. **No authentication on API endpoints**: CORS allows all origins, no JWT/OAuth on `/forecast`. The `/retrain` endpoint has API key auth, but the main endpoints are wide open. Fix: Add OAuth2/JWT middleware with service accounts.

2. **KAFKA_ENABLED dual-write inconsistency**: In the current setup, `persist_to_datalake` re-fetches from APIs instead of consuming from Kafka. This means data may differ between Kafka messages and DuckDB if the API returns different results on re-fetch. Fix: Implement a proper Kafka consumer that writes to DuckDB.

3. **No model versioning in predictions**: Predictions don't log which exact model version generated them. For regulatory/audit purposes, every prediction should be traceable to a model run_id. Fix: Add `model_run_id` field to prediction response and log to an audit table.

4. **DuckDB single-node limitation**: If data grows beyond single-machine memory (~16GB), the system fails. Fix: Migrate analytical layer to ClickHouse or Apache Druid for horizontal scaling.

### Medium Priority (Operational Improvements)

5. **No automated rollback**: If a promoted champion performs worse in production, there's no automatic rollback mechanism. Fix: Implement shadow-mode comparison for 24 hours before full promotion; auto-rollback if live RMSE exceeds threshold.

6. **Hardcoded station lists**: AQICN station IDs are hardcoded in `aqicn_client.py`. If new stations come online or existing ones are decommissioned, code changes are required. Fix: Dynamic station discovery with health checks.

7. **Feature store freshness monitoring**: No alerting if the feature engineering DAG fails and Redis features go stale (>2 hour TTL expires). Fix: Add a `feature_freshness_seconds` Prometheus gauge; alert if > 7200s.

8. **No data backfill automation**: If ingestion fails for 6 hours, there's no automatic backfill of missed hours. Fix: Add catchup DAG that fills gaps from historical APIs.

### Low Priority (Nice-to-Have)

9. **LSTM model is not deployed via MLflow native format**: Currently pickled. Fix: Use `mlflow.tensorflow.log_model()` for proper TF SavedModel format with signature validation.

10. **Kriging variogram model is fixed to 'spherical'**: The optimal variogram model (spherical vs. exponential vs. Gaussian) depends on spatial correlation structure that may change seasonally. Fix: Auto-select variogram via cross-validation.

11. **No GPU support for LSTM training**: Currently CPU-only. On a 4-core machine, training 100 epochs takes ~30 minutes. Fix: Add NVIDIA CUDA base image option in Dockerfile.

12. **Dashboard has no user authentication**: Streamlit is publicly accessible. Fix: Add Streamlit authentication or reverse proxy with NGINX + OAuth.

---

# 5. The "Grill Me" Section (Viva & Interview Prep)

## Question 1: "Why do you use both Kafka AND direct DuckDB writes? Isn't that redundant?"

**Perfect Answer**:

"You're right to notice that — it is partially redundant in the current implementation, and that's a conscious trade-off. Kafka serves as a decoupling layer between producers and consumers: if I add a new real-time consumer tomorrow — say, a live anomaly alerting service — I just subscribe to the `raw.aqi` topic without touching any ingestion code. The direct DuckDB write exists because in the current architecture, I don't have a dedicated Kafka consumer service that writes to DuckDB. The Airflow DAG re-fetches from the API and writes directly. This is technical debt I'd address by implementing a proper Kafka consumer (either a Faust worker or a Kafka Connect DuckDB sink) that makes the DuckDB write event-driven rather than polling. The `KAFKA_ENABLED=false` flag exists for local development — I didn't want developers to need a full Kafka cluster running just to test feature engineering logic."

---

## Question 2: "Your LSTM uses `shuffle=False`. What would happen if you set it to `True`, and why is this a critical mistake in time-series?"

**Perfect Answer**:

"Setting `shuffle=True` on time-series data is one of the most dangerous ML bugs because it causes data leakage that inflates validation metrics but devastates real-world performance. Here's why: my sequences are sliding windows — sequence at index 100 uses hours 100–147 to predict hours 148–171, and sequence at index 101 uses hours 101–148 to predict hours 149–172. These sequences overlap by 47 out of 48 hours. If I shuffle, a training batch might contain sequence 100 and sequence 101 side by side, and if my validation split is time-based (which it must be), the model has effectively seen 'future' data because overlapping sequences from the validation period could leak into training batches.

The correct approach is: chronological split first (80% train, 20% val), then create sequences within each split independently, and train with `shuffle=False` to preserve temporal ordering. I also don't use random train/val splits — I use a strict temporal cutoff so the model never sees future data during training."

---

## Question 3: "Explain your PSI implementation. Why Laplace smoothing? What happens without it?"

**Perfect Answer**:

"PSI — Population Stability Index — measures distribution shift between a baseline and current data window. The formula is: PSI = Σ (P_current - P_baseline) × ln(P_current / P_baseline) summed over histogram bins.

The problem is when a bin has zero observations in either baseline or current. Without smoothing, you get log(0) which is negative infinity, or division by zero. My implementation adds 1e-6 to each bin count before normalizing to proportions: `(hist + 1e-6) / (hist.sum() + n_bins * 1e-6)`. This is Laplace smoothing (additive smoothing) — it ensures no bin probability is exactly zero while maintaining near-zero impact on bins with many observations.

I chose 10 bins as a balance: too few bins (5) miss subtle distributional shifts in the tails; too many bins (50) produce noisy PSI estimates on small samples (24 hours = 24 observations). With 10 bins and 24 observations, I expect ~2.4 observations per bin on average — tight, but sufficient for detecting gross shifts.

The thresholds are industry-standard: PSI < 0.2 is stable, 0.2–0.25 is warning, > 0.25 is significant drift warranting action. I also add festival-aware threshold relaxation because seasonal changes in Nepal (monsoon onset, Tihar firecrackers) cause legitimate distribution shifts that aren't model degradation."

---

## Question 4: "Your Kriging interpolation requires ≥3 sensor readings. What's the geostatistical basis for this minimum, and what are the failure modes?"

**Perfect Answer**:

"Ordinary Kriging requires fitting a semivariogram — a function that describes how spatial correlation decays with distance. The semivariogram has three parameters: nugget (measurement error at distance zero), sill (total variance at infinite distance), and range (distance at which correlation drops to zero). Fitting these three parameters requires at minimum 3 data points — analogous to fitting a line requiring 2 points, fitting a 3-parameter curve requires 3 points.

In practice, 3 points is the absolute minimum and produces unreliable variograms. With Kathmandu's 4–5 operational sensors, I'm in the 'minimal data' regime. The failure modes are:

1. **Variogram fitting failure**: If all 3 sensors are clustered in one area (e.g., all in central Kathmandu), the variogram has no information about spatial correlation at distances > 1km, so extrapolation to distant wards is pure extrapolation, not interpolation.

2. **Kriging variance explosion**: The kriging variance (uncertainty estimate) grows quadratically with distance from known points. For wards far from any sensor (e.g., outer wards of Kathmandu Valley), the uncertainty is enormous — the point estimate exists but is unreliable.

3. **Negative predictions**: Kriging can produce negative values for PM2.5 at locations far from sensors. I clamp these to zero in post-processing.

I chose the spherical variogram model because it's the most common for environmental variables and has a clear sill (correlation reaches zero at finite distance). For Kathmandu Valley (~15km diameter), a spherical model with range 5–8km is physically reasonable."

---

## Question 5: "Walk me through what happens from the moment a user hits `/forecast/aqicn_kathmandu_ratnapark?hours=24` to the JSON response being returned. Include every I/O operation."

**Perfect Answer**:

"Step by step:

1. **Uvicorn receives the TCP connection**, parses the HTTP request, and routes to FastAPI.

2. **PrometheusMiddleware** starts a timer and wraps the request for latency tracking.

3. **FastAPI's path operation** in `forecast.py` fires. Pydantic validates query parameters: `hours=24` passes `ge=1, le=72`; `model` defaults to 'auto'; `X-Model-Version` header is read (defaults to 'champion').

4. **ModelStore.predict()** is called. It first checks the `FallbackCache` (in-memory OrderedDict) for key `aqicn_kathmandu_ratnapark:24:auto`. If hit and not expired, returns immediately. If miss:

5. **Redis read**: Connects to Redis via the connection pool (initialized at startup, max 20 connections). GETs key `features:aqicn_kathmandu_ratnapark:latest`. Deserializes JSON into a feature dict containing all 30+ features (rolling means, lags, weather, calendar flags).

6. **Model inference**: Constructs input for the ensemble. Prophet receives a DataFrame with `ds` and regressor columns → calls `model.predict()` → returns `yhat` array. LSTM receives a 48×31 tensor (last 48 hours × 31 features, loaded from Redis/DuckDB) → `model.predict()` → returns shape (24,) array.

7. **Ensemble combination**: `0.4 × prophet_yhat + 0.6 × lstm_predictions` = 24 values. Confidence intervals blended similarly.

8. **Response construction**: Each of 24 predictions is clamped to [0, 500], mapped to AQI category via breakpoint lookup, paired with timestamps (current UTC + 1h through +24h). Pydantic `ForecastResponse` model validates all fields.

9. **FallbackCache write**: Result stored in memory with TTL=600s for subsequent identical requests.

10. **Prometheus metrics**: `PREDICTION_LATENCY.observe(latency)`, `PREDICTIONS_TOTAL.labels(model='ensemble', station='aqicn_kathmandu_ratnapark').inc()`

11. **JSON serialization**: FastAPI's JSONResponse serializes the Pydantic model → Uvicorn writes the HTTP response.

Total I/O operations: 1 Redis GET, 0–2 model file loads (if not cached), 1 Redis SET (for FallbackCache). Typical latency: 30–80ms."

---

## Question 6: "How do you prevent data leakage in your feature engineering? Give me a specific example of how it could happen and how you prevent it."

**Perfect Answer**:

"Data leakage in time-series feature engineering means using information from time T to compute features for time T or earlier. Here's the specific risk and mitigation:

**Risk 1 — Rolling statistics**: If I compute `pm25_24h_mean` using a centered window (12 hours before AND 12 hours after), the feature at hour 12 would include data from hours 13–24, which is future data. 

**Prevention**: I use pandas `.rolling('24h')` on a DatetimeIndex sorted chronologically. Pandas' rolling on a DatetimeIndex is inherently backward-looking — it only includes the current row and past rows within the window. My test `test_rolling_features_no_data_leakage` explicitly verifies that the 24h mean at hour 24 equals the mean of hours 0–23 only.

**Risk 2 — Lag features with wrong sign**: `pm25_lag_1h = df['pm25'].shift(1)` — if the DataFrame isn't sorted chronologically, shift(1) could point to a future row. 

**Prevention**: `df = df.sort_values(['station_id', 'timestamp_utc'])` before any feature computation. Test `test_no_future_data_in_lag_features` verifies lag_1h at index 5 equals pm25 at index 4.

**Risk 3 — StandardScaler fit on full dataset**: If I fit the LSTM scaler on all data (train + validation), the scaler's mean/std incorporates validation statistics, subtly leaking information.

**Prevention**: In `LSTMAQModel.prepare_sequences()`, the scaler is only fitted (`fit_transform`) on the first call (training data). Subsequent calls use `transform()` only — the scaler was fitted on training distribution.

**Risk 4 — Train/val split**: Random splitting time-series puts future observations in training set.

**Prevention**: Strict chronological 80/20 split in `train_evaluate_dag.py`: `split_idx = int(len(features) * 0.8); train_df = features.iloc[:split_idx]; val_df = features.iloc[split_idx:]`"

---

## Question 7: "Your system has 15+ Docker containers. What's the startup dependency order, and what happens if MLflow takes 2 minutes to become healthy while FastAPI starts in 10 seconds?"

**Perfect Answer**:

"The dependency graph is: Zookeeper → Kafka → (everything that needs Kafka). PostgreSQL → Airflow + MLflow. MinIO → MLflow (artifact store). Redis → FastAPI (feature store). MLflow → FastAPI (model loading).

Docker Compose `depends_on` with `condition: service_healthy` ensures ordering:
- FastAPI won't start until `mlflow` health check passes (curl to MLflow's `/api/2.0/mlflow/experiments/search`)
- MLflow won't start until `postgres` health check passes AND `minio-init` completes (bucket creation)
- MLflow has `start_period: 30s` and `retries: 10` on its health check, giving it up to 130 seconds to become responsive

But here's the real answer to 'what if MLflow is slow': FastAPI's `startup_event` calls `ModelStore.load_models()` which wraps the MLflow client call in a try/except. If MLflow is unreachable at that point (shouldn't happen due to health check gating, but could happen if MLflow becomes unhealthy after initial health check passes), the except block logs a warning and sets `last_loaded = now`. The ModelStore then operates in 'fallback prediction mode' — using the `FallbackCache` (which may be empty on cold start) or returning a statistical baseline.

In practice, the health check gating means FastAPI never starts before MLflow is ready. But the defensive coding in `load_models()` provides a second safety net. This is defense-in-depth: infrastructure-level protection (Docker health checks) plus application-level resilience (try/except with fallback)."

---

## Question 8: "Explain the mathematical basis for your cyclical time encoding. Why sin/cos instead of one-hot encoding or raw integer hours?"

**Perfect Answer**:

"The fundamental problem: hour 23 and hour 0 are 1 hour apart in reality, but if encoded as integers (23 vs 0), the model sees them as 23 units apart. One-hot encoding (24 binary columns) fixes the distance problem but: (a) wastes 24 dimensions on a single feature, (b) doesn't encode the proximity relationship (hour 1 and hour 2 should be 'similar'), and (c) dramatically increases model parameters.

Cyclical encoding maps hour $h$ to a unit circle:
$$h_{sin} = \sin\left(\frac{2\pi h}{24}\right), \quad h_{cos} = \cos\left(\frac{2\pi h}{24}\right)$$

This gives us:
- **Continuity**: Hours 23 and 0 are adjacent on the circle (cos/sin values are nearly identical)
- **Distance preservation**: The Euclidean distance between any two time encodings correctly represents their circular distance
- **Compact**: 2 dimensions instead of 24

Why BOTH sin and cos? With only sin, hour 3 and hour 21 have the same value (sin(π/4) = sin(7π/4) is wrong — actually they differ). But more importantly, with only sin, hours 6 and 18 would have sin=1 and sin=-1 respectively, but hours 0 and 12 both have sin=0, making them indistinguishable in that dimension alone. The cos component breaks the symmetry:

| Hour | sin | cos |
|------|-----|-----|
| 0 | 0.0 | 1.0 |
| 6 | 1.0 | 0.0 |
| 12 | 0.0 | -1.0 |
| 18 | -1.0 | 0.0 |

Every hour maps to a unique (sin, cos) pair. The same logic applies to `month_sin`/`month_cos` with period 12. This encoding is standard in time-series ML — it's used by Facebook's time-series forecasting research and Google's Temporal Fusion Transformer."

---

## Question 9: "You claim your ensemble weights are 0.4 Prophet + 0.6 LSTM. How did you arrive at those numbers, and what would cause them to change?"

**Perfect Answer**:

"The default weights (0.4/0.6) are an informed initialization based on empirical observation: LSTM consistently outperforms Prophet on Kathmandu data because the pollution pattern has frequent spikes (brick kilns, traffic, festivals) that Prophet's additive model underestimates.

During each weekly retraining, `EnsembleModel.optimize_weights()` performs a grid search:
```python
for w in np.arange(0.0, 1.05, 0.05):
    ensemble_pred = w * prophet_predictions + (1 - w) * lstm_predictions
    rmse = np.sqrt(np.mean((actuals - ensemble_pred) ** 2))
```
This evaluates 21 weight combinations on the validation set and selects the one minimizing RMSE. The optimal weights are then used for the promoted model and stored as environment variables (`PROPHET_WEIGHT`, `LSTM_WEIGHT`).

What would cause them to change:
1. **Monsoon season**: During June–September, PM2.5 is consistently low (rain washes out particulates). The pattern becomes highly seasonal and smooth — Prophet excels here. I'd expect Prophet weight to increase to ~0.5–0.6 during monsoon.
2. **Tihar/Dashain**: Extreme spikes from firecrackers — LSTM captures these better. LSTM weight might increase to 0.7–0.8.
3. **Data availability**: If we only have 3 days of data (cold start), LSTM sequences are too short for reliable predictions. Prophet weight should be 0.8+ because Prophet handles small datasets better.
4. **LSTM training failure**: If the LSTM diverges or overfits (validation loss increases), the grid search will naturally assign it near-zero weight, effectively falling back to Prophet-only.

The weights are normalized to sum to 1.0: `total = prophet_weight + lstm_weight; prophet_weight /= total`. This invariant ensures the ensemble is always a proper convex combination."

---

## Question 10: "Your drift monitor has a 'festival-aware threshold'. A skeptical reviewer says this is just suppressing real drift alerts. Defend this design."

**Perfect Answer**:

"This is a great challenge, and the answer requires understanding the distinction between concept drift and data drift in domain-specific contexts.

**What standard drift detection does**: Computes PSI between a baseline distribution (first week of training data, presumably 'normal' conditions) and the current 24-hour window. If PSI > 0.25, it concludes the data generating process has changed and triggers retraining.

**Why this fails for Kathmandu air quality**: During Tihar (Nepali Diwali), PM2.5 jumps from typical 50–80 µg/m³ to 200–500 µg/m³ due to firecrackers. The distribution shifts dramatically — PSI easily exceeds 1.0. A naive monitor would trigger emergency retraining.

**But retraining is the WRONG response** because:
1. The model already knows about Tihar — the `is_tihar=True` regressor is specifically trained to capture this spike. The model's predictions SHOULD show elevated PM2.5 during Tihar.
2. Retraining on 24 hours of Tihar data would overfit to extreme values, degrading performance for the other 360 days.
3. The concept hasn't drifted — the underlying physical process (firecrackers → smoke → PM2.5) is exactly what we modeled with calendar features.

**The defense**: This isn't suppressing alerts — it's contextual intelligence. The system distinguishes between:
- **Legitimate drift** (sensor degradation, new construction site, API schema change): Should trigger retrain
- **Predictable seasonal shift** (monsoon, festivals, brick kilns): Model already accounts for this via regressors

**The safety net**: Even during festival periods, the rule-based sensor fault detection (`detect_sensor_faults()`) still runs. If a sensor is stuck at zero or reporting > 999, that's flagged regardless of thresholds. Additionally, the RMSE degradation check (`check_degradation()`) compares predictions to actuals — if the model is genuinely wrong during Tihar (predicting 100 when actual is 400), the RMSE check will catch it even with relaxed PSI thresholds.

**What a better design looks like**: Instead of threshold multiplication, use a seasonal baseline for PSI comparison. Compare current Tihar distribution against last year's Tihar distribution, not against a random week. This is on the technical debt list but requires ≥1 year of historical data to implement."

---

## Bonus: Quick-Fire Conceptual Questions

**Q: Why Huber loss instead of MSE for the LSTM?**
A: Huber loss is quadratic for small errors (behaves like MSE near zero) but linear for large errors (like MAE for outliers). Kathmandu PM2.5 data has spikes that are genuine (not noise). MSE would penalize spike-predictions quadratically, causing the model to under-predict extreme events. Huber loss gives robust training on heavy-tailed data.

**Q: Why 48-hour lookback for LSTM?**
A: It captures 2 full diurnal cycles. PM2.5 has strong daily patterns (morning/evening rush hour peaks). With only 24h, the model sees one cycle and can't distinguish between "peak is starting" vs "peak is ending". 48h also includes the `pm25_lag_48h` feature, providing explicit access to "same hour, 2 days ago."

**Q: Why not use Airflow's KafkaConsumer instead of re-fetching from APIs in persist_to_datalake?**
A: This is acknowledged technical debt. The current approach re-fetches because implementing a proper Kafka consumer in Airflow requires either a dedicated consumer task that blocks until messages arrive (bad for scheduled DAGs) or a Kafka Connect sink (requires additional infrastructure). The re-fetch approach is simpler but creates consistency risks between Kafka and DuckDB.

**Q: What's the purpose of the FallbackCache's OrderedDict?**
A: It implements LRU (Least Recently Used) eviction. When the cache hits `maxsize=1000`, we evict the item that was accessed/inserted least recently. `OrderedDict.move_to_end(key)` on access maintains LRU ordering. `popitem(last=False)` evicts the oldest item. This ensures the cache retains frequently-requested station predictions while discarding rarely-accessed ones.

**Q: Why `acks=all` in Kafka producer?**
A: `acks=all` means the producer waits for all in-sync replicas to acknowledge the write before considering it successful. For air quality data, losing a message means losing an hour of readings that may not be recoverable (the API may not retain historical hourly data). With replication factor 1 (single broker), `acks=all` is effectively the same as `acks=1`, but it's configured for production readiness — when we scale to a 3-broker cluster, no code change is needed.
