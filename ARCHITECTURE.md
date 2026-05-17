# Architecture

Detailed service interaction diagram for NepalAQI-Ops.

```mermaid
graph LR
    subgraph External
        OAQ[OpenAQ v3 API]
        AQICN[AQICN API]
        METEO[Open-Meteo API]
        TG[Telegram Bot API]
    end

    subgraph Docker Network: nepalaqiops
        subgraph Orchestration
            ZK[Zookeeper:2181]
            KFK[Kafka:9092]
            AIR_WEB[Airflow Webserver:8081]
            AIR_SCH[Airflow Scheduler]
        end

        subgraph Storage
            PG[(PostgreSQL:5432)]
            MINIO[MinIO:9000/9001]
            REDIS[Redis:6379]
            DUCK[(DuckDB Files)]
        end

        subgraph ML Platform
            MLF[MLflow:5000]
        end

        subgraph Application
            API[FastAPI:8080]
            DASH[Streamlit:8501]
        end

        subgraph Monitoring
            PROM[Prometheus:9090]
            GRAF[Grafana:3000]
            KEXP[Kafka Exporter:9308]
        end
    end

    %% Data flow
    OAQ -->|hourly poll| AIR_SCH
    AQICN -->|hourly poll| AIR_SCH
    METEO -->|hourly poll| AIR_SCH

    AIR_SCH -->|produce| KFK
    KFK -->|consume| DUCK

    AIR_SCH -->|compute features| REDIS
    AIR_SCH -->|train models| MLF
    MLF -->|store artifacts| MINIO
    MLF -->|store metadata| PG

    API -->|load model| MLF
    API -->|get features| REDIS
    API -->|metrics| PROM
    DASH -->|HTTP| API

    PROM -->|scrape| API
    PROM -->|scrape| KEXP
    KEXP -->|monitor| KFK
    GRAF -->|query| PROM

    AIR_SCH -->|drift alert| TG
    AIR_SCH -->|retrain notify| TG

    ZK -->|coordinate| KFK
    PG -->|airflow metadata| AIR_SCH
```

## Service Dependencies (Startup Order)

1. **Zookeeper** → Kafka
2. **PostgreSQL** → Airflow, MLflow
3. **MinIO** → MLflow (artifact store)
4. **Redis** → FastAPI (feature store)
5. **Kafka** → Ingestion DAGs
6. **MLflow** → Training DAGs, FastAPI
7. **Airflow** → All DAGs
8. **FastAPI** → Streamlit
9. **Prometheus** → Grafana

## Data Flow Stages

1. **Ingestion**: Airflow DAGs poll external APIs hourly → publish to Kafka topics
2. **Storage**: Kafka consumers write raw data to DuckDB (Parquet-backed)
3. **Feature Engineering**: Rolling stats, lag features, cyclical encoding, calendar flags, Kriging → Redis
4. **Training**: Weekly full retrain; daily drift check triggers conditional retrain
5. **Serving**: FastAPI loads champion model from MLflow, serves predictions from Redis features
6. **Monitoring**: Prometheus scrapes API metrics; Evidently computes PSI; Grafana visualizes; Telegram alerts
