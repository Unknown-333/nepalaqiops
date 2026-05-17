# =============================================================================
# NepalAQI-Ops Makefile
# =============================================================================
.PHONY: help up down build ingest train test lint clean logs

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

up: ## Start all services
	docker compose up --build -d

down: ## Stop all services
	docker compose down

build: ## Build all Docker images
	docker compose build

ingest: ## Run one-shot data ingestion
	docker compose exec airflow-scheduler airflow dags trigger ingest_aqi_dag
	docker compose exec airflow-scheduler airflow dags trigger ingest_weather_dag

train: ## Trigger model training DAG
	docker compose exec airflow-scheduler airflow dags trigger train_evaluate_dag

drift: ## Run drift monitoring check
	docker compose exec airflow-scheduler airflow dags trigger drift_monitor_dag

features: ## Run feature engineering
	docker compose exec airflow-scheduler airflow dags trigger feature_engineering_dag

test: ## Run pytest with coverage
	docker compose exec fastapi pytest --cov=. --cov-report=html --cov-report=term-missing

lint: ## Run ruff linter
	ruff check . --fix

typecheck: ## Run mypy type checker
	mypy serving/ models/ ingestion/

logs: ## Tail all service logs
	docker compose logs -f --tail=100

logs-api: ## Tail FastAPI logs
	docker compose logs -f fastapi

logs-airflow: ## Tail Airflow scheduler logs
	docker compose logs -f airflow-scheduler

status: ## Show service status
	docker compose ps

kafka-topics: ## List Kafka topics
	docker compose exec kafka kafka-topics --bootstrap-server localhost:9092 --list

clean: ## Remove all volumes and containers
	docker compose down -v --remove-orphans

restart-api: ## Restart FastAPI service
	docker compose restart fastapi

shell-api: ## Open shell in FastAPI container
	docker compose exec fastapi /bin/bash

shell-airflow: ## Open shell in Airflow scheduler
	docker compose exec airflow-scheduler /bin/bash
