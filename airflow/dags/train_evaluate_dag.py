"""
Airflow DAG: Train & Evaluate — weekly full model retraining with MLflow.
"""

import sys

sys.path.insert(0, "/opt/airflow")

from datetime import datetime, timedelta

from airflow.operators.python import BranchPythonOperator, PythonOperator

from airflow import DAG

default_args = {
    "owner": "nepalaqiops",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

dag = DAG(
    "train_evaluate_dag",
    default_args=default_args,
    description="Weekly model retraining with champion/challenger evaluation",
    schedule_interval="0 2 * * 0",  # Sunday 2am
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["training", "mlops"],
)


def validate_data_quality(**kwargs):
    """Fail fast if insufficient training data."""
    from storage.lake import DataLake

    lake = DataLake()
    count = lake.get_row_count("features")

    # Need at least 7 days × 24 hours × stations
    min_rows = 7 * 24
    if count < min_rows:
        raise ValueError(
            f"Insufficient training data: {count} rows (need ≥{min_rows}). "
            f"Wait for more data to accumulate."
        )

    kwargs["ti"].xcom_push(key="n_training_rows", value=count)
    return count


def train_prophet(**kwargs):
    """Train Prophet model with MLflow logging."""
    from storage.lake import DataLake
    from training.train import TrainingPipeline

    lake = DataLake()
    features = lake.get_features()

    if features.empty:
        raise ValueError("No features available for training")

    # Chronological 80/20 split
    split_idx = int(len(features) * 0.8)
    train_df = features.iloc[:split_idx]
    val_df = features.iloc[split_idx:]

    pipeline = TrainingPipeline()
    result = pipeline.train_prophet(train_df, val_df)

    kwargs["ti"].xcom_push(key="prophet_run_id", value=result["run_id"])
    kwargs["ti"].xcom_push(key="prophet_val_rmse", value=result["val_metrics"]["val_rmse"])
    return result["run_id"]


def train_lstm(**kwargs):
    """Train LSTM model with MLflow logging."""
    from storage.lake import DataLake
    from training.train import TrainingPipeline

    lake = DataLake()
    features = lake.get_features()

    if features.empty:
        raise ValueError("No features available for training")

    # Chronological 80/20 split
    split_idx = int(len(features) * 0.8)
    train_df = features.iloc[:split_idx]
    val_df = features.iloc[split_idx:]

    pipeline = TrainingPipeline()
    result = pipeline.train_lstm(train_df, val_df)

    kwargs["ti"].xcom_push(key="lstm_run_id", value=result["run_id"])
    kwargs["ti"].xcom_push(key="lstm_val_rmse", value=result["test_metrics"].get("test_rmse", float("inf")))
    return result["run_id"]


def evaluate_models(**kwargs):
    """Compare new models against current champion."""
    ti = kwargs["ti"]
    prophet_rmse = ti.xcom_pull(key="prophet_val_rmse", task_ids="train_prophet")
    lstm_rmse = ti.xcom_pull(key="lstm_val_rmse", task_ids="train_lstm")

    # Use the best of the two as the candidate
    if prophet_rmse < lstm_rmse:
        best_run_id = ti.xcom_pull(key="prophet_run_id", task_ids="train_prophet")
        best_rmse = prophet_rmse
        best_model = "prophet"
    else:
        best_run_id = ti.xcom_pull(key="lstm_run_id", task_ids="train_lstm")
        best_rmse = lstm_rmse
        best_model = "lstm"

    ti.xcom_push(key="best_run_id", value=best_run_id)
    ti.xcom_push(key="best_rmse", value=best_rmse)
    ti.xcom_push(key="best_model", value=best_model)
    return best_model


def promote_if_better(**kwargs):
    """Branch: promote new model or keep current champion."""
    from training.registry import ModelRegistry

    ti = kwargs["ti"]
    best_run_id = ti.xcom_pull(key="best_run_id", task_ids="evaluate_models")
    best_rmse = ti.xcom_pull(key="best_rmse", task_ids="evaluate_models")

    registry = ModelRegistry()
    version = registry.register_model(best_run_id)
    promoted = registry.auto_promote_if_better(version, best_rmse)

    ti.xcom_push(key="promoted", value=promoted)
    ti.xcom_push(key="model_version", value=version)

    return "promote_to_champion" if promoted else "keep_champion"


def promote_to_champion(**kwargs):
    """Log champion promotion."""
    ti = kwargs["ti"]
    version = ti.xcom_pull(key="model_version", task_ids="promote_if_better")
    return f"Model v{version} promoted to Production (Champion)"


def keep_champion(**kwargs):
    """Log keeping current champion."""
    return "Current champion retained — new model did not improve performance."


def generate_evidently_report(**kwargs):
    """Generate Evidently data quality and drift reports."""
    from monitoring.evidently_reports import generate_training_reports
    from storage.lake import DataLake

    lake = DataLake()
    features = lake.get_features()

    if not features.empty:
        generate_training_reports(features)


def notify_telegram(**kwargs):
    """Send Telegram notification about training completion."""
    from monitoring.telegram_alert import send_retrain_complete_alert

    ti = kwargs["ti"]
    best_model = ti.xcom_pull(key="best_model", task_ids="evaluate_models")
    best_rmse = ti.xcom_pull(key="best_rmse", task_ids="evaluate_models")
    promoted = ti.xcom_pull(key="promoted", task_ids="promote_if_better")
    version = ti.xcom_pull(key="model_version", task_ids="promote_if_better")

    send_retrain_complete_alert(
        model_name=best_model,
        version=version or "unknown",
        val_rmse=best_rmse or 0.0,
        prev_rmse=0.0,  # Will be fetched from registry in production
    )


# Task definitions
t_validate = PythonOperator(task_id="validate_data_quality", python_callable=validate_data_quality, dag=dag)
t_prophet = PythonOperator(task_id="train_prophet", python_callable=train_prophet, dag=dag)
t_lstm = PythonOperator(task_id="train_lstm", python_callable=train_lstm, dag=dag)
t_evaluate = PythonOperator(task_id="evaluate_models", python_callable=evaluate_models, dag=dag)
t_branch = BranchPythonOperator(task_id="promote_if_better", python_callable=promote_if_better, dag=dag)
t_promote = PythonOperator(task_id="promote_to_champion", python_callable=promote_to_champion, dag=dag)
t_keep = PythonOperator(task_id="keep_champion", python_callable=keep_champion, dag=dag)
t_evidently = PythonOperator(task_id="generate_evidently_report", python_callable=generate_evidently_report, dag=dag, trigger_rule="none_failed_min_one_success")
t_notify = PythonOperator(task_id="notify_telegram", python_callable=notify_telegram, dag=dag, trigger_rule="none_failed_min_one_success")

# DAG structure
t_validate >> [t_prophet, t_lstm] >> t_evaluate >> t_branch
t_branch >> [t_promote, t_keep]
[t_promote, t_keep] >> t_evidently >> t_notify
