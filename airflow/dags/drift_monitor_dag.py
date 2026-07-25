"""
Airflow DAG: Drift Monitor — daily PSI and RMSE drift detection.
Triggers emergency retrain if drift exceeds thresholds.
"""

import sys

sys.path.insert(0, "/opt/airflow")

import os
from datetime import datetime, timedelta

from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from airflow import DAG

default_args = {
    "owner": "nepalaqiops",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

PSI_RETRAIN_THRESHOLD = float(os.getenv("PSI_RETRAIN_THRESHOLD", "0.25"))
RMSE_DEGRADATION_THRESHOLD = float(os.getenv("RMSE_DEGRADATION_THRESHOLD", "0.15"))

# Festival-aware threshold multiplier: during known pollution events
# (Tihar firecrackers, Dashain bonfires, brick kiln season), drift is EXPECTED.
# Suppress false retrain triggers by relaxing thresholds.
FESTIVAL_THRESHOLD_MULTIPLIER = float(os.getenv("FESTIVAL_THRESHOLD_MULTIPLIER", "2.0"))

dag = DAG(
    "drift_monitor_dag",
    default_args=default_args,
    description="Daily drift detection — triggers retrain on degradation",
    schedule_interval="0 6 * * *",  # Daily at 6am
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["monitoring", "drift"],
)


def compute_psi(**kwargs):
    """Compute Population Stability Index on key features."""
    from monitoring.evidently_reports import compute_feature_psi
    from storage.lake import DataLake

    lake = DataLake()

    # Baseline: first week of training data
    baseline = lake.query("""
        SELECT pm25, pm10, temp_c, humidity_pct, wind_speed_kmh
        FROM features
        ORDER BY timestamp_utc
        LIMIT 168
    """)

    # Current: last 24 hours
    current = lake.query("""
        SELECT pm25, pm10, temp_c, humidity_pct, wind_speed_kmh
        FROM features
        WHERE timestamp_utc >= NOW() - INTERVAL 1 DAY
        ORDER BY timestamp_utc
    """)

    if baseline.empty or current.empty:
        kwargs["ti"].xcom_push(key="max_psi", value=0.0)
        return 0.0

    psi_scores = compute_feature_psi(baseline, current)
    max_psi = max(psi_scores.values()) if psi_scores else 0.0

    kwargs["ti"].xcom_push(key="psi_scores", value=psi_scores)
    kwargs["ti"].xcom_push(key="max_psi", value=max_psi)
    return max_psi


def compute_rmse_drift(**kwargs):
    """Compute live RMSE vs rolling baseline."""

    from training.evaluate import ModelEvaluator

    # In production, this would load recent predictions and compare to actuals
    # For now, simulate with stored metrics
    evaluator = ModelEvaluator()

    # Placeholder — in production, fetch from MLflow/Redis
    current_rmse = 15.0  # Would come from live prediction evaluation
    baseline_rmse = 12.0  # Would come from rolling historical average

    result = evaluator.check_degradation(current_rmse, baseline_rmse)
    kwargs["ti"].xcom_push(key="rmse_degraded", value=result["degraded"])
    kwargs["ti"].xcom_push(key="degradation_pct", value=result["degradation_pct"])
    return result["degraded"]


def check_drift_threshold(**kwargs):
    """
    Branch: trigger retrain or log OK status.
    
    Festival-aware: during Tihar, Dashain, or brick kiln season,
    drift is EXPECTED due to firecrackers/bonfires/kilns. We relax
    the PSI threshold by FESTIVAL_THRESHOLD_MULTIPLIER to avoid
    unnecessary retrains on known pollution events.
    """
    ti = kwargs["ti"]
    max_psi = ti.xcom_pull(key="max_psi", task_ids="compute_psi")
    rmse_degraded = ti.xcom_pull(key="rmse_degraded", task_ids="compute_rmse_drift")

    # Check if today falls within a known festival/event period
    execution_date = kwargs.get("execution_date", datetime.now())
    is_festival_period = _is_festival_or_event(execution_date)

    # Adjust threshold for festivals
    effective_psi_threshold = PSI_RETRAIN_THRESHOLD
    if is_festival_period:
        effective_psi_threshold = PSI_RETRAIN_THRESHOLD * FESTIVAL_THRESHOLD_MULTIPLIER
        import logging
        logging.getLogger(__name__).info(
            f"Festival period detected ({execution_date.strftime('%Y-%m-%d')}). "
            f"PSI threshold relaxed: {PSI_RETRAIN_THRESHOLD} → {effective_psi_threshold}"
        )

    psi_drift = max_psi > effective_psi_threshold
    should_retrain = psi_drift or rmse_degraded

    ti.xcom_push(key="should_retrain", value=should_retrain)
    ti.xcom_push(key="psi_drift", value=psi_drift)
    ti.xcom_push(key="is_festival_period", value=is_festival_period)
    ti.xcom_push(key="effective_threshold", value=effective_psi_threshold)

    if should_retrain:
        # Send drift alert
        from monitoring.telegram_alert import send_drift_alert
        psi_scores = ti.xcom_pull(key="psi_scores", task_ids="compute_psi") or {}
        worst_feature = max(psi_scores, key=psi_scores.get) if psi_scores else "unknown"
        send_drift_alert(
            feature_name=worst_feature,
            psi_score=max_psi,
            dag_run_id=kwargs["run_id"],
        )
        return "trigger_emergency_retrain"
    else:
        return "log_ok_status"


def _is_festival_or_event(date) -> bool:
    """
    Check if the given date falls within a known Nepal festival/pollution event.
    
    Major events that cause legitimate PM2.5 drift:
      - Tihar (Diwali): Oct-Nov, ~5 days of firecrackers
      - Dashain: Sep-Oct, bonfires and cooking smoke
      - Brick kiln season: Jan-Apr (industry fires)
      - Indra Jatra: Sep, city-wide celebrations
    
    Note: Nepal uses Bikram Sambat calendar; these are approximate Gregorian ranges.
    """
    month = date.month
    day = date.day

    # Brick kiln season: January through mid-April
    if month in (1, 2, 3) or (month == 4 and day <= 15):
        return True

    # Dashain: typically Oct 1-15 (approximate Gregorian)
    if month == 10 and day <= 15:
        return True

    # Tihar: typically late Oct to early Nov
    if (month == 10 and day >= 25) or (month == 11 and day <= 5):
        return True

    # Indra Jatra: typically September 1-10
    if month == 9 and day <= 10:
        return True

    return False


def log_ok_status(**kwargs):
    """Log that drift is within acceptable bounds."""
    ti = kwargs["ti"]
    max_psi = ti.xcom_pull(key="max_psi", task_ids="compute_psi")
    return f"Drift check OK. Max PSI: {max_psi:.4f} (threshold: {PSI_RETRAIN_THRESHOLD})"


# Task definitions
t_psi = PythonOperator(task_id="compute_psi", python_callable=compute_psi, dag=dag)
t_rmse = PythonOperator(task_id="compute_rmse_drift", python_callable=compute_rmse_drift, dag=dag)
t_branch = BranchPythonOperator(task_id="check_drift_threshold", python_callable=check_drift_threshold, dag=dag)

t_retrain = TriggerDagRunOperator(
    task_id="trigger_emergency_retrain",
    trigger_dag_id="train_evaluate_dag",
    dag=dag,
)

t_ok = PythonOperator(task_id="log_ok_status", python_callable=log_ok_status, dag=dag)

# DAG structure
[t_psi, t_rmse] >> t_branch >> [t_retrain, t_ok]
