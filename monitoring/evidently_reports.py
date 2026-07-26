"""
Evidently AI Monitoring — generates drift, data quality, and model performance reports.
"""

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_feature_psi(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    n_bins: int = 10,
) -> dict[str, float]:
    """
    Compute Population Stability Index (PSI) per feature.
    PSI > 0.2 = warning, PSI > 0.25 = significant drift.
    """
    psi_scores = {}

    for col in baseline.columns:
        if col not in current.columns:
            continue

        base_vals = baseline[col].dropna().values
        curr_vals = current[col].dropna().values

        if len(base_vals) == 0 or len(curr_vals) == 0:
            psi_scores[col] = 0.0
            continue

        # Bin edges from baseline quantiles (deciles) so each bin holds a
        # comparable share of the baseline and out-of-range current values are
        # absorbed by the open-ended tails. Fall back to equal-width bins for
        # near-constant / binary features where quantiles collapse.
        if np.unique(base_vals).size <= 2:
            bin_edges = np.linspace(
                min(base_vals.min(), curr_vals.min()),
                max(base_vals.max(), curr_vals.max()),
                n_bins + 1,
            )
        else:
            bin_edges = np.unique(np.quantile(base_vals, np.linspace(0, 1, n_bins + 1)))
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf

        base_hist, _ = np.histogram(base_vals, bins=bin_edges)
        curr_hist, _ = np.histogram(curr_vals, bins=bin_edges)

        # Normalize to proportions, flooring empty/sparse bins to a small value
        # so the log ratio stays finite without exploding on near-empty bins.
        base_prop = np.clip(base_hist / base_hist.sum(), 1e-3, None)
        curr_prop = np.clip(curr_hist / curr_hist.sum(), 1e-3, None)

        # PSI formula
        psi = np.sum((curr_prop - base_prop) * np.log(curr_prop / base_prop))
        psi_scores[col] = float(psi)

    return psi_scores


def generate_training_reports(features_df: pd.DataFrame) -> dict[str, str]:
    """
    Generate Evidently AI reports after training.
    Returns paths to generated HTML reports.
    """
    try:
        from evidently.metric_preset import (
            DataDriftPreset,
            DataQualityPreset,
        )
        from evidently.report import Report
    except ImportError:
        logger.warning("Evidently not installed. Skipping report generation.")
        return {}

    reports = {}

    # Split data for drift comparison (first week vs current)
    if len(features_df) > 168:
        reference = features_df.iloc[:168]
        current = features_df.iloc[-168:]
    else:
        reference = features_df.iloc[: len(features_df) // 2]
        current = features_df.iloc[len(features_df) // 2:]

    # Columns to monitor for drift
    drift_columns = ["pm25", "pm10", "temp_c", "humidity_pct", "wind_speed_kmh"]
    available_columns = [c for c in drift_columns if c in features_df.columns]

    # 1. Data Quality Report
    try:
        quality_report = Report(metrics=[DataQualityPreset()])
        quality_report.run(reference_data=reference, current_data=current)

        quality_path = "/tmp/evidently_data_quality.html"
        quality_report.save_html(quality_path)
        reports["data_quality"] = quality_path
        logger.info(f"Data quality report saved: {quality_path}")
    except Exception as e:
        logger.error(f"Data quality report failed: {e}")

    # 2. Data Drift Report
    try:
        if available_columns:
            ref_subset = reference[available_columns]
            curr_subset = current[available_columns]

            drift_report = Report(metrics=[DataDriftPreset()])
            drift_report.run(reference_data=ref_subset, current_data=curr_subset)

            drift_path = "/tmp/evidently_data_drift.html"
            drift_report.save_html(drift_path)
            reports["data_drift"] = drift_path
            logger.info(f"Data drift report saved: {drift_path}")
    except Exception as e:
        logger.error(f"Data drift report failed: {e}")

    return reports


def upload_reports_to_minio(report_paths: dict[str, str]) -> dict[str, str]:
    """Upload generated reports to MinIO bucket."""
    try:
        from datetime import datetime, timezone

        import boto3

        s3 = boto3.client(
            "s3",
            endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123"),
        )

        bucket = "evidently-reports"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        urls = {}

        for report_name, local_path in report_paths.items():
            key = f"{timestamp}/{report_name}.html"
            s3.upload_file(local_path, bucket, key, ExtraArgs={"ContentType": "text/html"})
            url = f"http://minio:9000/{bucket}/{key}"
            urls[report_name] = url
            logger.info(f"Uploaded {report_name} to {url}")

        return urls

    except Exception as e:
        logger.error(f"Failed to upload reports to MinIO: {e}")
        return {}
