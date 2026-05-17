"""
Training Pipeline — orchestrates model training with full MLflow experiment tracking.
"""

import os
import logging
import tempfile
from datetime import datetime, timezone
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


class TrainingPipeline:
    """Orchestrates model training with MLflow experiment tracking."""

    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self.run_date = datetime.now(timezone.utc).strftime("%Y%m%d")

    def train_prophet(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        experiment_name: str | None = None,
    ) -> dict[str, Any]:
        """Train Prophet model with full MLflow logging."""
        from models.prophet_model import ProphetAQModel

        exp_name = experiment_name or f"prophet_kathmandu_v{self.run_date}"
        mlflow.set_experiment(exp_name)

        model = ProphetAQModel()

        with mlflow.start_run(run_name=f"prophet_{self.run_date}") as run:
            # Log hyperparameters
            mlflow.log_params(model.get_hyperparameters())
            mlflow.log_params({
                "data_start": str(train_df["timestamp_utc"].min()),
                "data_end": str(train_df["timestamp_utc"].max()),
                "n_train_samples": len(train_df),
                "n_val_samples": len(val_df),
                "feature_list": ",".join(train_df.columns.tolist()[:20]),
            })

            # Train
            train_metrics = model.train(train_df)
            mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items() if isinstance(v, (int, float))})

            # Validate
            val_metrics = model.evaluate(val_df)
            mlflow.log_metrics(val_metrics)

            # Log model artifact
            with tempfile.TemporaryDirectory() as tmpdir:
                import pickle
                model_path = os.path.join(tmpdir, "prophet_model.pkl")
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
                mlflow.log_artifact(model_path)

            # Tag the run
            mlflow.set_tag("model_type", "prophet")
            mlflow.set_tag("stage", "staging")
            mlflow.set_tag("station", "kathmandu_all")

            result = {
                "run_id": run.info.run_id,
                "experiment_name": exp_name,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "model": model,
            }

        logger.info(f"Prophet training logged to MLflow run: {run.info.run_id}")
        return result

    def train_lstm(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        experiment_name: str | None = None,
    ) -> dict[str, Any]:
        """Train LSTM model with full MLflow logging."""
        from models.lstm_model import LSTMAQModel

        exp_name = experiment_name or f"lstm_kathmandu_v{self.run_date}"
        mlflow.set_experiment(exp_name)

        model = LSTMAQModel()

        with mlflow.start_run(run_name=f"lstm_{self.run_date}") as run:
            # Log hyperparameters
            mlflow.log_params(model.get_hyperparameters())
            mlflow.log_params({
                "data_start": str(train_df["timestamp_utc"].min()),
                "data_end": str(train_df["timestamp_utc"].max()),
                "n_train_samples": len(train_df),
                "n_val_samples": len(val_df),
            })

            # Train
            train_metrics = model.train(train_df, val_df)
            mlflow.log_metrics({k: v for k, v in train_metrics.items() if isinstance(v, (int, float))})

            # Evaluate
            test_metrics = model.evaluate(val_df)
            mlflow.log_metrics(test_metrics)

            # Log learning curves
            if model.history:
                for epoch, loss in enumerate(model.history.history["loss"]):
                    mlflow.log_metric("epoch_loss", loss, step=epoch)
                if "val_loss" in model.history.history:
                    for epoch, val_loss in enumerate(model.history.history["val_loss"]):
                        mlflow.log_metric("epoch_val_loss", val_loss, step=epoch)

            # Log model weights
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, "lstm_model.keras")
                model.model.save(model_path)
                mlflow.log_artifact(model_path)

                # Log scaler
                import pickle
                scaler_path = os.path.join(tmpdir, "scaler.pkl")
                with open(scaler_path, "wb") as f:
                    pickle.dump(model.scaler, f)
                mlflow.log_artifact(scaler_path)

            # Tag
            mlflow.set_tag("model_type", "lstm")
            mlflow.set_tag("stage", "staging")
            mlflow.set_tag("station", "kathmandu_all")

            result = {
                "run_id": run.info.run_id,
                "experiment_name": exp_name,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "model": model,
            }

        logger.info(f"LSTM training logged to MLflow run: {run.info.run_id}")
        return result

    def train_isolation_forest(
        self,
        train_df: pd.DataFrame,
        experiment_name: str = "anomaly_detection",
    ) -> dict[str, Any]:
        """Train Isolation Forest with MLflow logging."""
        from models.isolation_forest import AnomalyDetector

        mlflow.set_experiment(experiment_name)
        model = AnomalyDetector()

        with mlflow.start_run(run_name=f"iforest_{self.run_date}") as run:
            mlflow.log_params(model.get_hyperparameters())
            metrics = model.train(train_df)
            mlflow.log_metrics(metrics)

            # Save model
            with tempfile.TemporaryDirectory() as tmpdir:
                import pickle
                model_path = os.path.join(tmpdir, "isolation_forest.pkl")
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
                mlflow.log_artifact(model_path)

            mlflow.set_tag("model_type", "isolation_forest")

            result = {
                "run_id": run.info.run_id,
                "metrics": metrics,
                "model": model,
            }

        return result
