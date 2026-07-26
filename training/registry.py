"""
Model Registry — manages champion/challenger model lifecycle.
Promotes models through: None → Staging → Production (Champion).
"""

import logging
import os
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = "nepalaqiops-pm25-forecast"


class ModelRegistry:
    """Manages model promotion and champion/challenger routing."""

    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self.client = MlflowClient()

    def register_model(self, run_id: str, model_name: str = MODEL_NAME) -> str:
        """Register a model version from an MLflow run."""
        model_uri = f"runs:/{run_id}/model"

        # Ensure registered model exists
        try:
            self.client.get_registered_model(model_name)
        except mlflow.exceptions.MlflowException:
            self.client.create_registered_model(
                model_name,
                description="NepalAQI-Ops PM2.5 forecast model (ensemble)"
            )

        # Create a new version
        mv = self.client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id,
        )

        logger.info(f"Registered model version {mv.version} for {model_name}")
        return mv.version

    def promote_to_staging(self, version: str, model_name: str = MODEL_NAME):
        """Move model version to Staging."""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging",
        )
        logger.info(f"Model {model_name} v{version} promoted to Staging")

    def promote_to_production(self, version: str, model_name: str = MODEL_NAME):
        """
        Promote model to Production (Champion).
        Previous Production model becomes the Challenger (kept for 48h comparison).
        """
        # Archive current production versions
        current_prod = self.get_production_version(model_name)
        if current_prod:
            self.client.transition_model_version_stage(
                name=model_name,
                version=current_prod,
                stage="Archived",
            )
            # Tag old champion as challenger
            self.client.set_model_version_tag(
                name=model_name,
                version=current_prod,
                key="role",
                value="challenger",
            )

        # Promote new version
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
        )
        self.client.set_model_version_tag(
            name=model_name,
            version=version,
            key="role",
            value="champion",
        )

        logger.info(
            f"Model {model_name} v{version} promoted to Production (Champion). "
            f"Previous champion v{current_prod} is now Challenger."
        )

    def get_production_version(self, model_name: str = MODEL_NAME) -> str | None:
        """Get the current Production (Champion) model version."""
        try:
            versions = self.client.get_latest_versions(model_name, stages=["Production"])
            return versions[0].version if versions else None
        except mlflow.exceptions.MlflowException:
            return None

    def get_challenger_version(self, model_name: str = MODEL_NAME) -> str | None:
        """Get the current Challenger model version (recently archived)."""
        try:
            versions = self.client.get_latest_versions(model_name, stages=["Archived"])
            for v in versions:
                mv = self.client.get_model_version(model_name, v.version)
                if mv.tags.get("role") == "challenger":
                    return v.version
            return None
        except mlflow.exceptions.MlflowException:
            return None

    def get_model_info(self, model_name: str = MODEL_NAME) -> dict[str, Any]:
        """Get current champion and challenger info."""
        champion_version = self.get_production_version(model_name)
        challenger_version = self.get_challenger_version(model_name)

        info = {
            "model_name": model_name,
            "champion_version": champion_version,
            "challenger_version": challenger_version,
        }

        if champion_version:
            mv = self.client.get_model_version(model_name, champion_version)
            info["champion_run_id"] = mv.run_id
            info["champion_source"] = mv.source

        if challenger_version:
            mv = self.client.get_model_version(model_name, challenger_version)
            info["challenger_run_id"] = mv.run_id

        return info

    def auto_promote_if_better(
        self,
        new_version: str,
        new_rmse: float,
        model_name: str = MODEL_NAME,
    ) -> bool:
        """
        Automatically promote new model if it beats current champion.

        Returns:
            True if promoted, False if kept current champion.
        """
        current_version = self.get_production_version(model_name)

        if current_version is None:
            # No existing champion — promote directly
            self.promote_to_production(new_version, model_name)
            return True

        # Get champion's metrics
        mv = self.client.get_model_version(model_name, current_version)
        run = self.client.get_run(mv.run_id)
        champion_rmse = run.data.metrics.get("val_rmse", float("inf"))

        if new_rmse < champion_rmse:
            self.promote_to_production(new_version, model_name)
            logger.info(
                f"New model promoted: RMSE {new_rmse:.2f} < {champion_rmse:.2f}"
            )
            return True
        else:
            # Keep in staging as challenger
            self.promote_to_staging(new_version, model_name)
            logger.info(
                f"New model kept in Staging: RMSE {new_rmse:.2f} >= {champion_rmse:.2f}"
            )
            return False
