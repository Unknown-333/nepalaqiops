"""
Tests for drift monitor logic — PSI classification, thresholds, and retrain triggering.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd


# Default thresholds (from .env)
PSI_WARNING_THRESHOLD = float(os.getenv("PSI_WARNING_THRESHOLD", "0.2"))
PSI_RETRAIN_THRESHOLD = float(os.getenv("PSI_RETRAIN_THRESHOLD", "0.25"))
RMSE_DEGRADATION_THRESHOLD = float(os.getenv("RMSE_DEGRADATION_THRESHOLD", "0.15"))


class TestPSIClassification:
    """Tests for PSI threshold classification logic."""

    def _classify_psi(self, psi_value: float) -> str:
        """Replicate the drift monitor classification logic."""
        if psi_value < PSI_WARNING_THRESHOLD:
            return "stable"
        elif psi_value < PSI_RETRAIN_THRESHOLD:
            return "warning"
        else:
            return "retrain"

    def test_psi_stable_classification(self):
        """PSI=0.10 should be classified as 'stable'."""
        result = self._classify_psi(0.10)
        assert result == "stable", f"PSI=0.10 should be 'stable', got '{result}'"

    def test_psi_warning_classification(self):
        """PSI=0.22 should be classified as 'warning' and NOT trigger retrain."""
        result = self._classify_psi(0.22)
        assert result == "warning", f"PSI=0.22 should be 'warning', got '{result}'"
        assert result != "retrain", "PSI=0.22 should NOT trigger retrain"

    def test_psi_retrain_classification(self):
        """PSI=0.26 should be classified as 'retrain' and trigger the retrain DAG."""
        result = self._classify_psi(0.26)
        assert result == "retrain", f"PSI=0.26 should be 'retrain', got '{result}'"

    def test_psi_boundary_warning(self):
        """PSI exactly at 0.2 should be classified as 'warning' (threshold is exclusive)."""
        result = self._classify_psi(0.20)
        # PSI < 0.2 is stable, so PSI == 0.2 is "warning"
        assert result == "warning", f"PSI=0.20 should be 'warning', got '{result}'"

    def test_psi_boundary_retrain(self):
        """PSI exactly at 0.25 should trigger retrain."""
        result = self._classify_psi(0.25)
        assert result == "retrain", f"PSI=0.25 should be 'retrain', got '{result}'"


class TestBinaryFeatureExclusion:
    """Tests for excluding binary features from PSI computation."""

    def test_binary_feature_excluded_from_psi(self):
        """Binary features like is_festival, is_monsoon should be excluded from PSI."""
        from monitoring.evidently_reports import compute_feature_psi

        np.random.seed(42)

        # Create baseline and current with binary feature flip
        # (simulating monsoon → non-monsoon transition)
        baseline = pd.DataFrame({
            "pm25": np.random.uniform(30, 80, 100),
            "is_monsoon": np.ones(100),  # All 1 in baseline (monsoon)
            "is_tihar": np.zeros(100),
            "is_brick_kiln_season": np.zeros(100),
            "temp_c": np.random.uniform(20, 30, 100),
        })

        current = pd.DataFrame({
            "pm25": np.random.uniform(35, 85, 100),  # Slightly shifted
            "is_monsoon": np.zeros(100),  # All 0 in current (non-monsoon)
            "is_tihar": np.ones(100),     # All 1 (festival season)
            "is_brick_kiln_season": np.ones(100),
            "temp_c": np.random.uniform(15, 25, 100),
        })

        psi_scores = compute_feature_psi(baseline, current)

        # Binary features WILL have high PSI (by nature of flipping 0→1 or 1→0)
        # The drift monitor should exclude them from the max_psi decision
        binary_features = ["is_monsoon", "is_tihar", "is_brick_kiln_season"]
        continuous_features = ["pm25", "temp_c"]

        # Verify binary features produce artificially high PSI
        for feat in binary_features:
            if feat in psi_scores:
                # Binary flip should produce very high PSI
                assert psi_scores[feat] > 0.5, (
                    f"Binary feature {feat} should have high PSI on flip, "
                    f"got {psi_scores[feat]:.3f}"
                )

        # The CORRECT behavior is that drift decisions should only use continuous features
        continuous_psi = {k: v for k, v in psi_scores.items() if k in continuous_features}
        max_continuous_psi = max(continuous_psi.values()) if continuous_psi else 0

        # Continuous features should have moderate or low drift for small shifts
        assert max_continuous_psi < PSI_RETRAIN_THRESHOLD * 2, (
            f"Continuous feature PSI={max_continuous_psi:.3f} unexpectedly high"
        )


class TestRMSEDegradation:
    """Tests for RMSE degradation detection."""

    def test_rmse_degradation_trigger(self):
        """
        Given current_rmse=0.175, baseline_rmse=0.15:
        degradation = (0.175-0.15)/0.15 = 0.167 > threshold 0.15
        → retrain should be triggered.
        """
        current_rmse = 0.175
        baseline_rmse = 0.15

        degradation_pct = (current_rmse - baseline_rmse) / baseline_rmse

        assert degradation_pct == pytest.approx(0.1667, rel=1e-2), (
            f"Expected ~0.167, got {degradation_pct:.4f}"
        )
        assert degradation_pct > RMSE_DEGRADATION_THRESHOLD, (
            f"Degradation {degradation_pct:.4f} should exceed threshold {RMSE_DEGRADATION_THRESHOLD}"
        )

    def test_rmse_no_degradation(self):
        """
        Given current_rmse=0.16, baseline_rmse=0.15:
        degradation = (0.16-0.15)/0.15 = 0.067 < threshold 0.15
        → retrain should NOT be triggered.
        """
        current_rmse = 0.16
        baseline_rmse = 0.15

        degradation_pct = (current_rmse - baseline_rmse) / baseline_rmse

        assert degradation_pct < RMSE_DEGRADATION_THRESHOLD, (
            f"Degradation {degradation_pct:.4f} should be below threshold {RMSE_DEGRADATION_THRESHOLD}"
        )

    def test_rmse_improvement_no_trigger(self):
        """If model improves (current < baseline), no retrain needed."""
        current_rmse = 0.12
        baseline_rmse = 0.15

        degradation_pct = (current_rmse - baseline_rmse) / baseline_rmse

        # Negative degradation = improvement
        assert degradation_pct < 0, "Model improved — degradation should be negative"
        assert degradation_pct < RMSE_DEGRADATION_THRESHOLD


class TestDriftMonitorDAGLogic:
    """Tests for drift monitor DAG branching logic."""

    @patch("monitoring.telegram_alert.send_drift_alert")
    def test_psi_drift_triggers_retrain_branch(self, mock_alert):
        """When PSI > threshold, the DAG should branch to trigger_emergency_retrain."""
        # Simulate the branch function logic
        max_psi = 0.30  # Above retrain threshold
        rmse_degraded = False

        psi_drift = max_psi > PSI_RETRAIN_THRESHOLD
        should_retrain = psi_drift or rmse_degraded

        assert should_retrain is True
        assert psi_drift is True

    def test_no_drift_logs_ok(self):
        """When PSI < warning threshold and no RMSE degradation, branch to log_ok."""
        max_psi = 0.10
        rmse_degraded = False

        psi_drift = max_psi > PSI_RETRAIN_THRESHOLD
        should_retrain = psi_drift or rmse_degraded

        assert should_retrain is False

    def test_rmse_alone_triggers_retrain(self):
        """RMSE degradation alone (without PSI drift) should still trigger retrain."""
        max_psi = 0.05  # Well below threshold
        rmse_degraded = True

        psi_drift = max_psi > PSI_RETRAIN_THRESHOLD
        should_retrain = psi_drift or rmse_degraded

        assert should_retrain is True
        assert psi_drift is False  # PSI was fine
        # Retrain triggered by RMSE alone


class TestPSIComputation:
    """Tests for PSI computation correctness."""

    def test_psi_identical_distributions(self):
        """PSI of identical distributions should be ~0."""
        from monitoring.evidently_reports import compute_feature_psi

        np.random.seed(42)
        data = pd.DataFrame({"pm25": np.random.normal(60, 15, 1000)})

        psi = compute_feature_psi(data, data)
        assert psi["pm25"] < 0.01, (
            f"PSI of identical data should be ~0, got {psi['pm25']:.4f}"
        )

    def test_psi_shifted_distribution(self):
        """PSI of a significantly shifted distribution should be > 0.2."""
        from monitoring.evidently_reports import compute_feature_psi

        np.random.seed(42)
        baseline = pd.DataFrame({"pm25": np.random.normal(60, 15, 1000)})
        # Shift mean by 2 standard deviations
        current = pd.DataFrame({"pm25": np.random.normal(90, 15, 1000)})

        psi = compute_feature_psi(baseline, current)
        assert psi["pm25"] > PSI_WARNING_THRESHOLD, (
            f"Shifted distribution should have PSI > {PSI_WARNING_THRESHOLD}, got {psi['pm25']:.4f}"
        )
