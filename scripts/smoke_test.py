"""
NepalAQI-Ops — Automated Smoke Test (Python version).
Verifies end-to-end pipeline health without manual intervention.

Usage:
    python scripts/smoke_test.py [--verbose] [--timeout 120]
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import NamedTuple

import requests

# Configuration
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5000")
AIRFLOW_URL = os.getenv("AIRFLOW_URL", "http://localhost:8080")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class CheckResult(NamedTuple):
    name: str
    passed: bool
    latency_ms: float
    detail: str


def timed_check(name: str):
    """Decorator to time and wrap check functions."""
    def decorator(func):
        def wrapper(*args, **kwargs) -> CheckResult:
            start = time.time()
            try:
                passed, detail = func(*args, **kwargs)
                latency = (time.time() - start) * 1000
                return CheckResult(name=name, passed=passed, latency_ms=latency, detail=detail)
            except Exception as e:
                latency = (time.time() - start) * 1000
                return CheckResult(name=name, passed=False, latency_ms=latency, detail=str(e))
        return wrapper
    return decorator


@timed_check("FastAPI Health")
def check_health() -> tuple[bool, str]:
    resp = requests.get(f"{FASTAPI_URL}/health", timeout=10)
    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}"
    data = resp.json()
    if data.get("status") != "ok":
        return False, f"status={data.get('status')}"
    return True, f"champion={data.get('champion_model', 'none')}"


@timed_check("MLflow Registry")
def check_mlflow_model() -> tuple[bool, str]:
    resp = requests.get(
        f"{MLFLOW_URL}/api/2.0/mlflow/registered-models/search",
        params={"max_results": 10},
        timeout=10,
    )
    if resp.status_code != 200:
        return False, f"MLflow returned HTTP {resp.status_code}"

    data = resp.json()
    models = data.get("registered_models", [])
    if not models:
        return True, "No models registered yet (acceptable on fresh deploy)"

    # Check for Production stage model
    for model in models:
        versions = model.get("latest_versions", [])
        for v in versions:
            if v.get("current_stage") == "Production":
                return True, f"Champion: {model['name']} v{v['version']}"

    return True, f"{len(models)} model(s) registered, none in Production yet"


@timed_check("Forecast Endpoint")
def check_forecast() -> tuple[bool, str]:
    resp = requests.get(
        f"{FASTAPI_URL}/forecast/aqicn_kathmandu",
        params={"hours": 24},
        timeout=15,
    )
    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}: {resp.text[:100]}"

    data = resp.json()
    forecasts = data.get("forecasts", [])

    if len(forecasts) < 24:
        return False, f"Expected 24 forecasts, got {len(forecasts)}"

    # Validate predictions are reasonable
    pm25_values = [f["pm25_predicted"] for f in forecasts]
    if any(v is None or v < 0 or v > 500 for v in pm25_values):
        bad = [v for v in pm25_values if v is None or v < 0 or v > 500]
        return False, f"Invalid PM2.5 values: {bad[:3]}"

    # Check for flat-line (identical predictions = model not learning)
    import numpy as np
    std = np.std(pm25_values)
    if std < 0.5:
        return False, f"Flat-line detected: std={std:.4f} (model may be broken)"

    return True, f"24h forecast OK, range=[{min(pm25_values):.1f}, {max(pm25_values):.1f}]"


@timed_check("Heatmap GeoJSON")
def check_heatmap() -> tuple[bool, str]:
    resp = requests.get(f"{FASTAPI_URL}/forecast/heatmap", timeout=10)
    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}"

    data = resp.json()
    if data.get("type") != "FeatureCollection":
        return False, "Not a valid GeoJSON FeatureCollection"

    features = data.get("features", [])
    if len(features) < 32:
        return False, f"Expected 32 wards, got {len(features)}"

    return True, f"{len(features)} ward features present"


@timed_check("Anomalies Endpoint")
def check_anomalies() -> tuple[bool, str]:
    resp = requests.get(f"{FASTAPI_URL}/anomalies/latest", timeout=10)
    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}"

    data = resp.json()
    if not isinstance(data, (list, dict)):
        return False, "Response is not JSON array/object"

    return True, "OK"


@timed_check("Challenger Routing")
def check_challenger_routing() -> tuple[bool, str]:
    resp = requests.get(
        f"{FASTAPI_URL}/forecast/aqicn_kathmandu",
        params={"hours": 6},
        headers={"X-Model-Version": "challenger"},
        timeout=10,
    )
    # 200 (challenger exists) or graceful response (no challenger) both acceptable
    # 500 is a failure
    if resp.status_code == 500:
        return False, f"Server error on challenger routing: {resp.text[:100]}"

    return True, f"HTTP {resp.status_code} (graceful handling)"


@timed_check("Prometheus Metrics")
def check_metrics() -> tuple[bool, str]:
    resp = requests.get(f"{FASTAPI_URL}/metrics", timeout=5)
    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}"

    if "nepalaqiops" not in resp.text and "python_info" not in resp.text:
        return False, "No recognizable metrics in response"

    return True, "Prometheus metrics exposed"


def run_all_checks() -> list[CheckResult]:
    """Execute all smoke test checks sequentially."""
    checks = [
        check_health,
        check_mlflow_model,
        check_forecast,
        check_heatmap,
        check_anomalies,
        check_challenger_routing,
        check_metrics,
    ]
    return [check() for check in checks]


def print_results(results: list[CheckResult]) -> int:
    """Print results table and return exit code."""
    print("\n" + "=" * 70)
    print(" NepalAQI-Ops Smoke Test Results")
    print(f" {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    passed = 0
    failed = 0

    for r in results:
        icon = "\033[92m✓\033[0m" if r.passed else "\033[91m✗\033[0m"
        status = "PASS" if r.passed else "FAIL"
        print(f"  {icon} [{status}] {r.name:<24} ({r.latency_ms:>6.0f}ms) — {r.detail}")
        if r.passed:
            passed += 1
        else:
            failed += 1

    print("-" * 70)
    print(f"  PASSED: {passed}  |  FAILED: {failed}  |  TOTAL: {len(results)}")
    print("=" * 70)

    return 0 if failed == 0 else 1


def wait_for_service(url: str, timeout: int = 120) -> bool:
    """Wait for a service to become available."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code < 500:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(2)
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NepalAQI-Ops Smoke Test")
    parser.add_argument("--timeout", type=int, default=120, help="Service readiness timeout (seconds)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Wait for FastAPI to be ready
    print(f"Waiting for FastAPI at {FASTAPI_URL}...")
    if not wait_for_service(f"{FASTAPI_URL}/health", timeout=args.timeout):
        print(f"\033[91mFATAL: FastAPI not reachable after {args.timeout}s\033[0m")
        sys.exit(2)

    results = run_all_checks()
    exit_code = print_results(results)
    sys.exit(exit_code)
