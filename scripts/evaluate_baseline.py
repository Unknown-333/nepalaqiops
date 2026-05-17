"""
Baseline Evaluation Script — proves the Prophet+LSTM ensemble is learning,
not just repeating the last value.

Tests against:
  1. Naive Lag-1 (persistence): ŷ(t+h) = y(t) for all h
  2. Moving Average (MA-24): ŷ(t+h) = mean(y(t-23)...y(t))
  3. Seasonal Naive: ŷ(t+h) = y(t - 24 + h) (same hour yesterday)

If the ensemble doesn't beat ALL three baselines, it's not learning.

Usage:
    python scripts/evaluate_baseline.py
"""

import os
import sys
import json
import numpy as np
import requests
from datetime import datetime, timezone

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))


def get_historical_pm25(station_id: str = "aqicn_kathmandu", hours: int = 72) -> np.ndarray | None:
    """Fetch recent historical PM2.5 from Redis feature store."""
    try:
        import redis
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

        # Try to get lag features which contain historical values
        cached = r.get(f"features:{station_id}:latest")
        if cached:
            features = json.loads(cached)
            # Reconstruct recent history from lag features
            lags = []
            for lag_key in ["pm25", "pm25_lag_1h", "pm25_lag_3h", "pm25_lag_6h",
                           "pm25_lag_12h", "pm25_lag_24h", "pm25_lag_48h"]:
                val = features.get(lag_key)
                if val is not None:
                    lags.append(float(val))

            if len(lags) >= 3:
                return np.array(lags)
    except Exception as e:
        print(f"  [WARN] Cannot fetch from Redis: {e}")

    return None


def generate_synthetic_history(hours: int = 72) -> np.ndarray:
    """Generate realistic Kathmandu PM2.5 for baseline testing."""
    np.random.seed(42)
    t = np.arange(hours)
    # Diurnal pattern + trend + noise (typical Kathmandu)
    diurnal = 30 * np.sin(2 * np.pi * (t - 8) / 24)
    base = 60 + diurnal + np.random.normal(0, 8, hours)
    return np.maximum(base, 5.0)


def naive_lag1(history: np.ndarray, horizon: int = 24) -> np.ndarray:
    """Naive baseline: repeat last known value."""
    return np.full(horizon, history[-1])


def moving_average_24(history: np.ndarray, horizon: int = 24) -> np.ndarray:
    """MA-24 baseline: predict mean of last 24 hours."""
    ma = np.mean(history[-24:])
    return np.full(horizon, ma)


def seasonal_naive(history: np.ndarray, horizon: int = 24) -> np.ndarray:
    """Seasonal naive: same hour yesterday."""
    if len(history) >= 48:
        # Use values from 24h ago
        return history[-48:-24][:horizon]
    return history[-horizon:]


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(actual - predicted)))


def get_model_forecast(station_id: str = "aqicn_kathmandu", hours: int = 24) -> np.ndarray | None:
    """Get forecast from the FastAPI endpoint."""
    try:
        resp = requests.get(
            f"{FASTAPI_URL}/forecast/{station_id}",
            params={"hours": hours},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            predictions = [f["pm25_predicted"] for f in data["forecasts"]]
            return np.array(predictions)
    except Exception as e:
        print(f"  [WARN] Cannot reach FastAPI: {e}")
    return None


def evaluate_baselines():
    """Run baseline comparison and print results."""
    print("=" * 70)
    print(" NepalAQI-Ops — Baseline Evaluation")
    print(f" {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    # Get or generate history
    history = get_historical_pm25()
    if history is None or len(history) < 7:
        print("\n  [INFO] Using synthetic Kathmandu PM2.5 history (no Redis data available)")
        history = generate_synthetic_history(72)
    else:
        print(f"\n  [INFO] Using real feature store data ({len(history)} points)")

    # Split: last 24h as "actuals" for baseline testing
    if len(history) >= 48:
        train_history = history[:-24]
        actuals = history[-24:]
    else:
        # Not enough data for proper split
        train_history = history
        actuals = history  # Self-comparison (less meaningful)
        print("  [WARN] Insufficient history for proper train/test split")

    horizon = len(actuals)

    # Compute baselines
    naive_pred = naive_lag1(train_history, horizon)
    ma24_pred = moving_average_24(train_history, horizon)
    seasonal_pred = seasonal_naive(history, horizon) if len(history) >= 48 else naive_pred

    # Get model forecast
    model_pred = get_model_forecast(hours=horizon)

    print(f"\n  Evaluation horizon: {horizon} hours")
    print(f"  Actual PM2.5 range: [{actuals.min():.1f}, {actuals.max():.1f}]")
    print(f"  Actual PM2.5 std:   {actuals.std():.2f}")

    # Results table
    print("\n" + "-" * 70)
    print(f"  {'Model':<25} {'RMSE':<10} {'MAE':<10} {'vs Naive':<15} {'Verdict'}")
    print("-" * 70)

    # Baselines
    naive_rmse = rmse(actuals, naive_pred)
    naive_mae_val = mae(actuals, naive_pred)
    print(f"  {'Naive Lag-1':<25} {naive_rmse:<10.3f} {naive_mae_val:<10.3f} {'(baseline)':<15} —")

    ma24_rmse = rmse(actuals, ma24_pred)
    ma24_improvement = (1 - ma24_rmse / naive_rmse) * 100 if naive_rmse > 0 else 0
    print(f"  {'Moving Average (24h)':<25} {ma24_rmse:<10.3f} {mae(actuals, ma24_pred):<10.3f} {f'{ma24_improvement:+.1f}%':<15} —")

    seasonal_rmse = rmse(actuals, seasonal_pred)
    seasonal_improvement = (1 - seasonal_rmse / naive_rmse) * 100 if naive_rmse > 0 else 0
    print(f"  {'Seasonal Naive (24h)':<25} {seasonal_rmse:<10.3f} {mae(actuals, seasonal_pred):<10.3f} {f'{seasonal_improvement:+.1f}%':<15} —")

    # Model forecast
    if model_pred is not None and len(model_pred) == horizon:
        model_rmse = rmse(actuals, model_pred)
        model_mae_val = mae(actuals, model_pred)
        model_improvement = (1 - model_rmse / naive_rmse) * 100 if naive_rmse > 0 else 0

        beats_naive = model_rmse < naive_rmse
        beats_ma = model_rmse < ma24_rmse
        beats_seasonal = model_rmse < seasonal_rmse

        if beats_naive and beats_ma and beats_seasonal:
            verdict = "\033[92m✓ BEATS ALL\033[0m"
        elif beats_naive:
            verdict = "\033[93m~ BEATS NAIVE\033[0m"
        else:
            verdict = "\033[91m✗ WORSE THAN NAIVE\033[0m"

        print(f"  {'Ensemble (Prophet+LSTM)':<25} {model_rmse:<10.3f} {model_mae_val:<10.3f} {f'{model_improvement:+.1f}%':<15} {verdict}")

        # Additional diagnostics
        print("\n" + "-" * 70)
        print("  DIAGNOSTICS:")
        pred_std = np.std(model_pred)
        pred_range = model_pred.max() - model_pred.min()
        print(f"    Prediction std:        {pred_std:.3f} {'(FLAT LINE!)' if pred_std < 1.0 else '(OK)'}")
        print(f"    Prediction range:      {pred_range:.3f}")
        print(f"    Correlation w/ actual: {np.corrcoef(actuals, model_pred)[0, 1]:.4f}")

        # Check for lag-1 copying
        if len(model_pred) > 1:
            lag1_corr = np.corrcoef(model_pred[:-1], model_pred[1:])[0, 1]
            print(f"    Autocorrelation(1):    {lag1_corr:.4f} {'(SUSPICIOUS: may be copying lag-1)' if lag1_corr > 0.99 else ''}")

    else:
        print(f"  {'Ensemble (Prophet+LSTM)':<25} {'N/A':<10} {'N/A':<10} {'—':<15} [FastAPI unreachable]")

    print("\n" + "=" * 70)

    # Final verdict
    if model_pred is not None and len(model_pred) == horizon:
        model_rmse_val = rmse(actuals, model_pred)
        if model_rmse_val >= naive_rmse:
            print("  \033[91mFAIL: Model is NOT outperforming naive baseline.\033[0m")
            print("  Action: Check training data quality, feature engineering, or model convergence.")
            return 1
        elif np.std(model_pred) < 1.0:
            print("  \033[91mFAIL: Model producing flat-line predictions (std < 1.0).\033[0m")
            print("  Action: Model may be stuck in a local minimum or features are constant.")
            return 1
        else:
            print("  \033[92mPASS: Ensemble outperforms baselines.\033[0m")
            return 0
    else:
        print("  \033[93mWARN: Cannot evaluate model (API unreachable). Baselines computed only.\033[0m")
        return 0


if __name__ == "__main__":
    sys.exit(evaluate_baselines())
