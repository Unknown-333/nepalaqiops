"""
Telegram Alerting — sends alerts for AQI spikes, drift, and retrain events.
"""

import os
import logging

import requests

logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"


def _send_message(text: str) -> bool:
    """Send a message via Telegram Bot API."""
    if not BOT_TOKEN or not CHAT_ID:
        logger.warning("Telegram credentials not configured. Skipping alert.")
        logger.info(f"[TELEGRAM ALERT]: {text}")
        return False

    try:
        response = requests.post(
            TELEGRAM_API_URL,
            json={
                "chat_id": CHAT_ID,
                "text": text,
                "parse_mode": "HTML",
            },
            timeout=10,
        )
        response.raise_for_status()
        logger.info("Telegram alert sent successfully")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send Telegram alert: {e}")
        return False


def send_hazardous_alert(
    station_name: str,
    pm25_value: float,
    aqi: int,
    timestamp: str,
    forecast_trend: str = "stable",
) -> bool:
    """Send AQI Hazardous Alert."""
    text = (
        "🔴 <b>HAZARDOUS AIR QUALITY ALERT</b>\n\n"
        f"Station: {station_name}, Kathmandu\n"
        f"PM2.5: {pm25_value:.1f} µg/m³ (AQI: {aqi})\n"
        f"Time: {timestamp} (NPT)\n"
        f"Forecast next 6h: {forecast_trend}\n\n"
        "⚠️ Wear N95 mask. Avoid outdoor activity."
    )
    return _send_message(text)


def send_drift_alert(
    feature_name: str,
    psi_score: float,
    dag_run_id: str,
) -> bool:
    """Send Model Drift Alert."""
    text = (
        "⚠️ <b>MODEL DRIFT DETECTED — NepalAQI-Ops</b>\n\n"
        f"Feature: {feature_name}\n"
        f"PSI Score: {psi_score:.3f} (threshold: 0.25)\n"
        f"Action: Emergency retrain triggered.\n"
        f"Run ID: {dag_run_id}"
    )
    return _send_message(text)


def send_retrain_complete_alert(
    model_name: str,
    version: str,
    val_rmse: float,
    prev_rmse: float,
) -> bool:
    """Send Retrain Complete Alert."""
    if prev_rmse > 0:
        improvement = ((prev_rmse - val_rmse) / prev_rmse) * 100
    else:
        improvement = 0.0

    text = (
        "✅ <b>MODEL RETRAIN COMPLETE</b>\n\n"
        f"New champion: {model_name} v{version}\n"
        f"Val RMSE: {val_rmse:.2f} (prev: {prev_rmse:.2f})\n"
        f"Improvement: {improvement:.1f}%"
    )
    return _send_message(text)
