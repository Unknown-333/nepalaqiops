"""
NepalAQI-Ops Streamlit Dashboard — live AQI map, forecasts, SHAP explainability.
"""

import os
import json
from datetime import datetime, timedelta, timezone

import streamlit as st
import requests
import pandas as pd
import numpy as np

# Configuration
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi:8000")
st.set_page_config(
    page_title="NepalAQI-Ops Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar navigation
st.sidebar.title("🌍 NepalAQI-Ops")
st.sidebar.markdown("**Air Quality Intelligence**\nKathmandu Valley, Nepal")

page = st.sidebar.radio(
    "Navigate",
    ["🗺️ Live AQI Map", "📈 24-Hour Forecast", "🧠 SHAP Explainability",
     "📊 Model Health", "🚨 Anomaly Log"],
)


def fetch_api(endpoint: str, params: dict = None) -> dict | None:
    """Fetch data from FastAPI backend."""
    try:
        response = requests.get(f"{FASTAPI_URL}{endpoint}", params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


# ===========================================================================
# PAGE 1: Live AQI Map
# ===========================================================================
if page == "🗺️ Live AQI Map":
    st.title("🗺️ Live Air Quality Map — Kathmandu Valley")

    # Pollutant selector
    pollutant = st.sidebar.selectbox("Pollutant", ["PM2.5", "PM10", "NO2"])

    # Auto-refresh
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=300000, limit=None, key="map_refresh")  # 5 min
    except ImportError:
        pass

    # Fetch heatmap data
    heatmap_data = fetch_api("/forecast/heatmap")

    if heatmap_data:
        import folium
        from streamlit_folium import folium_static

        # Create map centered on Kathmandu
        m = folium.Map(location=[27.7172, 85.3240], zoom_start=12, tiles="CartoDB positron")

        # Color mapping for AQI categories
        color_map = {
            "Good": "green",
            "Moderate": "orange",
            "USG": "darkorange",
            "Unhealthy": "red",
            "Very Unhealthy": "purple",
            "Hazardous": "darkred",
            "Unknown": "gray",
        }

        for feature in heatmap_data.get("features", []):
            props = feature["properties"]
            coords = feature["geometry"]["coordinates"]
            pm25 = props.get("pm25")
            category = props.get("aqi_category", "Unknown")
            color = color_map.get(category, "gray")

            popup_text = (
                f"<b>{props['ward_name']}</b><br>"
                f"PM2.5: {pm25:.1f} µg/m³<br>"
                f"Category: {category}"
            ) if pm25 else f"<b>{props['ward_name']}</b><br>No data"

            folium.CircleMarker(
                location=[coords[1], coords[0]],
                radius=10,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_text, max_width=200),
            ).add_to(m)

        # Legend
        legend_html = """
        <div style="position: fixed; bottom: 50px; left: 50px; z-index:1000; 
             background: white; padding: 10px; border-radius: 5px; border: 1px solid gray;">
        <b>AQI Categories</b><br>
        🟢 Good (0-12)<br>🟡 Moderate (12-35)<br>🟠 USG (35-55)<br>
        🔴 Unhealthy (55-150)<br>🟣 Very Unhealthy (150-250)<br>⚫ Hazardous (250+)
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        folium_static(m, width=900, height=600)
    else:
        st.info("No heatmap data available. Ensure the API is running and data has been ingested.")

# ===========================================================================
# PAGE 2: 24-Hour Forecast
# ===========================================================================
elif page == "📈 24-Hour Forecast":
    st.title("📈 24-Hour PM2.5 Forecast")

    # Station selector
    stations = [f"aqicn_{city}" for city in ["kathmandu", "patan", "bhaktapur", "kirtipur"]]
    station = st.sidebar.selectbox("Station", stations)

    # Model toggle
    show_models = st.sidebar.multiselect(
        "Show Models", ["ensemble", "prophet", "lstm"], default=["ensemble"]
    )

    # Fetch forecast
    forecast_data = fetch_api(f"/forecast/{station}", params={"hours": 24})

    if forecast_data:
        import plotly.graph_objects as go

        forecasts = forecast_data["forecasts"]
        hours = [f["hour"] for f in forecasts]
        predictions = [f["pm25_predicted"] for f in forecasts]
        lower = [f["confidence_lower"] for f in forecasts]
        upper = [f["confidence_upper"] for f in forecasts]

        fig = go.Figure()

        # Confidence interval
        fig.add_trace(go.Scatter(
            x=hours + hours[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor="rgba(0, 100, 200, 0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% Confidence",
            showlegend=True,
        ))

        # Prediction line
        fig.add_trace(go.Scatter(
            x=hours,
            y=predictions,
            mode="lines+markers",
            name=f"Forecast ({forecast_data['model_used']})",
            line=dict(color="rgb(0, 100, 200)", width=2),
        ))

        # AQI threshold lines
        fig.add_hline(y=35.4, line_dash="dash", line_color="orange",
                      annotation_text="Moderate", annotation_position="top right")
        fig.add_hline(y=55.4, line_dash="dash", line_color="red",
                      annotation_text="Unhealthy (Sensitive)", annotation_position="top right")
        fig.add_hline(y=150.4, line_dash="dash", line_color="purple",
                      annotation_text="Unhealthy", annotation_position="top right")

        fig.update_layout(
            title=f"PM2.5 Forecast — {station}",
            xaxis_title="Time",
            yaxis_title="PM2.5 (µg/m³)",
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Current", f"{predictions[0]:.1f} µg/m³")
        col2.metric("Max (next 24h)", f"{max(predictions):.1f} µg/m³")
        col3.metric("Model", forecast_data["model_used"])
    else:
        st.info("No forecast data available.")

# ===========================================================================
# PAGE 3: SHAP Explainability
# ===========================================================================
elif page == "🧠 SHAP Explainability":
    st.title("🧠 Why is AQI High Today?")
    st.markdown("SHAP analysis of the key factors driving current PM2.5 levels.")

    # Simulated SHAP values (in production, fetched from model explainer)
    shap_features = {
        "Brick kiln season": 12.5,
        "Previous day PM2.5": 8.3,
        "Low wind speed": 6.1,
        "High humidity": 4.2,
        "Rush hour traffic": 3.8,
        "Temperature inversion": 2.9,
        "No precipitation": 2.1,
        "Weekend effect": -1.5,
        "Monsoon (off)": 1.2,
        "Festival (none)": 0.3,
    }

    # Waterfall-style chart
    import plotly.graph_objects as go

    features = list(shap_features.keys())
    values = list(shap_features.values())
    colors = ["red" if v > 0 else "green" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker_color=colors,
        text=[f"+{v:.1f}" if v > 0 else f"{v:.1f}" for v in values],
        textposition="outside",
    ))

    fig.update_layout(
        title="SHAP Feature Contributions to Current PM2.5 Prediction",
        xaxis_title="Contribution (µg/m³)",
        height=450,
        margin=dict(l=200),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Plain English interpretation
    st.subheader("📝 Interpretation")
    top_positive = [(k, v) for k, v in shap_features.items() if v > 2]
    interpretation = "Today's elevated PM2.5 is primarily driven by: "
    interpretation += "; ".join([f"**{k}** (+{v:.1f} µg/m³)" for k, v in top_positive])
    st.markdown(interpretation)

# ===========================================================================
# PAGE 4: Model Health
# ===========================================================================
elif page == "📊 Model Health":
    st.title("📊 Model Health Dashboard")

    # Fetch health info
    health_data = fetch_api("/health")

    if health_data:
        col1, col2, col3 = st.columns(3)
        col1.metric("Champion Model", health_data.get("champion_model", "N/A"))
        col2.metric("Challenger Model", health_data.get("challenger_model", "N/A"))
        col3.metric("Last Retrain", health_data.get("last_retrain", "N/A")[:10])

    # PSI Scores (simulated)
    st.subheader("Feature Drift (PSI Scores)")
    psi_data = {
        "Feature": ["pm25", "pm10", "temp_c", "humidity_pct", "wind_speed_kmh"],
        "PSI Score": [0.08, 0.12, 0.05, 0.15, 0.03],
    }
    psi_df = pd.DataFrame(psi_data)

    import plotly.express as px
    fig = px.bar(psi_df, x="Feature", y="PSI Score", color="PSI Score",
                 color_continuous_scale=["green", "yellow", "red"])
    fig.add_hline(y=0.25, line_dash="dash", line_color="red",
                  annotation_text="Retrain Threshold")
    fig.add_hline(y=0.2, line_dash="dot", line_color="orange",
                  annotation_text="Warning Threshold")
    st.plotly_chart(fig, use_container_width=True)

    # Model RMSE over time (simulated)
    st.subheader("Champion vs Challenger RMSE (Last 30 Days)")
    dates = pd.date_range(end=datetime.now(), periods=30, freq="D")
    champion_rmse = np.random.normal(12, 1.5, 30).cumsum() / np.arange(1, 31) + 10
    challenger_rmse = np.random.normal(11.5, 1.5, 30).cumsum() / np.arange(1, 31) + 9.5

    rmse_df = pd.DataFrame({
        "Date": dates,
        "Champion RMSE": champion_rmse,
        "Challenger RMSE": challenger_rmse,
    })
    st.line_chart(rmse_df.set_index("Date"))

# ===========================================================================
# PAGE 5: Anomaly Log
# ===========================================================================
elif page == "🚨 Anomaly Log":
    st.title("🚨 Anomaly Detection Log")

    # Fetch anomalies
    anomaly_data = fetch_api("/anomalies/latest", params={"limit": 100})

    if anomaly_data and anomaly_data.get("events"):
        events = anomaly_data["events"]
        df = pd.DataFrame(events)

        # Filters
        col1, col2 = st.columns(2)
        if "station_id" in df.columns:
            station_filter = col1.multiselect(
                "Filter by Station", df["station_id"].unique()
            )
            if station_filter:
                df = df[df["station_id"].isin(station_filter)]

        if "anomaly_score" in df.columns:
            score_threshold = col2.slider("Min Anomaly Score", -2.0, 0.0, -1.0)
            df = df[df["anomaly_score"] <= score_threshold]

        st.dataframe(df, use_container_width=True)

        # Export button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Export to CSV",
            data=csv,
            file_name="anomaly_events.csv",
            mime="text/csv",
        )

        st.metric("Total Anomalies", len(df))
    else:
        st.info("No anomaly events detected yet. Events will appear after data ingestion begins.")
