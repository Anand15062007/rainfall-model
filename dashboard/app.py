"""Coordinate-based rainfall forecast website."""
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import json

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from app.data.openweather_forecast import get_forecast, summarize_date

load_dotenv()
st.set_page_config(page_title="Rainfall Forecast", page_icon="🌧️", layout="wide")
st.title("🌧️ Rainfall Forecast")
st.caption("Enter a location and date to see OpenWeather's forecast rain chance and estimated rainfall.")

now_utc = datetime.now(timezone.utc)
today_utc = now_utc.date()
with st.form("forecast_form"):
    left, middle, right = st.columns(3)
    latitude = left.number_input("Latitude", min_value=-90.0, max_value=90.0,
                                 value=40.7128, format="%.5f", help="Range: -90 to 90")
    longitude = middle.number_input("Longitude", min_value=-180.0, max_value=180.0,
                                    value=-74.0060, format="%.5f", help="Range: -180 to 180")
    forecast_date = right.date_input("Forecast date (UTC)", value=today_utc,
                                     min_value=today_utc, max_value=today_utc + timedelta(days=4))
    submitted = st.form_submit_button("Get rainfall forecast", type="primary", use_container_width=True)

if submitted:
    with st.spinner("Getting forecast for these coordinates…"):
        try:
            place, forecast = get_forecast(latitude, longitude)
            summary = summarize_date(forecast, forecast_date)
            st.session_state["rain_forecast_result"] = {
                "place": place,
                "forecast": forecast,
                "summary": summary,
                "latitude": latitude,
                "longitude": longitude,
                "date": forecast_date,
            }
        except (RuntimeError, ValueError) as exc:
            st.session_state.pop("rain_forecast_result", None)
            st.error(str(exc))

result = st.session_state.get("rain_forecast_result")
if result:
    place = result["place"]
    summary = result["summary"]
    day = summary["intervals"]
    place_name = ", ".join(str(x) for x in (place.get("name"), place.get("country")) if x)
    heading = f"{place_name} · {result['date'].isoformat()} UTC" if place_name else f"{result['date'].isoformat()} UTC"
    st.subheader(heading)
    a, b, c = st.columns(3)
    a.metric("Peak 3-hour rain chance", f"{summary['rain_probability']:.0%}")
    b.metric("Forecast rainfall total", f"{summary['rainfall_total_mm']:.1f} mm")
    if pd.notna(summary["temperature_max_c"]) and pd.notna(summary["temperature_min_c"]):
        c.metric("Forecast temperature", f"{summary['temperature_min_c']:.1f}–{summary['temperature_max_c']:.1f} °C")
    else:
        c.metric("Forecast intervals", len(day))
    st.info("The percentage is OpenWeather's highest precipitation probability among the returned 3-hour periods on this UTC date. It is not a calibrated probability of rain at any time during the full day.")
    chart = day.set_index("timestamp_utc")[["rain_probability", "rain_mm_3h"]].rename(
        columns={"rain_probability": "Rain chance (0–1)", "rain_mm_3h": "Rainfall (mm / 3h)"})
    st.markdown("#### Forecast by 3-hour period")
    st.line_chart(chart, height=300)
    display = day[["timestamp_utc", "description", "rain_probability", "rain_mm_3h", "temperature_c"]].copy()
    display["timestamp_utc"] = display.timestamp_utc.dt.strftime("%Y-%m-%d %H:%M")
    display["rain_probability"] = display.rain_probability.map(lambda x: f"{x:.0%}")
    display = display.rename(columns={"timestamp_utc": "Time (UTC)", "description": "Conditions",
        "rain_probability": "Rain chance", "rain_mm_3h": "Rainfall (mm / 3h)", "temperature_c": "Temperature (°C)"})
    st.dataframe(display, hide_index=True, use_container_width=True)
    st.caption(f"Location: {result['latitude']:.5f}, {result['longitude']:.5f} · Source: OpenWeather 5-day / 3-hour forecast API")
else:
    st.markdown("### How it works")
    st.write("The site requests a forecast for your coordinates and summarizes the selected UTC date. The OpenWeather API returns forecasts in 3-hour periods, so the rain chance shown is the highest returned period chance for that date.")
    st.caption("Forecast access depends on your OpenWeather account and API key. Add OPENWEATHER_API_KEY to the local .env file.")

with st.expander("Project data and model results"):
    metrics = Path("experiments/daily_live_metrics.json")
    comparison = Path("experiments/daily_live_model_comparison.csv")
    if metrics.exists():
        scores = json.loads(metrics.read_text(encoding="utf-8"))
        st.caption(f"Historical model target: {scores.get('target', 'not recorded')}")
        st.caption(f"Data source: {scores.get('data_source', 'not recorded')}")
        if comparison.exists():
            st.dataframe(pd.read_csv(comparison), hide_index=True, use_container_width=True)
    else:
        st.info("No real-data model evaluation is available yet. The coordinate forecast above uses OpenWeather's forecast API directly.")
