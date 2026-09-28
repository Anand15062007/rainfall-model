"""Small adapter for OpenWeather's 5-day / 3-hour forecast endpoint."""
from __future__ import annotations

import os
from datetime import date

import pandas as pd
import requests
from dotenv import load_dotenv

FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"


def get_forecast(latitude: float, longitude: float, api_key: str | None = None) -> tuple[dict, pd.DataFrame]:
    """Fetch and normalize forecast intervals; timestamps and dates are UTC."""
    if not -90 <= latitude <= 90:
        raise ValueError("Latitude must be between -90 and 90 degrees")
    if not -180 <= longitude <= 180:
        raise ValueError("Longitude must be between -180 and 180 degrees")
    load_dotenv()
    token = api_key or os.getenv("OPENWEATHER_API_KEY") or os.getenv("WEATHER_API_KEY")
    if not token:
        raise RuntimeError("OPENWEATHER_API_KEY is missing. Add it to the local .env file and restart the website.")
    try:
        response = requests.get(
            FORECAST_URL,
            params={"lat": latitude, "lon": longitude, "appid": token, "units": "metric"},
            timeout=(10, 30),
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"OpenWeather request failed ({type(exc).__name__}). Check your internet connection.") from exc
    if response.status_code in (401, 403):
        raise RuntimeError("OpenWeather rejected the request. Check the API key and account access.")
    if response.status_code == 429:
        raise RuntimeError("OpenWeather rate limit reached. Wait a little before trying again.")
    if not response.ok:
        raise RuntimeError(f"OpenWeather returned HTTP {response.status_code}.")
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("OpenWeather returned an invalid response.") from exc
    if payload.get("cod") not in (None, 200, "200") or not isinstance(payload.get("list"), list):
        raise RuntimeError("OpenWeather did not return forecast intervals for these coordinates.")

    rows = []
    for item in payload["list"]:
        stamp = item.get("dt")
        if stamp is None:
            continue
        weather = (item.get("weather") or [{}])[0]
        rows.append({
            "timestamp_utc": pd.to_datetime(stamp, unit="s", utc=True),
            "rain_probability": float(item.get("pop") or 0.0),
            "rain_mm_3h": float((item.get("rain") or {}).get("3h") or 0.0),
            "temperature_c": (item.get("main") or {}).get("temp"),
            "description": weather.get("description", "Forecast"),
        })
    if not rows:
        raise RuntimeError("OpenWeather returned no forecast intervals.")
    frame = pd.DataFrame(rows).sort_values("timestamp_utc").reset_index(drop=True)
    frame["date_utc"] = frame.timestamp_utc.dt.date
    return payload.get("city", {}), frame


def summarize_date(frame: pd.DataFrame, target_date: date) -> dict:
    """Summarize a UTC date without inventing an unsupported daily probability.

    OpenWeather supplies a probability for each 3-hour interval. The returned
    probability is the maximum interval probability, not an independent-event
    combination or a calibrated probability of any rain over the full day.
    """
    day = frame[frame.date_utc == target_date].copy()
    if day.empty:
        raise ValueError("That date is outside the forecast intervals returned by OpenWeather.")
    peak_ix = day.rain_probability.idxmax()
    peak = day.loc[peak_ix]
    return {
        "rain_probability": float(peak.rain_probability),
        "peak_interval_utc": peak.timestamp_utc,
        "rainfall_total_mm": float(day.rain_mm_3h.sum()),
        "temperature_max_c": float(pd.to_numeric(day.temperature_c, errors="coerce").max()),
        "temperature_min_c": float(pd.to_numeric(day.temperature_c, errors="coerce").min()),
        "intervals": day,
    }
