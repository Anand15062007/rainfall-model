"""Download OpenWeather historical hourly observations for the configured city center.

The endpoint is subscription-gated. The API key is read from `.env` and is
never written to output files or logged.
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
from app.config.settings import load_config

URL = "https://history.openweathermap.org/data/2.5/history/city"


def _read_observation(item: dict, lat: float, lon: float) -> dict:
    main = item.get("main") or {}
    wind = item.get("wind") or {}
    weather = (item.get("weather") or [{}])[0]
    rain = item.get("rain") or {}
    row = {
        "timestamp": pd.to_datetime(item.get("dt"), unit="s", utc=True),
        "latitude": lat,
        "longitude": lon,
        "temperature": main.get("temp"),
        "feels_like": main.get("feels_like"),
        "relative_humidity": main.get("humidity"),
        "pressure": main.get("pressure"),
        "wind_speed": wind.get("speed"),
        "wind_direction": wind.get("deg"),
        "cloud_cover": (item.get("clouds") or {}).get("all"),
        "rainfall_1h_mm": rain.get("1h", 0.0),
        "weather_id": weather.get("id"),
        "weather_main": weather.get("main"),
    }
    if "dew_point" in main:
        row["dew_point"] = main["dew_point"]
    return row


def download_history(lat: float, lon: float, start_date: str, end_date: str,
                     output: str | Path, api_key: str | None = None,
                     pause_seconds: float = 1.0) -> pd.DataFrame:
    token = api_key or os.getenv("OPENWEATHER_API_KEY") or os.getenv("WEATHER_API_KEY")
    if not token:
        raise RuntimeError("OPENWEATHER_API_KEY is missing; set it in the local .env file")
    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_exclusive = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)
    if end_exclusive <= start:
        raise ValueError("end date must be on or after start date")

    session = requests.Session()
    rows = []
    cursor = start
    while cursor < end_exclusive:
        chunk_end = min(cursor + timedelta(days=7), end_exclusive)
        params = {"lat": lat, "lon": lon, "type": "hour",
                  "start": int(cursor.timestamp()), "end": int(chunk_end.timestamp()),
                  "appid": token}
        response = None
        for attempt in range(5):
            try:
                response = session.get(URL, params=params, timeout=(10, 90))
            except requests.RequestException as exc:
                if attempt == 4:
                    raise RuntimeError(f"OpenWeather request failed ({type(exc).__name__}); key omitted") from exc
                time.sleep(min(2 ** attempt, 30))
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 4:
                    raise RuntimeError(f"OpenWeather returned HTTP {response.status_code}; reduce request range or check service status")
                time.sleep(min(2 ** (attempt + 1), 60))
                continue
            break
        if response is None:
            raise RuntimeError("OpenWeather request failed")
        if response.status_code in (401, 403):
            raise RuntimeError("OpenWeather denied historical access (HTTP %s). Check key validity and that the account subscription includes the Weather History API." % response.status_code)
        if not response.ok:
            raise RuntimeError(f"OpenWeather returned HTTP {response.status_code}; request URL/key redacted")
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("OpenWeather returned a non-JSON response") from exc
        hourly = payload.get("list", [])
        if not isinstance(hourly, list):
            raise RuntimeError("OpenWeather response has no hourly 'list' field")
        rows.extend(_read_observation(item, lat, lon) for item in hourly if item.get("dt") is not None)
        print(f"OpenWeather: {cursor:%Y-%m-%d} to {chunk_end:%Y-%m-%d}: {len(hourly)} hourly rows")
        cursor = chunk_end
        if pause_seconds > 0 and cursor < end_exclusive:
            time.sleep(pause_seconds)

    if not rows:
        raise RuntimeError("OpenWeather returned no historical rows for the selected period")
    frame = pd.DataFrame(rows).drop_duplicates("timestamp").sort_values("timestamp")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"Saved {len(frame)} hourly observations to {path}")
    return frame


def hourly_to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly OWM history into causal, issue-day predictors."""
    frame = hourly.copy()
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
    frame["date"] = frame.timestamp.dt.floor("D")
    wind = pd.to_numeric(frame.get("wind_speed"), errors="coerce")
    direction = np.deg2rad(pd.to_numeric(frame.get("wind_direction"), errors="coerce"))
    frame["wind_u"] = wind * np.sin(direction)
    frame["wind_v"] = wind * np.cos(direction)
    numeric = [c for c in ("temperature", "feels_like", "relative_humidity", "pressure",
                            "wind_speed", "wind_u", "wind_v", "cloud_cover",
                            "rainfall_1h_mm", "dew_point") if c in frame.columns]
    for col in numeric:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    agg = {f"{c}_mean": (c, "mean") for c in numeric if c != "rainfall_1h_mm"}
    if "rainfall_1h_mm" in frame:
        agg["rainfall_1h_mm_sum"] = ("rainfall_1h_mm", "sum")
    for c in ("temperature", "wind_speed", "cloud_cover", "rainfall_1h_mm"):
        if c in frame:
            agg[f"{c}_max"] = (c, "max")
    agg["owm_observation_count"] = ("timestamp", "count")
    daily = frame.groupby("date", sort=True).agg(**agg)
    daily = daily.rename(columns={"rainfall_1h_mm_sum": "owm_rainfall_mm",
                                  "rainfall_1h_mm_max": "owm_rainfall_1h_max"})
    if "weather_main" in frame:
        mode = frame.groupby("date").weather_main.agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown")
        daily["weather_main"] = mode
    full_dates = pd.date_range(daily.index.min(), daily.index.max(), freq="D", tz="UTC")
    daily = daily.reindex(full_dates)
    daily.index.name = "date_index"
    daily["owm_observation_count"] = daily.owm_observation_count.fillna(0)
    daily["date"] = daily.index
    daily = daily.reset_index(drop=True)
    # Antecedent precipitation only: at date t these summaries include rain observed
    # through t, used to predict the CHIRPS target for t+1.
    for days in (3, 7):
        daily[f"owm_rainfall_{days}d_sum"] = daily.owm_rainfall_mm.rolling(days, min_periods=days).sum()
    daily["day_of_year"] = daily.date.dt.dayofyear
    daily["doy_sin"] = np.sin(2 * np.pi * daily.day_of_year / 365.25)
    daily["doy_cos"] = np.cos(2 * np.pi * daily.day_of_year / 365.25)
    daily["pressure_change_hpa"] = daily.pressure_mean.diff() if "pressure_mean" in daily else np.nan
    return daily


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--start", help="UTC date YYYY-MM-DD; defaults to data.live_start_year")
    parser.add_argument("--end", help="UTC date YYYY-MM-DD; defaults to data.live_end_year")
    parser.add_argument("--output", default="data/raw/openweather_hourly.csv")
    parser.add_argument("--pause", type=float, default=1.0)
    args = parser.parse_args()
    load_dotenv()
    cfg = load_config(args.config)
    west, south, east, north = cfg["city"]["bbox"]
    lat, lon = (south + north) / 2, (west + east) / 2
    start = args.start or f"{cfg['data'].get('live_start_year', 2015)}-01-01"
    end = args.end or f"{cfg['data'].get('live_end_year', 2025)}-12-31"
    print(f"Querying historical weather near configured bbox center ({lat:.4f}, {lon:.4f})")
    download_history(lat, lon, start, end, args.output, pause_seconds=args.pause)


if __name__ == "__main__":
    main()
