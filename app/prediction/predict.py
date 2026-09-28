"""Load saved hourly or daily models and write rain forecasts to CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.data.openweather_history import hourly_to_daily
from app.features.build import build_features
from app.models.lstm import LSTMModel


def _positive_probability(model, X, groups=None):
    if not hasattr(model, "predict_proba"):
        raise ValueError("This model does not provide rain probabilities")
    values = np.asarray(model.predict_proba(X, groups=groups) if isinstance(model, LSTMModel)
                        else model.predict_proba(X))
    return values[:, 1] if values.ndim == 2 else values.reshape(-1)


def predict(model_path, input_path, output_path="data/processed/predictions.csv"):
    path = Path(model_path)
    if path.suffix == ".pt":
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = LSTMModel.load(path)
        meta = payload.get("metadata", {})
        bundle = {}
    else:
        bundle = joblib.load(path)
        model = bundle["model"]
        meta = bundle.get("metadata", {})

    daily_bundle = bool(bundle.get("forecast_horizon_days") and bundle.get("source"))
    features = bundle.get("features") if daily_bundle else meta.get("features")
    if not features:
        raise ValueError("Model bundle has no saved feature list")

    raw = pd.read_csv(input_path)
    if daily_bundle:
        if "timestamp" not in raw:
            raise ValueError("Daily models need hourly input with a timestamp column")
        frame = hourly_to_daily(raw)
        frame = frame[frame.owm_observation_count >= 18].copy()
        if frame.empty:
            raise ValueError("No complete weather days found; provide at least 18 hourly observations for one day")
        row = frame.tail(1).copy()
        missing = sorted(set(features) - set(row.columns))
        if missing:
            raise ValueError(f"Input data is missing model features: {', '.join(missing)}")
        X = row[features]
        issued = pd.to_datetime(row.date, utc=True).reset_index(drop=True)
        forecast_dates = issued + pd.Timedelta(days=1)
        groups = None
        target = bundle.get("target", "target_rain_event")
        threshold = bundle.get("threshold")
    else:
        if "timestamp" not in raw:
            raise ValueError("Hourly models need a timestamp column")
        config = meta.get("config", {})
        feature_cfg = config.get("features", {})
        frame = build_features(
            raw,
            lags=feature_cfg.get("lags", (1, 3, 6, 12, 24)),
            rolling_windows=feature_cfg.get("rolling_windows", (3, 6, 24)),
            rolling_columns=feature_cfg.get("rolling_columns", ("temperature", "relative_humidity", "rainfall_1h")),
            horizon_hours=config.get("forecast", {}).get("horizon_hours", 1),
            threshold_mm=config.get("data", {}).get("rainfall_threshold_mm", .1),
        )
        missing = sorted(set(features) - set(frame.columns))
        if missing:
            raise ValueError(f"Input data is missing model features: {', '.join(missing)}")
        frame = frame.dropna(subset=features).copy()
        if frame.empty:
            raise ValueError("No rows have complete model features")
        preprocessor = meta.get("preprocessor")
        if preprocessor is None:
            raise ValueError("Hourly model bundle is missing its fitted preprocessor")
        X = preprocessor.transform(frame[features])
        groups = frame.grid_id.to_numpy() if isinstance(model, LSTMModel) and "grid_id" in frame else None
        issued = pd.to_datetime(frame.timestamp, utc=True).reset_index(drop=True)
        horizon = int(config.get("forecast", {}).get("horizon_hours", 1))
        forecast_dates = issued + pd.to_timedelta(horizon, unit="h")
        target = meta.get("target", "target_rain_event")
        threshold = meta.get("threshold")

    if isinstance(model, LSTMModel):
        prediction = model.predict(X, groups=groups)
    else:
        prediction = model.predict(X)
    output = pd.DataFrame({
        "issued_at": issued,
        "forecast_for": forecast_dates,
        "model": path.stem,
        "prediction": np.asarray(prediction).reshape(-1),
    })
    is_classifier = "event" in str(target) or (
        not target and getattr(model, "task", None) == "classification"
    )
    if is_classifier:
        probability = _positive_probability(model, X, groups)
        output["rain_probability"] = probability
        if threshold is not None:
            output["prediction"] = (probability >= float(threshold)).astype(int)
    if daily_bundle:
        output["forecast_rainfall_mm"] = np.maximum(0, np.asarray(prediction).reshape(-1)) if not is_classifier else np.nan
    else:
        output.insert(0, "grid_id", frame.grid_id.to_numpy() if "grid_id" in frame else "single_location")

    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(target_path, index=False)
    print(f"Wrote {len(output)} predictions to {target_path}")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default="models/random_forest_classifier.joblib")
    parser.add_argument("--input", default="data/synthetic/weather.csv")
    parser.add_argument("--output", default="data/processed/predictions.csv")
    args = parser.parse_args()
    predict(args.model, args.input, args.output)


if __name__ == "__main__":
    main()
