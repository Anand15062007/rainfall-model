import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.prediction.predict import predict


def test_daily_pipeline_bundle_predicts_rain_probability(tmp_path):
    input_path = tmp_path / "openweather_hourly.csv"
    model_path = tmp_path / "daily_classifier.joblib"
    output_path = tmp_path / "predictions.csv"

    timestamps = pd.date_range("2025-01-01", periods=5 * 24, freq="h", tz="UTC")
    hours = np.arange(len(timestamps))
    weather = pd.DataFrame({
        "timestamp": timestamps,
        "temperature": 8 + hours / 24,
        "feels_like": 7 + hours / 24,
        "relative_humidity": 45 + hours % 24,
        "pressure": 1005 + hours % 4,
        "wind_speed": 2 + hours % 3,
        "wind_direction": 160,
        "cloud_cover": 30 + hours % 40,
        "rainfall_1h_mm": 0,
        "weather_main": "Clouds",
    })
    weather.to_csv(input_path, index=False)

    features = ["temperature_mean", "relative_humidity_mean"]
    model = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression()),
    ])
    model.fit(pd.DataFrame([[0, 0], [1, 1], [2, 0], [3, 1]], columns=features), [0, 0, 1, 1])
    joblib.dump({
        "model": model,
        "features": features,
        "target": "target_rain_event",
        "threshold": 0.5,
        "source": "OpenWeather + CHIRPS",
        "forecast_horizon_days": 1,
    }, model_path)

    result = predict(model_path, input_path, output_path)
    assert len(result) == 1
    assert result.forecast_for.iloc[0] == pd.Timestamp("2025-01-06", tz="UTC")
    assert result.rain_probability.between(0, 1).all()
    assert output_path.exists()
