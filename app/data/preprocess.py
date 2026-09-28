"""Write causally engineered features from a local CSV."""
import argparse
from app.config.settings import load_config
from app.api.providers import CSVWeatherProvider
from app.features.build import build_features
p=argparse.ArgumentParser(); p.add_argument("path",nargs="?",default="data/synthetic/weather.csv"); a=p.parse_args(); c=load_config()
df=CSVWeatherProvider(a.path).fetch(); out=build_features(df,c['features']['lags'],c['features']['rolling_windows'],c['features']['rolling_columns'],c['forecast']['horizon_hours'],c['data']['rainfall_threshold_mm'],c['features'].get('base_features')); out.to_csv("data/processed/features.csv",index=False); print(f"Wrote {len(out)} engineered rows")
