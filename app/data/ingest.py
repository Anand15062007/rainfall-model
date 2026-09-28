"""CSV-backed local ingestion CLI."""
import argparse
from app.api.providers import CSVWeatherProvider
from app.database.store import Store
p=argparse.ArgumentParser(); p.add_argument("path",nargs="?",default="data/synthetic/weather.csv"); a=p.parse_args()
df=CSVWeatherProvider(a.path).fetch(); Store().put_frame("weather_observations",df); print(f"Ingested {len(df)} validated observations")
