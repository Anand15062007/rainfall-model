"""Weather provider boundary. Live APIs are deliberately unimplemented pending credentials."""
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd

REQUIRED=["timestamp","grid_id","latitude","longitude"]
class WeatherDataProvider(ABC):
    @abstractmethod
    def fetch(self,**kwargs)->pd.DataFrame: ...
    @staticmethod
    def validate(frame):
        missing=set(REQUIRED)-set(frame.columns)
        if missing: raise ValueError(f"missing required observation columns: {sorted(missing)}")
        out=frame.copy(); out["timestamp"]=pd.to_datetime(out.timestamp,utc=True,errors="coerce")
        if out.timestamp.isna().any(): raise ValueError("invalid timestamps in provider response")
        return out
class CSVWeatherProvider(WeatherDataProvider):
    def __init__(self,path): self.path=Path(path)
    def fetch(self,**kwargs): return self.validate(pd.read_csv(self.path))
class MockWeatherProvider(WeatherDataProvider):
    def __init__(self,frame): self.frame=frame
    def fetch(self,**kwargs): return self.validate(self.frame.copy())
class APIWeatherProvider(WeatherDataProvider):
    """Adapter placeholder. Provider-specific auth/endpoint/schema mapping comes later."""
    def __init__(self,base_url=None,api_key_env="WEATHER_API_KEY",timeout=10,retries=3):
        self.base_url=base_url; self.api_key_env=api_key_env; self.timeout=timeout; self.retries=retries
    def fetch(self,**kwargs):
        raise NotImplementedError("Live provider awaits the selected vendor API and credentials; local providers work now.")
