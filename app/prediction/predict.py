"""Load a persisted model bundle and use its exact feature order/preprocessor."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from app.features.build import build_features
from app.models.lstm import LSTMModel

def predict(model_path,input_path):
    path=Path(model_path)
    if path.suffix==".pt":
        import torch
        payload=torch.load(path,map_location="cpu",weights_only=False)
        model=LSTMModel.load(path); meta=payload.get("metadata",{})
    else:
        bundle=joblib.load(path); model=bundle["model"]; meta=bundle["metadata"]
    if not meta.get("features") or "preprocessor" not in meta:
        raise ValueError("Model bundle must include feature ordering and fitted preprocessing state")
    config=meta.get("config",{}); feature_cfg=config.get("features",{})
    df=pd.read_csv(input_path,parse_dates=["timestamp"])
    df=build_features(df,lags=feature_cfg.get("lags",(1,3,6,12,24)),rolling_windows=feature_cfg.get("rolling_windows",(3,6,24)),rolling_columns=feature_cfg.get("rolling_columns",("temperature","relative_humidity","rainfall_1h")),horizon_hours=config.get("forecast",{}).get("horizon_hours",1),threshold_mm=config.get("data",{}).get("rainfall_threshold_mm",.1))
    features=meta["features"]; df=df.dropna(subset=[c for c in features if c in df.columns]).copy()
    X=meta["preprocessor"].transform(df[features])
    groups=df.grid_id.to_numpy() if isinstance(model,LSTMModel) else None
    pred=model.predict(X,groups=groups) if isinstance(model,LSTMModel) else model.predict(X)
    out=pd.DataFrame({"grid_id":df.grid_id.to_numpy(),"timestamp":df.timestamp.to_numpy(),"model":path.stem,"prediction":pred})
    if model.task=="classification":
        out["rain_probability"]=model.predict_proba(X,groups=groups) if isinstance(model,LSTMModel) else model.predict_proba(X)
        if meta.get("threshold") is not None: out["prediction"]=(out.rain_probability>=meta["threshold"]).astype(int)
    out.to_csv("data/processed/predictions.csv",index=False)
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument("model",nargs="?",default="models/random_forest_classifier.joblib"); p.add_argument("--input",default="data/synthetic/weather.csv"); a=p.parse_args()
    result=predict(a.model,a.input); print(f"Wrote {len(result)} predictions")
if __name__=="__main__": main()
