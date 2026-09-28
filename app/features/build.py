"""Causal feature and future-target generation, grouped by grid."""
import numpy as np
import pandas as pd
TARGETS={"rain_event","rainfall_amount_mm","target_rain_event","target_rainfall_mm"}
ID_COLS={"timestamp","grid_id","latitude","longitude"}

def build_features(frame, lags=(1,3,6,12,24), rolling_windows=(3,6,24), rolling_columns=("temperature","relative_humidity","rainfall_1h"), horizon_hours=1, threshold_mm=.1, base_features=None):
    df=frame.copy(); df["timestamp"]=pd.to_datetime(df.timestamp,utc=True); df=df.sort_values(["grid_id","timestamp"]).reset_index(drop=True)
    t=df.timestamp
    temporal={"hour":t.dt.hour,"day":t.dt.day,"day_of_week":t.dt.dayofweek,"month":t.dt.month,"day_of_year":t.dt.dayofyear}
    temporal["season"]=(t.dt.month%12//3).astype(int)
    temporal["hour_sin"]=np.sin(2*np.pi*t.dt.hour/24); temporal["hour_cos"]=np.cos(2*np.pi*t.dt.hour/24)
    temporal["doy_sin"]=np.sin(2*np.pi*t.dt.dayofyear/365.25); temporal["doy_cos"]=np.cos(2*np.pi*t.dt.dayofyear/365.25)
    new_features=dict(temporal)
    groups=df.groupby("grid_id",sort=False)
    # Only physical weather fields get lags; do not recursively lag generated or ID columns.
    source=[c for c in frame.select_dtypes(include="number").columns if c not in ID_COLS|TARGETS]
    if base_features is not None: source=[c for c in base_features if c in df.columns]
    for col in source:
        for lag in lags: new_features[f"{col}_lag_{lag}"]=groups[col].shift(lag).to_numpy()
        if col in rolling_columns:
            shifted=groups[col].shift(1)
            roll=shifted.groupby(df.grid_id).rolling(max(rolling_windows),min_periods=1)
            # Windows are calculated separately to keep each result aligned to the original index.
            for w in rolling_windows:
                r=shifted.groupby(df.grid_id).rolling(w,min_periods=1)
                for stat in ("mean","std","min","max"):
                    new_features[f"{col}_rolling_{stat}_{w}"]=getattr(r,stat)().reset_index(level=0,drop=True).sort_index().to_numpy()
    df=pd.concat([df,pd.DataFrame(new_features,index=df.index)],axis=1)
    if horizon_hours < 1: raise ValueError("horizon_hours must be a positive integer")
    # Forecast accumulated rain over the next H hourly intervals, excluding hour t.
    future=sum(groups["rainfall_1h"].shift(-step) for step in range(1,horizon_hours+1))
    future=future.where(groups["rainfall_1h"].shift(-horizon_hours).notna())
    df["target_rainfall_mm"]=future.to_numpy()
    df["target_rain_event"]=(future>=threshold_mm).where(future.notna()).to_numpy()
    return df

def feature_columns(df, selected_features=None):
    numeric=[c for c in df.select_dtypes(include="number").columns if c not in TARGETS]
    if selected_features is None: return numeric
    engineered={"hour","day","day_of_week","month","day_of_year","season","hour_sin","hour_cos","doy_sin","doy_cos"}
    wanted=set(selected_features)|engineered|{"latitude","longitude"}
    wanted.update(c for c in numeric if any(c.startswith(f"{base}_lag_") or c.startswith(f"{base}_rolling_") for base in selected_features))
    return [c for c in numeric if c in wanted]
