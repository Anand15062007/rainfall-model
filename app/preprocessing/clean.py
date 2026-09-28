"""Train-fitted numeric imputation/scaling; targets and identifiers are excluded."""
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class Preprocessor:
    def __init__(self, strategy="median", scaler="standard", indicators=True):
        self.imputer=SimpleImputer(strategy=strategy,add_indicator=indicators,keep_empty_features=True)
        self.scaler=StandardScaler() if scaler=="standard" else MinMaxScaler() if scaler=="minmax" else None
        self.columns=None
    def fit(self, X):
        self.columns=list(X.columns); z=self.imputer.fit_transform(X[self.columns])
        if self.scaler: self.scaler.fit(z)
        self.feature_names_=self.imputer.get_feature_names_out(self.columns).tolist()
        return self
    def transform(self, X):
        z=self.imputer.transform(X[self.columns]); return self.scaler.transform(z) if self.scaler else z
    def fit_transform(self,X): return self.fit(X).transform(X)

def regularize_hourly(frame):
    """Insert explicit missing-hour rows per grid so lag/target shifts mean hours, not records."""
    import pandas as pd
    if frame.empty: return frame.copy()
    data=frame.copy(); data["timestamp"]=pd.to_datetime(data.timestamp,utc=True)
    if data.duplicated(["grid_id","timestamp"]).any():
        data=data.sort_values("timestamp").drop_duplicates(["grid_id","timestamp"],keep="last")
    parts=[]
    for grid_id,group in data.groupby("grid_id",sort=False):
        group=group.sort_values("timestamp").set_index("timestamp")
        hours=pd.date_range(group.index.min().floor("h"),group.index.max().ceil("h"),freq="h",tz="UTC")
        group=group.reindex(hours); group.index.name="timestamp"; group["grid_id"]=grid_id
        for c in ("latitude","longitude"):
            if c in group: group[c]=group[c].ffill().bfill()
        parts.append(group.reset_index())
    return pd.concat(parts,ignore_index=True).sort_values(["grid_id","timestamp"]).reset_index(drop=True)
