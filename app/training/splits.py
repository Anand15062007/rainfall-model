"""Chronological, rolling-origin and spatial split utilities."""
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GroupKFold

def chronological_split(df, train=.6, validation=.2, time_col="timestamp", gap=1):
    if train<=0 or validation<=0 or train+validation>=1: raise ValueError("split fractions must be positive and sum below 1")
    unique=np.array(sorted(df[time_col].unique())); a=int(len(unique)*train); b=int(len(unique)*(train+validation))
    if gap < 0 or a <= gap or b-a <= gap: raise ValueError("split too small for requested purge gap")
    return tuple(df[df[time_col].isin(s)].copy() for s in (unique[:a-gap],unique[a:b-gap],unique[b:]))

def time_series_folds(df, folds=5, time_col="timestamp", gap=1):
    times=np.array(sorted(df[time_col].unique())); splitter=TimeSeriesSplit(n_splits=folds)
    for tr,va in splitter.split(times):
        tr=tr[:-gap] if gap else tr
        if len(tr)==0: continue
        yield df[df[time_col].isin(times[tr])].copy(),df[df[time_col].isin(times[va])].copy()

def spatial_split(df, train=.5, validation=.3, seed=42, group_col="grid_id"):
    if train<=0 or validation<=0 or train+validation>=1: raise ValueError("split fractions must sum below 1")
    groups=np.array(sorted(df[group_col].unique())); rng=np.random.default_rng(seed); rng.shuffle(groups)
    a=int(len(groups)*train); b=int(len(groups)*(train+validation)); sets=np.split(groups,[a,b])
    return tuple(df[df[group_col].isin(s)].copy() for s in sets)

def group_folds(df, folds=5, group_col="grid_id"):
    cv=GroupKFold(n_splits=folds)
    for tr,va in cv.split(df,groups=df[group_col]): yield df.iloc[tr].copy(),df.iloc[va].copy()

def assert_causal_features(df, feature_cols):
    banned={"rain_event","rainfall_amount_mm","target_rain_event","target_rainfall_mm"}
    leak=banned.intersection(feature_cols)
    if leak: raise ValueError(f"target leakage in feature list: {sorted(leak)}")
