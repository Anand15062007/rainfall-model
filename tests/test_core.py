import numpy as np
import pandas as pd
import pytest
from app.grid.generate import generate_grid
from app.data.synthetic import generate_synthetic
from app.features.build import build_features,feature_columns
from app.preprocessing.clean import Preprocessor
from app.training.splits import chronological_split,spatial_split,assert_causal_features
from app.evaluation.metrics import classification_metrics,regression_metrics
from app.api.providers import MockWeatherProvider

def sample():
    g=generate_grid("x",(-.03,-.03,.03,.03),1)
    return g,generate_synthetic(g,periods=80,seed=3)

def test_grid_metric_and_shape():
    g=generate_grid("x",(-.03,-.03,.03,.03),1)
    assert len(g)>0 and g.grid_id.is_unique and (g.grid_size_km==1).all()
    assert g.latitude_center.between(-.03,.03).any()

def test_provider_and_synthetic():
    g,d=sample(); assert len(d)==len(g)*80; assert MockWeatherProvider(d).fetch().timestamp.notna().all()

def test_causal_features_targets():
    g,d=sample(); f=build_features(d,lags=(1,3),rolling_windows=(3,),horizon_hours=1)
    one=d[d.grid_id==g.grid_id.iloc[0]].sort_values('timestamp').reset_index(drop=True)
    ff=f[f.grid_id==one.grid_id.iloc[0]].sort_values('timestamp').reset_index(drop=True)
    assert ff.loc[0,'temperature_lag_1']!=ff.loc[0,'temperature_lag_1']
    assert ff.loc[10,'temperature_lag_1']==pytest.approx(one.loc[9,'temperature'])
    assert ff.loc[10,'target_rainfall_mm']==pytest.approx(one.loc[11,'rainfall_1h'])
    f3=build_features(d,lags=(1,),rolling_windows=(2,),horizon_hours=3)
    f3=f3[f3.grid_id==one.grid_id.iloc[0]].sort_values('timestamp').reset_index(drop=True)
    assert f3.loc[10,'target_rainfall_mm']==pytest.approx(one.loc[11:13,'rainfall_1h'].sum())
    assert_causal_features(ff,feature_columns(ff))

def test_preprocessing_is_train_fit_only():
    pp=Preprocessor().fit(pd.DataFrame({'x':[0.,2.,np.nan]})); assert pp.transform(pd.DataFrame({'x':[100.]})).shape[0]==1

def test_splits_and_metrics():
    _,d=sample(); d.timestamp=pd.to_datetime(d.timestamp,utc=True)
    tr,va,te=chronological_split(d,.6,.2); assert (va.timestamp.min()-tr.timestamp.max()).total_seconds()>=7200
    assert (te.timestamp.min()-va.timestamp.max()).total_seconds()>=7200
    a,b,c=spatial_split(d,.5,.25); assert not(set(a.grid_id)&set(b.grid_id)); assert not(set(a.grid_id)&set(c.grid_id))
    m=classification_metrics([0,1,1,0],[0,1,0,0],[.1,.8,.4,.2]); assert m['csi']==.5 and 'pr_auc' in m and 'ets' in m
    assert regression_metrics([0,1],[0,2])['mae']==.5

def test_lstm_sequences_respect_grid_boundaries_and_persistence(tmp_path):
    torch=pytest.importorskip("torch")
    from app.models.lstm import LSTMModel
    X=np.arange(20,dtype=np.float32).reshape(10,2); y=np.array([0,1]*5,dtype=np.float32); groups=np.array(["a"]*5+["b"]*5)
    seq,labels=LSTMModel.sequences(X,y,3,groups)
    assert seq.shape==(10,3,2) and len(labels)==10
    assert np.all(seq[5:,0,0]>=10) # no history from previous grid
    model=LSTMModel(sequence_length=3,hidden_size=4,epochs=2,patience=1,batch_size=8).fit(X,y,groups=groups)
    path=tmp_path/"lstm.pt"; model.save(path); loaded=LSTMModel.load(path)
    assert loaded.predict(X,groups=groups).shape==(10,)

def test_feature_whitelist_is_applied():
    from app.features.build import feature_columns
    data=pd.DataFrame({"timestamp":pd.date_range("2024-01-01",periods=8,freq="h",tz="UTC"),"grid_id":"a","latitude":1.,"longitude":2.,"temperature":np.arange(8.),"relative_humidity":50.,"rainfall_1h":0.,"pressure":1000.})
    f=build_features(data,lags=(1,),rolling_windows=(2,),horizon_hours=1,base_features=["temperature","rainfall_1h"])
    columns=feature_columns(f,["temperature","rainfall_1h"])
    assert "temperature_lag_1" in columns and "rainfall_1h_lag_1" in columns
    assert "pressure" not in columns and "pressure_lag_1" not in columns

def test_era5_netcdf_conversion_units_and_native_cells(tmp_path):
    xr=pytest.importorskip("xarray")
    from app.data.era5 import netcdf_to_observations,grid_catalog_from_observations
    t=pd.date_range("2024-01-01",periods=2,freq="h")
    coords={"valid_time":t,"latitude":[38.75],"longitude":[-77.0]}
    shape=(2,1,1)
    ds=xr.Dataset({"t2m":(("valid_time","latitude","longitude"),np.full(shape,280.)),"d2m":(("valid_time","latitude","longitude"),np.full(shape,275.)),"msl":(("valid_time","latitude","longitude"),np.full(shape,101325.)),"u10":(("valid_time","latitude","longitude"),np.full(shape,3.)),"v10":(("valid_time","latitude","longitude"),np.full(shape,4.)),"tcc":(("valid_time","latitude","longitude"),np.full(shape,.5)),"tp":(("valid_time","latitude","longitude"),np.full(shape,.002)),"tcwv":(("valid_time","latitude","longitude"),np.full(shape,20.)),"cape":(("valid_time","latitude","longitude"),np.full(shape,500.)),"cin":(("valid_time","latitude","longitude"),np.full(shape,10.)),"blh":(("valid_time","latitude","longitude"),np.full(shape,1000.)),"z":(("valid_time","latitude","longitude"),np.full(shape,98.0665))},coords=coords)
    path=tmp_path/"small.nc"; ds.to_netcdf(path)
    obs=netcdf_to_observations(path,bbox=(-78,38,-76,40)); assert obs.rainfall_1h.tolist()==pytest.approx([2.,2.]); assert obs.temperature.iloc[0]==pytest.approx(6.85); assert obs.pressure.iloc[0]==pytest.approx(1013.25); assert obs.cloud_cover.iloc[0]==pytest.approx(50.)
    catalog=grid_catalog_from_observations(obs); assert catalog.grid_id.iloc[0].startswith("ERA5_") and catalog.grid_size_km.iloc[0]>20

def test_validation_threshold_never_uses_test():
    from app.evaluation.thresholds import select_threshold
    threshold=select_threshold([0,0,1,1],[.1,.3,.6,.8],"f1")
    assert .3<=threshold<=.8


def test_hourly_regularization_preserves_missing_as_nan():
    from app.preprocessing.clean import regularize_hourly
    d=pd.DataFrame({"timestamp":pd.to_datetime(["2024-01-01T00:00Z","2024-01-01T02:00Z"]),"grid_id":["a","a"],"latitude":[1.,1.],"longitude":[2.,2.],"rainfall_1h":[0.,2.]})
    r=regularize_hourly(d)
    assert len(r)==3 and pd.isna(r.loc[1,"rainfall_1h"]) and r.timestamp.diff().dropna().eq(pd.Timedelta(hours=1)).all()
