"""End-to-end offline demo: grid -> synthetic observations -> causal features -> temporal holdout -> classical baselines."""
import json
import os
from pathlib import Path
# Keep BLAS/OpenMP thread pools bounded for stable mixed sklearn/XGBoost/PyTorch runs.
for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")
import numpy as np
import pandas as pd
from app.config.settings import load_config,seed_everything
from app.grid.generate import generate_grid
from app.data.synthetic import generate_synthetic
from app.data.era5 import grid_catalog_from_observations
from app.api.providers import CSVWeatherProvider
from app.features.build import build_features,feature_columns
from app.preprocessing.clean import Preprocessor,regularize_hourly
from app.training.splits import chronological_split,assert_causal_features,time_series_folds
from app.models.classical import LogisticRegressionModel,RandomForestModel,XGBoostModel
from app.evaluation.metrics import classification_metrics,regression_metrics
from app.evaluation.thresholds import select_threshold
from sklearn.inspection import permutation_importance
from app.evaluation.tracking import log_experiment
from app.visualization.plots import rainfall_series,comparison_plot,grid_map
from app.database.store import Store

def run(config_path="configs/default.yaml", include_optional=False, input_path=None):
    c=load_config(config_path); seed_everything(c["random_seed"]); Path("data/processed").mkdir(parents=True,exist_ok=True)
    if input_path:
        df=CSVWeatherProvider(input_path).fetch(); df["timestamp"]=pd.to_datetime(df.timestamp,utc=True)
        required={"rainfall_1h","grid_id","latitude","longitude"}
        if required-set(df.columns): raise ValueError(f"input dataset lacks required fields: {sorted(required-set(df.columns))}")
        df["rain_event"]=(df.rainfall_1h>=c['data']['rainfall_threshold_mm']).astype("int8"); df["rainfall_amount_mm"]=df.rainfall_1h
        g=grid_catalog_from_observations(df); is_synthetic=False; dataset_version="era5-single-levels"
    else:
        g=generate_grid(c['city']['name'],tuple(c['city']['bbox']),c['city']['grid_size_km'])
        df=generate_synthetic(g,c['data']['start'],c['data']['periods'],c['data']['frequency'],c['random_seed'])
        df.timestamp=pd.to_datetime(df.timestamp,utc=True); is_synthetic=True; dataset_version="synthetic-v1"
        df.to_csv("data/synthetic/weather.csv",index=False)
    g.to_csv("data/processed/grids.csv",index=False)
    df=regularize_hourly(df)
    base_features=c['features'].get('base_features')
    f=build_features(df,c['features']['lags'],c['features']['rolling_windows'],c['features']['rolling_columns'],c['forecast']['horizon_hours'],c['data']['rainfall_threshold_mm'],base_features)
    f=f.dropna(subset=["target_rain_event","target_rainfall_mm"]).reset_index(drop=True)
    train,val,test=chronological_split(f,c['split']['train'],c['split']['validation'],gap=c['forecast']['horizon_hours'])
    cols=feature_columns(f,base_features); assert_causal_features(f,cols)
    forbidden={"rain_event","rainfall_amount_mm","target_rain_event","target_rainfall_mm"}
    cols=[x for x in cols if x not in forbidden]
    scaler=Preprocessor(c['preprocessing']['missing'],c['preprocessing']['scaler']); Xtr=scaler.fit_transform(train[cols]); Xv=scaler.transform(val[cols]); Xt=scaler.transform(test[cols])
    results={}; predictions={}; specs=[]
    if c['models'].get('logistic_regression',True): specs.append(("logistic_regression",lambda task:LogisticRegressionModel(**c.get('model_params',{}).get('logistic_regression',{})),"classification"))
    if c['models'].get('random_forest',True):
        specs.extend([("random_forest_classifier",lambda task:RandomForestModel(task,random_state=c['random_seed'],**c.get('model_params',{}).get('random_forest',{})),"classification"),("random_forest_regressor",lambda task:RandomForestModel(task,random_state=c['random_seed'],**c.get('model_params',{}).get('random_forest',{})),"regression")])
    if c['models'].get('xgboost',False):
        for task in ("classification","regression"):
            try: XGBoostModel(task)
            except ImportError as e: print(f"Skipping XGBoost ({task}): {e}")
            else: specs.append((f"xgboost_{task}",lambda t=task:XGBoostModel(t,random_state=c['random_seed'],**c.get('model_params',{}).get('xgboost',{})),task))
    for name,factory,task in specs:
        ycol="target_rain_event" if task=="classification" else "target_rainfall_mm"
        ytr=train[ycol].astype(int if task=="classification" else float).to_numpy(); yt=test[ycol].astype(int if task=="classification" else float).to_numpy()
        model=factory(task); model.fit(Xtr,ytr)
        threshold=None
        if task=="classification":
            val_prob=model.predict_proba(Xv); threshold=select_threshold(val[ycol].astype(int).to_numpy(),val_prob,c['models'].get('selection_metric','f1'))
            test_prob=model.predict_proba(Xt); pred=(test_prob>=threshold).astype(int); metrics=classification_metrics(yt,pred,test_prob)
        else:
            pred=model.predict(Xt); test_prob=None; metrics=regression_metrics(yt,pred)
        results[name]=metrics; predictions[name]=pred
        if hasattr(model,"feature_importances_"):
            Path("experiments").mkdir(exist_ok=True)
            pd.DataFrame({"feature":scaler.feature_names_,"importance":model.feature_importances_}).sort_values("importance",ascending=False).to_csv(f"experiments/feature_importance_{name}.csv",index=False)
        if name.startswith("xgboost_"):
            score=c['models'].get('selection_metric','f1') if task=="classification" else "neg_mean_absolute_error"
            if score=="csi": score="f1"
            perm=permutation_importance(model.estimator,Xv,val[ycol].astype(int if task=="classification" else float).to_numpy(),scoring=score,n_repeats=2,random_state=c['random_seed'],n_jobs=1,max_samples=.5)
            pd.DataFrame({"feature":scaler.feature_names_,"validation_importance_mean":perm.importances_mean,"validation_importance_std":perm.importances_std}).sort_values("validation_importance_mean",ascending=False).to_csv(f"experiments/validation_permutation_importance_{name}.csv",index=False)
        Path("models").mkdir(exist_ok=True)
        model.save(f"models/{name}.joblib",{"features":cols,"target":ycol,"config":c,"preprocessor":scaler,"threshold":threshold,"synthetic":is_synthetic})
        log_experiment({"dataset_version":dataset_version,"features":cols,"grid_size_km":float(g.grid_size_km.mean()),"forecast_horizon_hours":c['forecast']['horizon_hours'],"selection_threshold":threshold,"model":name,"hyperparameters":model.estimator.get_params(),"training_period":[str(train.timestamp.min()),str(train.timestamp.max())],"validation_period":[str(val.timestamp.min()),str(val.timestamp.max())],"test_period":[str(test.timestamp.min()),str(test.timestamp.max())],"metrics":metrics,"random_seed":c['random_seed'],"fit_rows":len(train),"test_rows":len(test)})
    if c['models'].get('lstm',False):
        from app.models.lstm import LSTMModel
        lc=c.get('lstm',{})
        for task in ("classification","regression"):
            ycol="target_rain_event" if task=="classification" else "target_rainfall_mm"
            ytr=train[ycol].astype(int if task=="classification" else float).to_numpy()
            yv=val[ycol].astype(int if task=="classification" else float).to_numpy()
            yt=test[ycol].astype(int if task=="classification" else float).to_numpy()
            model=LSTMModel(task=task,sequence_length=lc.get('sequence_length',24),hidden_size=lc.get('hidden_size',32),epochs=lc.get('epochs',15),patience=lc.get('patience',3),batch_size=lc.get('batch_size',64),seed=c['random_seed'])
            model.fit(Xtr,ytr,Xv,yv,groups=train.grid_id.to_numpy(),val_groups=val.grid_id.to_numpy())
            threshold=None
            if task=="classification":
                val_prob=model.predict_proba(Xv,groups=val.grid_id.to_numpy()); threshold=select_threshold(yv,val_prob,c['models'].get('selection_metric','f1'))
                test_prob=model.predict_proba(Xt,groups=test.grid_id.to_numpy()); pred=(test_prob>=threshold).astype(int); metrics=classification_metrics(yt,pred,test_prob)
            else:
                pred=model.predict(Xt,groups=test.grid_id.to_numpy()); metrics=regression_metrics(yt,pred)
            name=f"lstm_{task}"; results[name]=metrics; predictions[name]=pred
            meta={"features":cols,"target":ycol,"config":c,"preprocessor":scaler,"threshold":threshold,"synthetic":is_synthetic}
            model.save(f"models/{name}.pt",meta)
            log_experiment({"dataset_version":dataset_version,"features":cols,"grid_size_km":float(g.grid_size_km.mean()),"forecast_horizon_hours":c['forecast']['horizon_hours'],"selection_threshold":threshold,"model":name,"hyperparameters":lc,"training_period":[str(train.timestamp.min()),str(train.timestamp.max())],"validation_period":[str(val.timestamp.min()),str(val.timestamp.max())],"test_period":[str(test.timestamp.min()),str(test.timestamp.max())],"metrics":metrics,"random_seed":c['random_seed'],"fit_rows":len(train),"test_rows":len(test)})
    Path("experiments").mkdir(exist_ok=True)
    Path("experiments/latest_metrics.json").write_text(json.dumps(results,indent=2,allow_nan=True),encoding="utf-8")
    pd.DataFrame([{"model":k,**v} for k,v in results.items()]).to_csv("experiments/model_comparison.csv",index=False)
    rainfall_series(df,"experiments/rainfall_timeseries.png"); comparison_plot(results,"experiments/model_comparison.png")
    # Generate a sample map using classifier predictions aligned to the final forecast time.
    if predictions:
        valid=test.copy(); key="random_forest_regressor" if "random_forest_regressor" in predictions else next(iter(predictions))
        if key.endswith("regressor"): rain=np.maximum(0,predictions[key]); prob=np.zeros(len(rain))
        else: rain=np.zeros(len(test)); prob=predictions[key]
        sample=pd.DataFrame({"grid_id":valid.grid_id,"predicted_rainfall_mm":rain,"rain_probability":prob}).groupby("grid_id").last().reset_index()
        grid_map(g,sample)
    Store().put_frame("grids",g)
    # Rolling-origin CV is available independently and run as diagnostic splits on the feature data.
    fold_count=min(c['cross_validation']['folds'],max(2,len(f.timestamp.unique())//3))
    cv_folds=sum(1 for _ in time_series_folds(f,fold_count,gap=c['forecast']['horizon_hours'])) if fold_count>=2 else 0
    return {"grids":len(g),"observations":len(df),"train_rows":len(train),"validation_rows":len(val),"test_rows":len(test),"rolling_origin_folds":cv_folds,"models":results,"synthetic_data":is_synthetic}

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/default.yaml"); p.add_argument("--data",help="standardized hourly CSV; omit to use synthetic data"); a=p.parse_args(); print(json.dumps(run(a.config,input_path=a.data),indent=2))
