"""Train and evaluate next-day rainfall models using OpenWeather + CHIRPS."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from app.config.settings import load_config, seed_everything
from app.data.openweather_history import hourly_to_daily
from app.evaluation.metrics import classification_metrics, regression_metrics
from app.evaluation.thresholds import select_threshold


def _temporal_split(frame: pd.DataFrame, train_fraction=.6, validation_fraction=.2, gap_days=1):
    dates = np.array(sorted(frame.date.unique()))
    a, b = int(len(dates) * train_fraction), int(len(dates) * (train_fraction + validation_fraction))
    if a <= gap_days or b - a <= gap_days or b >= len(dates):
        raise ValueError(f"Not enough daily records ({len(dates)}) for chronological train/validation/test split")
    train_dates = dates[:a-gap_days]
    val_dates = dates[a:b-gap_days]
    test_dates = dates[b:]
    return tuple(frame[frame.date.isin(x)].copy() for x in (train_dates, val_dates, test_dates))


def _make_table(owm_hourly: pd.DataFrame, chirps_daily: pd.DataFrame, threshold_mm: float):
    owm = hourly_to_daily(owm_hourly)
    chirps = chirps_daily.copy()
    chirps["date"] = pd.to_datetime(chirps["date"], utc=True).dt.floor("D")
    chirps["rainfall_mm"] = pd.to_numeric(chirps["rainfall_mm"], errors="coerce")
    chirps = chirps.dropna(subset=["rainfall_mm"]).drop_duplicates("date").sort_values("date")
    targets = chirps[["date", "rainfall_mm"]].rename(columns={"date": "target_date", "rainfall_mm": "target_rainfall_mm"})
    owm["date"] = pd.to_datetime(owm["date"], utc=True).dt.floor("D")
    owm["target_date"] = owm.date + pd.Timedelta(days=1)
    frame = owm.merge(targets, on="target_date", how="inner", validate="one_to_one")
    frame["target_rain_event"] = (frame.target_rainfall_mm >= threshold_mm).astype("int8")
    # Exclude partial history days. Missing source fields stay missing and are
    # imputed by transformers fitted on the training window only.
    frame = frame[frame.owm_observation_count >= 18].copy()
    frame = frame.sort_values("date").reset_index(drop=True)
    if len(frame) < 120:
        raise ValueError(f"Only {len(frame)} aligned complete daily rows. Need at least 120; check OWM history coverage and CHIRPS dates.")
    return frame


def _preprocessor(numeric, categorical):
    num = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler())])
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("numeric", num, numeric), ("categorical", cat, categorical)],
                             remainder="drop", verbose_feature_names_out=False)


def _model_specs(seed, n_train, y_train, include_xgboost=True):
    specs = {
        "logistic_regression": ("classification", LogisticRegression(max_iter=1500, class_weight="balanced", C=.5)),
        "random_forest_classifier": ("classification", RandomForestClassifier(n_estimators=350, min_samples_leaf=3,
            max_features=.8, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)),
        "random_forest_regressor": ("regression", TransformedTargetRegressor(
            regressor=RandomForestRegressor(n_estimators=350, min_samples_leaf=3, max_features=.8,
                n_jobs=-1, random_state=seed), func=np.log1p, inverse_func=np.expm1, check_inverse=False)),
    }
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError:
        if include_xgboost:
            print("XGBoost not installed; skipping. Install with: pip install -e '.[xgboost]'")
    else:
        positives = max(int(np.sum(y_train)), 1)
        negatives = max(len(y_train) - positives, 1)
        specs["xgboost_classifier"] = ("classification", XGBClassifier(
            n_estimators=350, max_depth=4, learning_rate=.035, subsample=.85,
            colsample_bytree=.85, reg_lambda=2, scale_pos_weight=negatives / positives,
            eval_metric="logloss", n_jobs=1, random_state=seed))
        specs["xgboost_regressor"] = ("regression", TransformedTargetRegressor(
            regressor=XGBRegressor(n_estimators=350, max_depth=4, learning_rate=.035,
                subsample=.85, colsample_bytree=.85, reg_lambda=2, objective="reg:squarederror",
                n_jobs=1, random_state=seed), func=np.log1p, inverse_func=np.expm1, check_inverse=False))
    return specs


def run(config_path="configs/default.yaml", input_dir="data/raw", output_dir="experiments"):
    cfg = load_config(config_path)
    seed = int(cfg.get("random_seed", 42))
    seed_everything(seed)
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    owm_path = input_dir / "openweather_hourly.csv"
    chirps_path = input_dir / "chirps_v3_daily.csv"
    if not owm_path.exists() or not chirps_path.exists():
        raise FileNotFoundError(f"Expected both {owm_path} and {chirps_path}; download each dataset first (see README live-data commands).")
    threshold = float(cfg["data"].get("daily_rainfall_threshold_mm", 1.0))
    table = _make_table(pd.read_csv(owm_path), pd.read_csv(chirps_path), threshold)
    numeric = [c for c in table.select_dtypes(include="number").columns
               if c not in {"target_rainfall_mm", "target_rain_event", "owm_observation_count", "day_of_year"}]
    categorical = [c for c in ("weather_main",) if c in table.columns]
    features = numeric + categorical
    forbidden = {"target_rainfall_mm", "target_rain_event", "rainfall_mm", "date", "target_date"}
    features = [c for c in features if c not in forbidden]
    train, val, test = _temporal_split(table, cfg["split"]["train"], cfg["split"]["validation"], gap_days=1)
    if train.target_rain_event.nunique() < 2:
        raise ValueError("Training interval contains only one rainfall class; widen the historical interval or review the event threshold")
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    table.to_csv("data/processed/openweather_chirps_daily.csv", index=False)

    specs = _model_specs(seed, len(train), train.target_rain_event.to_numpy())
    fitted = {}
    metrics = {}
    thresholds = {}
    validation_scores = {}
    importance_rows = []
    for name, (task, estimator) in specs.items():
        ycol = "target_rain_event" if task == "classification" else "target_rainfall_mm"
        prep = _preprocessor(numeric, categorical)
        model = Pipeline([("preprocess", prep), ("model", estimator)])
        model.fit(train[features], train[ycol])
        if task == "classification":
            p_val = model.predict_proba(val[features])[:, 1]
            threshold_val = select_threshold(val[ycol].to_numpy(), p_val, cfg["models"].get("selection_metric", "f1"))
            yhat_val = p_val >= threshold_val
            score = float(__import__("sklearn.metrics", fromlist=["average_precision_score"]).average_precision_score(val[ycol], p_val))
            p_test = model.predict_proba(test[features])[:, 1]
            yhat_test = p_test >= threshold_val
            result = classification_metrics(test[ycol].to_numpy(), yhat_test.astype(int), p_test)
            result["threshold_from_validation"] = threshold_val
            thresholds[name] = threshold_val
            importance_score = "average_precision"
        else:
            pred_val = np.maximum(0, model.predict(val[features]))
            score = float(np.mean(np.abs(val[ycol].to_numpy() - pred_val)))
            pred_test = np.maximum(0, model.predict(test[features]))
            result = regression_metrics(test[ycol].to_numpy(), pred_test)
            importance_score = "neg_mean_absolute_error"
        metrics[name] = {"task": task, "validation_score": score, "test": result}
        validation_scores[name] = score
        fitted[name] = model
        bundle = {"model": model, "features": features, "numeric_features": numeric,
                  "categorical_features": categorical, "threshold": thresholds.get(name),
                  "target": ycol, "source": "openweathermap-history + CHIRPS-v3-daily-final-sat",
                  "forecast_horizon_days": 1, "synthetic": False}
        joblib.dump(bundle, Path("models") / f"daily_{name}.joblib", compress=3)
        try:
            imp = permutation_importance(model, val[features], val[ycol], scoring=importance_score,
                n_repeats=3, random_state=seed, n_jobs=1, max_samples=min(1.0, 1500 / max(len(val), 1)))
            importance_rows.extend({"model": name, "task": task, "feature": feature,
                "importance_mean": float(mean), "importance_std": float(std)}
                for feature, mean, std in zip(features, imp.importances_mean, imp.importances_std))
        except Exception as exc:
            print(f"Validation permutation importance skipped for {name}: {type(exc).__name__}")
        print(f"{name}: validation={score:.4f}; test metrics calculated")

    classifiers = {k: v for k, v in validation_scores.items() if metrics[k]["task"] == "classification"}
    regressors = {k: v for k, v in validation_scores.items() if metrics[k]["task"] == "regression"}
    best_classifier = max(classifiers, key=classifiers.get) if classifiers else None
    best_regressor = min(regressors, key=regressors.get) if regressors else None
    if importance_rows:
        rank = pd.DataFrame(importance_rows)
        rank.to_csv(output_dir / "validation_feature_importance_by_model.csv", index=False)
        rank["positive_importance"] = rank.importance_mean.clip(lower=0)
        rank["within_model_share"] = rank.groupby("model").positive_importance.transform(
            lambda x: x / x.sum() if x.sum() else x)
        overall = rank.groupby("feature", as_index=False).within_model_share.mean().sort_values(
            "within_model_share", ascending=False)
        overall.to_csv(output_dir / "validation_feature_ranking.csv", index=False)
    summary = {
        "data_source": "OpenWeather historical hourly observations + CHIRPS v3 daily final satellite product",
        "target": "next calendar day's CHIRPS area-mean rainfall (mm)",
        "rain_event_threshold_mm_per_day": threshold,
        "synthetic_data": False,
        "spatial_resolution": "CHIRPS 0.05-degree native cells; bbox area mean, not 1-km truth",
        "rows": {"all": len(table), "train": len(train), "validation": len(val), "test": len(test)},
        "periods": {name: [str(part.date.min().date()), str(part.date.max().date())]
                    for name, part in (("train", train), ("validation", val), ("test", test))},
        "features": features,
        "models": metrics,
        "selected_by_validation_only": {"classification_best_average_precision": best_classifier,
                                        "regression_best_mae": best_regressor},
        "feature_importance_method": "permutation importance on validation only; grouped raw input columns",
    }
    (output_dir / "daily_live_metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=True))
    pd.DataFrame([{"model": name, "task": val["task"], "validation_score": val["validation_score"],
                   **{f"test_{k}": v for k, v in val["test"].items() if not isinstance(v, list)}}
                  for name, val in metrics.items()]).to_csv(output_dir / "daily_live_model_comparison.csv", index=False)
    print(json.dumps(summary["selected_by_validation_only"], indent=2))
    print(f"Saved test results and validation feature ranks under {output_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input-dir", default="data/raw")
    parser.add_argument("--output-dir", default="experiments")
    args = parser.parse_args()
    run(args.config, args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
