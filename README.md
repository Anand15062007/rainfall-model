# City-level rainfall prediction and mapping

An offline-first research system for city-scale gridded rainfall experiments. It creates metric-sized geographic cells, generates clearly labeled correlated synthetic hourly observations, builds causal features and future rainfall targets, performs time-based holdouts, trains classical baselines, records experiments, and generates a local grid map. Synthetic results are software demonstrations, not evidence of real-world forecast skill.

## Architecture

`app/grid` handles projected grids; `app/data` handles synthetic and CSV observations; `app/preprocessing` fits train-only transformations; `app/features` creates causal time, lag, rolling, and future-target columns; `app/training` contains temporal and spatial splitters; `app/models` contains estimator adapters; `app/evaluation` scores and tracks runs; `app/api` defines data-provider interfaces; `app/database` stores local records in SQLite; `app/visualization` creates plots/maps; `dashboard/app.py` presents artifacts.

## Install

Use Python 3.12 for the full PyTorch/XGBoost/ERA5 stack. On Windows Command Prompt, these commands install and prepare the project:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
copy .env.example .env
notepad .env
notepad configs\default.yaml
```

Add your CDS token to `.env` when ready. In `configs/default.yaml`, set the actual city bbox as `[west_lon, south_lat, east_lon, north_lat]`. Then:

```bat
python -m app.data.era5_download --start-year 2015 --end-year 2025
python -m app.data.era5_convert --input-dir data/raw/era5 --output data/processed/era5_observations.csv
python -m app.pipeline --data data/processed/era5_observations.csv
streamlit run dashboard/app.py
```

XGBoost and PyTorch are optional individually: install `.[xgboost]` or `.[torch]`. Streamlit and interactive map extras are included in `.[all]`. No external service or API key is needed for local use.

## Configure and run

Edit `configs/default.yaml` for city bbox (`min_lon,min_lat,max_lon,max_lat`), grid size, forecast horizon, rainfall event threshold, feature windows, split fractions, and model options. Generate grid and synthetic data and run the local experiment:

```bash
python -m app.grid.generate
python -m app.data.synthetic
python -m app.pipeline --config configs/default.yaml
```

The pipeline trains all enabled models (Logistic Regression, Random Forest classifier/regressor, XGBoost classifier/regressor, and LSTM classification/regression). It writes grid/weather CSVs, model bundles, experiment JSONL/CSV, feature-importance CSVs, comparison metrics and plots, an interactive map, and a SQLite database. Model hyperparameters and LSTM sequence length/early stopping are in `configs/default.yaml`. Run `python -m app.data.ingest [CSV_PATH]` to validate and store CSV observations; `python -m app.data.preprocess [CSV_PATH]` produces engineered data. Saved estimator bundle examples (including LSTM `.pt` bundles):

```bash
python -m app.prediction.predict models/random_forest_classifier.joblib
```

Individual model use is exposed as `LogisticRegressionModel`, `RandomForestModel(task=...)`, and `XGBoostModel(task=...)`. Missing XGBoost produces an explicit installation message. PyTorch LSTM sequence support is available in `app.models.lstm.LSTMModel`; sequence length and early stopping settings are configurable in YAML.

## ERA5 dataset workflow

The ERA5 predictor subset is preselected in `configs/default.yaml` (see [ERA5_FEATURES.md](ERA5_FEATURES.md) for rationale): 2 m temperature/dewpoint (relative humidity derived), 10 m east/north winds, mean sea-level pressure, total cloud cover, total-column water vapour, CAPE, CIN, boundary-layer height, geopotential/elevation, and **lagged/current already-observed precipitation**. These represent moisture supply, instability/lift, cloud, circulation and orographic controls. Validation-only XGBoost permutation-importance rankings are saved under `experiments/validation_permutation_importance_*.csv`; feature ranking never reads the final test period. For imbalanced rain/no-rain data, the default classification cutoff is selected on validation to maximize F1; change `models.selection_metric` to `accuracy` if raw hit-rate is specifically desired. ERA5 total precipitation is an accumulated hourly field in metres; the importer converts it to mm for the internal schema.

ERA5 atmospheric data is hourly at 0.25° (~28 km north-south). The converter preserves the native ERA5 cells and does not copy one coarse pixel into fake 1 km training samples. A 1 km grid can be added later with genuine high-resolution radar/gauge observations or a validated downscaling method; interpolation alone cannot add rainfall detail.

After you have accepted the ERA5 licence on CDS and added `CDS_API_KEY` to `.env`, set the city bounding box in `configs/default.yaml`, then run:

```bash
python -m app.data.era5_download --start-year 2015 --end-year 2025
python -m app.data.era5_convert --input-dir data/raw/era5 --output data/processed/era5_observations.csv
python -m app.pipeline --data data/processed/era5_observations.csv
```

For Windows Command Prompt activate with `.venv\Scripts\activate.bat` first; the same module commands then work. For PowerShell use `.venv\Scripts\Activate.ps1`. Download requests are one month at a time and skip non-empty files, so rerunning resumes a partial year. The downloader requests a compact city area and the selected variables only. Do not include future-time weather or precipitation fields as input features. The forecast target is accumulated rain from t+1 through t+horizon.

## Dashboard and tests

```bash
streamlit run dashboard/app.py
pytest
```

## Dataset schema

Required provider fields are timestamp, grid ID, latitude, longitude. Standard weather columns include temperature, relative humidity, pressure, wind speed/direction, cloud cover, dew point, rainfall accumulation windows, and optional environmental measurements. Missing optional values remain missing until train-fitted imputation. The synthetic generator adds rainfall event and amount labels for convenience.

## Evaluation and leakage controls

The canonical split is chronological by unique timestamp. Validation and test observations are held out from preprocessing fitting, with a horizon-sized purge at each split to prevent boundary labels from reaching into the next window. Rolling-origin `TimeSeriesSplit` and spatially disjoint splitters are also provided independently. The default run reports the number of valid rolling-origin folds; per-fold model score aggregation is a separate experiment step. Lag and rolling features use past values; future rainfall is isolated in `target_*` columns and prohibited from the feature set. Classification outputs accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, POD, FAR, CSI, ETS. Regression reports MAE, RMSE, R², bias, and correlation. Prefer event recall/F1/PR-AUC/CSI for rare rain and consider all metrics together; classification probability thresholds are tuned on validation with `models.selection_metric`, and the untouched chronological test window is used once for final reporting.

## API keys and integration

`.env.example` lists the CDS API token plus future weather, satellite, radar, and map credentials. Never place secrets in source. `MockWeatherProvider` and `CSVWeatherProvider` work locally. `APIWeatherProvider` is an intentional adapter placeholder: choose a vendor before implementing endpoint/auth/schema mapping. Configure credentials via environment variables, then add timeout/retry/rate-limit behavior and mocked response tests before live requests. No credentials are currently required.

## Limitations and next steps

Synthetic data encodes plausible covariation but is not calibrated to a city or observation product. The orchestrated benchmark runs Logistic Regression, Random Forest, XGBoost and LSTM classification/regression using the same temporal holdout. Spatial/spatiotemporal experiments have split primitives rather than a complete benchmark runner. Grid clipping is bbox intersection, so edge cells may extend past the bbox. For operational forecasting, obtain quality-controlled, time-aligned historical data, establish appropriate validation protocols, and calibrate the full workflow.
