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

## Coordinate rainfall forecast website

Run the website after installing the project dependencies and adding `OPENWEATHER_API_KEY` to `.env`:

```bash
streamlit run dashboard/app.py
```

Open the local address printed by Streamlit (normally `http://localhost:8501`). Enter latitude, longitude, and a UTC forecast date. The site uses OpenWeather's forecast endpoint and displays the highest 3-hour precipitation probability returned for that date plus the sum of forecast 3-hour rainfall amounts. The API provides up to five days of 3-hour forecast intervals. This page reports the API forecast directly; it does not use the synthetic-trained model bundles as real-world predictions.

To run the trained daily model after real historical training, pass hourly OpenWeather observations to the saved daily bundle. It uses the latest complete day with at least 18 hourly records to predict the next day:

```bash
python -m app.prediction.predict models/daily_random_forest_classifier.joblib --input data/raw/openweather_hourly.csv
```

## Dataset schema

Required provider fields are timestamp, grid ID, latitude, longitude. Standard weather columns include temperature, relative humidity, pressure, wind speed/direction, cloud cover, dew point, rainfall accumulation windows, and optional environmental measurements. Missing optional values remain missing until train-fitted imputation. The synthetic generator adds rainfall event and amount labels for convenience.

## Evaluation and leakage controls

The canonical split is chronological by unique timestamp. Validation and test observations are held out from preprocessing fitting, with a horizon-sized purge at each split to prevent boundary labels from reaching into the next window. Rolling-origin `TimeSeriesSplit` and spatially disjoint splitters are also provided independently. The default run reports the number of valid rolling-origin folds; per-fold model score aggregation is a separate experiment step. Lag and rolling features use past values; future rainfall is isolated in `target_*` columns and prohibited from the feature set. Classification outputs accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, POD, FAR, CSI, ETS. Regression reports MAE, RMSE, R², bias, and correlation. Prefer event recall/F1/PR-AUC/CSI for rare rain and consider all metrics together; classification probability thresholds are tuned on validation with `models.selection_metric`, and the untouched chronological test window is used once for final reporting.

## API keys and integration

`.env.example` lists the CDS API token plus future weather, satellite, radar, and map credentials. Never place secrets in source. `MockWeatherProvider` and `CSVWeatherProvider` work locally. `APIWeatherProvider` is an intentional adapter placeholder: choose a vendor before implementing endpoint/auth/schema mapping. Configure credentials via environment variables, then add timeout/retry/rate-limit behavior and mocked response tests before live requests. No credentials are currently required.

## Limitations and next steps

Synthetic data encodes plausible covariation but is not calibrated to a city or observation product. The orchestrated benchmark runs Logistic Regression, Random Forest, XGBoost and LSTM classification/regression using the same temporal holdout. Spatial/spatiotemporal experiments have split primitives rather than a complete benchmark runner. Grid clipping is bbox intersection, so edge cells may extend past the bbox. For operational forecasting, obtain quality-controlled, time-aligned historical data, establish appropriate validation protocols, and calibrate the full workflow.

## OpenWeather + CHIRPS daily experiment

The live-data experiment uses OpenWeather's **historical hourly observations** as predictors and the CHIRPS v3 **daily final satellite** product as rainfall target. The model predicts next-day area-mean rainfall for the configured bbox; it does not claim 1-km truth. CHIRPS cells are approximately 0.05 degrees (about 5 km), and the area mean stays at native-cell scale. CHIRPS daily satellite data starts in 1998. Its daily values distribute pentad totals using IMERG daily rainfall. OpenWeather's history endpoint is subscription-gated; a key that works for current conditions may not have history access. [CHIRPS v3 documentation](https://chc.ucsb.edu/data/chirps3) and [OpenWeather History API](https://openweathermap.org/api/history?collection=historical).

The sample bbox in `configs/default.yaml` is only a placeholder. Replace the city name and bbox with the intended location before downloading. The code reads `OPENWEATHER_API_KEY` from the ignored local `.env`; never commit that file.

Windows Command Prompt setup (replace the sample bbox in `configs\default.yaml` first):

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -e .[all]
copy .env.example .env
notepad .env
```

Set `OPENWEATHER_API_KEY` in `.env` and save it. Then download and train:

```bat
python -m app.data.chirps_download --start-year 2015 --end-year 2025
python -m app.data.openweather_history --start 2015-01-01 --end 2025-12-31
python -m app.data.daily_pipeline
```

On macOS/Linux, activate with `source .venv/bin/activate`, copy the env template with `cp .env.example .env`, and use the same `python -m ...` commands. `pip install -e '.[all]'` installs the optional model, dashboard, ERA5, CHIRPS, and test dependencies too. Full historical OpenWeather downloads can make hundreds of requests and require an eligible subscription; the downloader reports a clear access error if history is not enabled.

CHIRPS is read from remote NetCDF files using HTTP byte ranges, extracting only chunks that intersect the bbox. OpenWeather requests are issued in seven-day UTC windows and saved without credentials. The daily target uses a configurable 1.0 mm event threshold. The model table uses weather observed through day *t* to predict CHIRPS rainfall on day *t+1*. It compares Logistic Regression, Random Forest classification/regression, and XGBoost when installed. Preprocessing is fitted on the training interval; the classifier cutoff and model selection are chosen on validation data, then each model is scored on the later test interval. Validation-only permutation rankings are written to `experiments/validation_feature_ranking.csv` and `experiments/validation_feature_importance_by_model.csv`. Test metrics are written to `experiments/daily_live_metrics.json` and `experiments/daily_live_model_comparison.csv`.

If OpenWeather returns HTTP 401/403, its historical API is not enabled for the account or key. Current-weather requests are not a substitute for historical training data. Do not train a historical model by joining today's live weather to old CHIRPS dates.
