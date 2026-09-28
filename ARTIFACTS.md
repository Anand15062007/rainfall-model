# Generated artifacts

The checked-in data, database, model bundles, plots, and experiment reports are outputs from the bundled synthetic-data run. They are useful for inspecting the application and reproducing its software workflow; they are not measured real-world rainfall skill or ERA5 observations.

`data/rainfall.sqlite` contains the local run database. `data/synthetic/weather.csv` is the input generated for the demo. `data/processed/` and `experiments/` contain the run's predictions, comparison metrics, feature importance tables, and plots.

Model binaries are stored with Git LFS. The large random-forest regression bundle is gzip-compressed as `models/random_forest_regressor.joblib.gz` to fit GitHub's per-file LFS limit. `joblib.load("models/random_forest_regressor.joblib.gz")` can read the compressed file; to restore the original uncompressed artifact, run `gzip -dk models/random_forest_regressor.joblib.gz`.
