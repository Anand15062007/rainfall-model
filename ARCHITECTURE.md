# Architecture

Data flows from `WeatherDataProvider` to standardized observation frames, projected grid generation, causal feature construction, chronological split, train-fitted preprocessing, task-specific `BaseModel` implementations, metric calculation and experiment tracking. Persistence is independent of models: SQLite stores local relational records while joblib model bundles include estimator metadata, fitted preprocessor, feature ordering and config.

The provider boundary prevents vendor dependencies from reaching model code. Spatial grids use a local azimuthal equidistant projection centered on the city bbox; cells are square in projected meters. Mapping is isolated in `app.visualization`.

`BaseModel` is intended for future CNN and spatiotemporal implementations. The LSTM is a separate sequence estimator using configurable history length. The pipeline constructs histories independently by grid and uses identical temporal holdout periods.
