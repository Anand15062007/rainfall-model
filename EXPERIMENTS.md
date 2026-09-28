# Experiment tracking

Each orchestrated run appends JSON records to `experiments/runs.jsonl` and refreshes `runs.csv`, recording ID, timestamp, synthetic dataset version, feature ordering, grid size, horizon, estimator parameters, train/validation/test periods, scores, row counts and random seed. `latest_metrics.json` and `model_comparison.csv` summarize the latest run. Compare models under identical splits and report task/threshold/horizon with all relevant scores.
