# Models

Logistic Regression is the binary probability baseline. Random Forest classifier and regressor cover rain occurrence and amount, expose feature importance, accept estimator hyperparameters, and persist through joblib. XGBoost classifier/regressor are optional and fail with a clear install instruction when absent. The PyTorch LSTM supports classification/regression heads, configurable sequence length, validation-loss early stopping, persistence, and per-grid sequence handling in the benchmark. Tree models do not need scaling; the shared example pipeline scales all models for a simple consistent interface, with scaler fitted on training data only.


Classification probability thresholds are selected with validation probabilities using `models.selection_metric` (`f1` by default, `accuracy` is available). XGBoost permutation importance is calculated from validation data and saved separately from train-fit tree impurity importance. Neither process reads the final test period.
