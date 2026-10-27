
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

from predictive_maintenance.config import PROCESSED_DATA_DIR, REPORT_DIR

RANDOM_SEED = 42


def train_baseline():
    # 1. Load the prepared dataset
    data_path = PROCESSED_DATA_DIR / "train_FD001_with_RUL.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {data_path}"
        )

    df = pd.read_csv(data_path)

    # 2. Select sensor readings and operating settings
    feature_columns = [
        "setting_1",
        "setting_2",
        "setting_3",
    ] + [f"sensor_{i}" for i in range(1, 22)]

    X = df[feature_columns]
    y = df["RUL"]
    groups = df["engine_id"]

    # 3. Split by engine to prevent engine-level data leakage
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_SEED,
    )

    train_idx, val_idx = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    train_engines = set(groups.iloc[train_idx])
    val_engines = set(groups.iloc[val_idx])

    assert train_engines.isdisjoint(val_engines)

    # 4. Train the Random Forest model
    model = RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=2,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    # 5. Evaluate Random Forest
    rf_mae = mean_absolute_error(y_val, predictions)
    rf_rmse = root_mean_squared_error(y_val, predictions)

    # 6. Evaluate the median-RUL baseline
    baseline_predictions = [y_train.median()] * len(y_val)

    baseline_mae = mean_absolute_error(
        y_val, baseline_predictions
    )
    baseline_rmse = root_mean_squared_error(
        y_val, baseline_predictions
    )

    # 7. Save results
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame([
        {
            "model": "Random Forest",
            "MAE_cycles": rf_mae,
            "RMSE_cycles": rf_rmse,
        },
        {
            "model": "Median baseline",
            "MAE_cycles": baseline_mae,
            "RMSE_cycles": baseline_rmse,
        },
    ])

    results_path = REPORT_DIR / "baseline_results.csv"
    results.to_csv(results_path, index=False)

    # 8. Display the results
    print("\n--- Dataset Split ---")
    print(f"Training rows: {len(train_idx):,}")
    print(f"Validation rows: {len(val_idx):,}")
    print(f"Training engines: {len(train_engines)}")
    print(f"Validation engines: {len(val_engines)}")

    print("\n--- Model Comparison ---")
    print(results.round(2).to_string(index=False))
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    train_baseline()
