
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

from predictive_maintenance.config import PROCESSED_DATA_DIR

SPLIT_SEEDS = [42, 123, 2026]


def run_experiment():
    data_path = PROCESSED_DATA_DIR / "train_FD001_with_RUL.csv"
    df = pd.read_csv(data_path)

    feature_columns = [
        "setting_1",
        "setting_2",
        "setting_3",
    ] + [f"sensor_{i}" for i in range(1, 22)]

    X = df[feature_columns]
    y = df["RUL"]
    groups = df["engine_id"]

    results = []

    for seed in SPLIT_SEEDS:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=seed,
        )

        train_idx, val_idx = next(
            splitter.split(X, y, groups=groups)
        )

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        train_engines = set(groups.iloc[train_idx])
        val_engines = set(groups.iloc[val_idx])

        assert train_engines.isdisjoint(val_engines)

        model = RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=2,
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        rf_mae = mean_absolute_error(y_val, predictions)
        rf_rmse = root_mean_squared_error(y_val, predictions)

        median_predictions = [y_train.median()] * len(y_val)
        baseline_mae = mean_absolute_error(y_val, median_predictions)
        baseline_rmse = root_mean_squared_error(
            y_val, median_predictions
        )

        results.append({
            "split_seed": seed,
            "training_engines": len(train_engines),
            "validation_engines": len(val_engines),
            "RF_MAE": rf_mae,
            "RF_RMSE": rf_rmse,
            "median_MAE": baseline_mae,
            "median_RMSE": baseline_rmse,
        })

        print(
            f"Seed {seed} | "
            f"RF MAE: {rf_mae:.2f} | "
            f"Median MAE: {baseline_mae:.2f} | "
            f"RF RMSE: {rf_rmse:.2f} | "
            f"Median RMSE: {baseline_rmse:.2f}"
        )

    results_df = pd.DataFrame(results)

    report_dir = PROCESSED_DATA_DIR.parent.parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    output_path = report_dir / "multi_split_results.csv"
    results_df.to_csv(output_path, index=False)

    print("\n--- Results Across Splits ---")
    print(
        results_df[
            ["RF_MAE", "RF_RMSE", "median_MAE", "median_RMSE"]
        ].agg(["mean", "std"]).round(2).to_string()
    )
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    run_experiment()
