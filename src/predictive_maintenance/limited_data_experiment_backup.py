
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

from predictive_maintenance.config import PROCESSED_DATA_DIR

RANDOM_SEED = 42
SUBSET_REPEATS = 5
ENGINE_COUNTS = [20, 40, 60, 80]


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

    # Keep the same held-out validation engines
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_SEED,
    )
    train_idx, val_idx = next(splitter.split(X, y, groups=groups))

    train_data = df.iloc[train_idx].copy()
    val_data = df.iloc[val_idx].copy()

    training_engines = sorted(train_data["engine_id"].unique())
    X_val = val_data[feature_columns]
    y_val = val_data["RUL"]

    all_results = []

    print(f"Validation engines: {val_data['engine_id'].nunique()}")
    print(f"Random subset repetitions: {SUBSET_REPEATS}")
    print("\n--- Repeated Limited-Data Experiment ---")

    for engine_count in ENGINE_COUNTS:
        for repeat in range(SUBSET_REPEATS):
            # Choose a new subset of training engines
            rng = __import__("random").Random(
                RANDOM_SEED + engine_count * 100 + repeat
            )
            selected_engines = sorted(
                rng.sample(training_engines, engine_count)
            )

            subset = train_data[
                train_data["engine_id"].isin(selected_engines)
            ]

            model = RandomForestRegressor(
                n_estimators=100,
                min_samples_leaf=2,
                random_state=RANDOM_SEED,
                n_jobs=2,
            )

            model.fit(subset[feature_columns], subset["RUL"])
            predictions = model.predict(X_val)

            mae = mean_absolute_error(y_val, predictions)
            rmse = root_mean_squared_error(y_val, predictions)

            all_results.append({
                "training_engines": engine_count,
                "repeat": repeat + 1,
                "training_rows": len(subset),
                "MAE_cycles": mae,
                "RMSE_cycles": rmse,
            })

            print(
                f"Engines: {engine_count:2} | "
                f"Repeat: {repeat + 1} | "
                f"MAE: {mae:.2f} | RMSE: {rmse:.2f}"
            )

    results_df = pd.DataFrame(all_results)

    summary = results_df.groupby("training_engines").agg(
        mean_MAE=("MAE_cycles", "mean"),
        std_MAE=("MAE_cycles", "std"),
        mean_RMSE=("RMSE_cycles", "mean"),
        std_RMSE=("RMSE_cycles", "std"),
    ).reset_index()

    report_dir = PROCESSED_DATA_DIR.parent.parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
        report_dir / "limited_data_repeated_runs.csv", index=False
    )
    summary.to_csv(
        report_dir / "limited_data_summary.csv", index=False
    )

    print("\n--- Summary: Mean and Standard Deviation ---")
    print(summary.round(2).to_string(index=False))
    print("\nSaved detailed runs and summary in the reports folder.")


if __name__ == "__main__":
    run_experiment()
