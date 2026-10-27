
import random

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

from predictive_maintenance.config import PROCESSED_DATA_DIR, REPORT_DIR

RANDOM_SEED = 42
SPLIT_SEEDS = [42, 123, 2026]
SUBSET_REPEATS = 5
ENGINE_COUNTS = [20, 40, 60, 80]


def run_experiment():
    data_path = PROCESSED_DATA_DIR / "train_FD001_with_RUL.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    feature_columns = [
        "setting_1",
        "setting_2",
        "setting_3",
    ] + [f"sensor_{i}" for i in range(1, 22)]

    X = df[feature_columns]
    y = df["RUL"]
    groups = df["engine_id"]

    all_results = []

    for split_seed in SPLIT_SEEDS:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=split_seed,
        )

        train_idx, val_idx = next(
            splitter.split(X, y, groups=groups)
        )

        train_data = df.iloc[train_idx].copy()
        val_data = df.iloc[val_idx].copy()

        training_engines = sorted(
            train_data["engine_id"].unique().tolist()
        )
        validation_engines = set(
            val_data["engine_id"].unique()
        )

        assert set(training_engines).isdisjoint(validation_engines)

        X_val = val_data[feature_columns]
        y_val = val_data["RUL"]

        print(
            f"\nSplit seed: {split_seed} | "
            f"Training engines: {len(training_engines)} | "
            f"Validation engines: {len(validation_engines)}"
        )

        for engine_count in ENGINE_COUNTS:
            for repeat in range(SUBSET_REPEATS):
                rng = random.Random(
                    RANDOM_SEED
                    + split_seed * 1000
                    + engine_count * 100
                    + repeat
                )

                selected_engines = rng.sample(
                    training_engines, engine_count
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

                model.fit(
                    subset[feature_columns],
                    subset["RUL"],
                )

                predictions = model.predict(X_val)

                mae = mean_absolute_error(y_val, predictions)
                rmse = root_mean_squared_error(y_val, predictions)

                all_results.append({
                    "split_seed": split_seed,
                    "training_engines": engine_count,
                    "repeat": repeat + 1,
                    "training_rows": len(subset),
                    "validation_engines": len(validation_engines),
                    "MAE_cycles": mae,
                    "RMSE_cycles": rmse,
                })

                print(
                    f"Engines: {engine_count:2} | "
                    f"Repeat: {repeat + 1} | "
                    f"MAE: {mae:.2f} | RMSE: {rmse:.2f}"
                )

    results_df = pd.DataFrame(all_results)

    # Summarize across all split and subset runs.
    summary = results_df.groupby("training_engines").agg(
        runs=("MAE_cycles", "count"),
        mean_MAE=("MAE_cycles", "mean"),
        std_MAE=("MAE_cycles", "std"),
        mean_RMSE=("RMSE_cycles", "mean"),
        std_RMSE=("RMSE_cycles", "std"),
    ).reset_index()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
        REPORT_DIR / "limited_data_multisplit_runs.csv",
        index=False,
    )
    summary.to_csv(
        REPORT_DIR / "limited_data_multisplit_summary.csv",
        index=False,
    )

    print("\n--- Summary Across Splits and Subsets ---")
    print(summary.round(2).to_string(index=False))
    print("\nNew results saved in the reports folder.")


if __name__ == "__main__":
    run_experiment()
