
import random

import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

from predictive_maintenance.config import PROCESSED_DATA_DIR, REPORT_DIR

RANDOM_SEED = 42
SPLIT_SEEDS = [42, 123, 2026]
LABELED_ENGINE_COUNTS = [20, 40, 60]
SUBSET_REPEATS = 5

SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]

FEATURE_SETS = {
    "All sensors": SETTING_COLUMNS + SENSOR_COLUMNS,
    "Constant sensors removed": SETTING_COLUMNS + [
        sensor for sensor in SENSOR_COLUMNS
        if sensor not in {
            "sensor_1", "sensor_5", "sensor_10",
            "sensor_16", "sensor_18", "sensor_19",
        }
    ],
    "Constant and sensor_6 removed": SETTING_COLUMNS + [
        sensor for sensor in SENSOR_COLUMNS
        if sensor not in {
            "sensor_1", "sensor_5", "sensor_6",
            "sensor_10", "sensor_16", "sensor_18", "sensor_19",
        }
    ],
}


def make_model():
    return ExtraTreesRegressor(
        n_estimators=100,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=2,
    )


def run_experiment():
    data_path = PROCESSED_DATA_DIR / "train_FD001_with_RUL.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    results = []

    for split_seed in SPLIT_SEEDS:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=split_seed,
        )

        train_idx, val_idx = next(
            splitter.split(
                df[SETTING_COLUMNS + SENSOR_COLUMNS],
                df["RUL"],
                groups=df["engine_id"],
            )
        )

        train_data = df.iloc[train_idx].copy()
        val_data = df.iloc[val_idx].copy()

        train_engines = sorted(train_data["engine_id"].unique())
        validation_engines = set(val_data["engine_id"].unique())
        assert set(train_engines).isdisjoint(validation_engines)

        y_val = val_data["RUL"]

        for labeled_count in LABELED_ENGINE_COUNTS:
            for repeat in range(SUBSET_REPEATS):
                rng = random.Random(
                    RANDOM_SEED
                    + split_seed * 1000
                    + labeled_count * 100
                    + repeat
                )

                selected_engines = rng.sample(
                    train_engines, labeled_count
                )

                subset = train_data[
                    train_data["engine_id"].isin(selected_engines)
                ]

                for feature_set_name, feature_columns in FEATURE_SETS.items():
                    model = make_model()
                    model.fit(
                        subset[feature_columns],
                        subset["RUL"],
                    )

                    predictions = model.predict(
                        val_data[feature_columns]
                    )

                    results.append({
                        "split_seed": split_seed,
                        "repeat": repeat + 1,
                        "labeled_engines": labeled_count,
                        "feature_set": feature_set_name,
                        "feature_count": len(feature_columns),
                        "MAE_cycles": mean_absolute_error(
                            y_val, predictions
                        ),
                        "RMSE_cycles": root_mean_squared_error(
                            y_val, predictions
                        ),
                    })

    results_df = pd.DataFrame(results)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    detailed_path = REPORT_DIR / "feature_selection_runs.csv"
    results_df.to_csv(detailed_path, index=False)

    summary = results_df.groupby(
        ["labeled_engines", "feature_set"]
    ).agg(
        runs=("MAE_cycles", "count"),
        mean_MAE=("MAE_cycles", "mean"),
        std_MAE=("MAE_cycles", "std"),
        mean_RMSE=("RMSE_cycles", "mean"),
    ).reset_index()

    summary_path = REPORT_DIR / "feature_selection_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n--- Feature Selection Results ---")
    print(summary.round(2).to_string(index=False))
    print(f"\nDetailed results saved to: {detailed_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    run_experiment()
