
import random

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

from predictive_maintenance.config import PROCESSED_DATA_DIR, REPORT_DIR

RANDOM_SEED = 42
SPLIT_SEEDS = [42, 123, 2026]
LABELED_ENGINE_COUNTS = [20, 40, 60]
SUBSET_REPEATS = 5


def make_model():
    return RandomForestRegressor(
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

    feature_columns = [
        "setting_1",
        "setting_2",
        "setting_3",
    ] + [f"sensor_{i}" for i in range(1, 22)]

    all_results = []

    for split_seed in SPLIT_SEEDS:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=split_seed,
        )

        train_idx, val_idx = next(
            splitter.split(
                df[feature_columns],
                df["RUL"],
                groups=df["engine_id"],
            )
        )

        train_data = df.iloc[train_idx].copy()
        val_data = df.iloc[val_idx].copy()

        train_engines = sorted(train_data["engine_id"].unique())
        validation_engines = set(val_data["engine_id"].unique())

        assert set(train_engines).isdisjoint(validation_engines)

        X_val = val_data[feature_columns]
        y_val = val_data["RUL"]

        for labeled_count in LABELED_ENGINE_COUNTS:
            for repeat in range(SUBSET_REPEATS):
                rng = random.Random(
                    RANDOM_SEED
                    + split_seed * 1000
                    + labeled_count * 100
                    + repeat
                )

                labeled_engines = sorted(
                    rng.sample(train_engines, labeled_count)
                )
                labeled_engine_set = set(labeled_engines)

                labeled_data = train_data[
                    train_data["engine_id"].isin(labeled_engine_set)
                ]

                unlabeled_data = train_data[
                    ~train_data["engine_id"].isin(labeled_engine_set)
                ]

                # Supervised baseline: real RUL labels only.
                baseline_model = make_model()
                baseline_model.fit(
                    labeled_data[feature_columns],
                    labeled_data["RUL"],
                )
                baseline_predictions = baseline_model.predict(X_val)

                # Semi-supervised prototype:
                # predict pseudo-labels for the remaining training engines.
                semi_model = make_model()
                semi_model.fit(
                    labeled_data[feature_columns],
                    labeled_data["RUL"],
                )

                pseudo_labels = semi_model.predict(
                    unlabeled_data[feature_columns]
                )

                augmented_X = pd.concat(
                    [
                        labeled_data[feature_columns],
                        unlabeled_data[feature_columns],
                    ],
                    ignore_index=True,
                )

                augmented_y = pd.concat(
                    [
                        labeled_data["RUL"].reset_index(drop=True),
                        pd.Series(pseudo_labels),
                    ],
                    ignore_index=True,
                )

                semi_model.fit(augmented_X, augmented_y)
                semi_predictions = semi_model.predict(X_val)

                all_results.append({
                    "split_seed": split_seed,
                    "repeat": repeat + 1,
                    "labeled_engines": labeled_count,
                    "unlabeled_engines": len(
                        unlabeled_data["engine_id"].unique()
                    ),
                    "baseline_MAE": mean_absolute_error(
                        y_val, baseline_predictions
                    ),
                    "semi_supervised_MAE": mean_absolute_error(
                        y_val, semi_predictions
                    ),
                    "baseline_RMSE": root_mean_squared_error(
                        y_val, baseline_predictions
                    ),
                    "semi_supervised_RMSE": root_mean_squared_error(
                        y_val, semi_predictions
                    ),
                })

    results_df = pd.DataFrame(all_results)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    detailed_path = REPORT_DIR / "semi_supervised_randomized_runs.csv"
    results_df.to_csv(detailed_path, index=False)

    summary = results_df.groupby("labeled_engines").agg(
        runs=("baseline_MAE", "count"),
        baseline_MAE=("baseline_MAE", "mean"),
        baseline_MAE_std=("baseline_MAE", "std"),
        semi_supervised_MAE=("semi_supervised_MAE", "mean"),
        semi_supervised_MAE_std=("semi_supervised_MAE", "std"),
        baseline_RMSE=("baseline_RMSE", "mean"),
        semi_supervised_RMSE=("semi_supervised_RMSE", "mean"),
    ).reset_index()

    summary_path = REPORT_DIR / "semi_supervised_randomized_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n--- Randomized Experiment Summary ---")
    print(summary.round(2).to_string(index=False))
    print(f"\nDetailed results saved to: {detailed_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    run_experiment()
