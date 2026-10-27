
import random

import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

from predictive_maintenance.config import PROCESSED_DATA_DIR, REPORT_DIR

RANDOM_SEED = 42
SPLIT_SEEDS = [42, 123, 2026]
LABELED_ENGINE_COUNTS = [20, 40, 60]
SUBSET_REPEATS = 5


def make_random_forest():
    return RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=2,
    )


def make_extra_trees():
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

                selected_engines = rng.sample(
                    train_engines, labeled_count
                )

                subset = train_data[
                    train_data["engine_id"].isin(selected_engines)
                ]

                models = {
                    "Random Forest": make_random_forest(),
                    "Extra Trees": make_extra_trees(),
                }

                row = {
                    "split_seed": split_seed,
                    "repeat": repeat + 1,
                    "labeled_engines": labeled_count,
                    "training_rows": len(subset),
                }

                for model_name, model in models.items():
                    model.fit(
                        subset[feature_columns],
                        subset["RUL"],
                    )

                    predictions = model.predict(X_val)

                    row[f"{model_name}_MAE"] = mean_absolute_error(
                        y_val, predictions
                    )
                    row[f"{model_name}_RMSE"] = (
                        root_mean_squared_error(y_val, predictions)
                    )

                all_results.append(row)

    results_df = pd.DataFrame(all_results)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    detailed_path = REPORT_DIR / "tree_model_comparison_runs.csv"
    results_df.to_csv(detailed_path, index=False)

    summary = results_df.groupby("labeled_engines").agg(
        runs=("Random Forest_MAE", "count"),
        RF_MAE=("Random Forest_MAE", "mean"),
        RF_MAE_std=("Random Forest_MAE", "std"),
        ExtraTrees_MAE=("Extra Trees_MAE", "mean"),
        ExtraTrees_MAE_std=("Extra Trees_MAE", "std"),
        RF_RMSE=("Random Forest_RMSE", "mean"),
        ExtraTrees_RMSE=("Extra Trees_RMSE", "mean"),
    ).reset_index()

    summary_path = REPORT_DIR / "tree_model_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n--- Random Forest vs Extra Trees ---")
    print(summary.round(2).to_string(index=False))
    print(f"\nDetailed results saved to: {detailed_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    run_experiment()
