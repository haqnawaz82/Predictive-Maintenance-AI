
import random

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

from predictive_maintenance.config import PROCESSED_DATA_DIR, REPORT_DIR

RANDOM_SEED = 42
SPLIT_SEEDS = [42, 123, 2026]
ENGINE_COUNTS = [20, 40, 60, 80]
SUBSET_REPEATS = 5


def run_evaluation():
    data_path = PROCESSED_DATA_DIR / "train_FD001_with_RUL.csv"
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

        print(f"\nSplit seed: {split_seed}")

        for engine_count in ENGINE_COUNTS:
            for repeat in range(SUBSET_REPEATS):
                rng = random.Random(
                    RANDOM_SEED
                    + split_seed * 1000
                    + engine_count * 100
                    + repeat
                )

                selected_engines = rng.sample(
                    train_engines, engine_count
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

                predictions = model.predict(val_data[feature_columns])

                evaluated = val_data[["engine_id", "RUL"]].copy()
                evaluated["prediction"] = predictions
                evaluated["absolute_error"] = (
                    evaluated["RUL"] - evaluated["prediction"]
                ).abs()
                evaluated["squared_error"] = (
                    evaluated["RUL"] - evaluated["prediction"]
                ) ** 2

                # Calculate errors separately for each held-out engine.
                per_engine = evaluated.groupby("engine_id").agg(
                    engine_MAE=("absolute_error", "mean"),
                    engine_MSE=("squared_error", "mean"),
                )

                per_engine["engine_RMSE"] = per_engine["engine_MSE"] ** 0.5

                all_results.append({
                    "split_seed": split_seed,
                    "training_engines": engine_count,
                    "repeat": repeat + 1,
                    "row_weighted_MAE": mean_absolute_error(
                        evaluated["RUL"], evaluated["prediction"]
                    ),
                    "row_weighted_RMSE": root_mean_squared_error(
                        evaluated["RUL"], evaluated["prediction"]
                    ),
                    "equal_engine_MAE": per_engine["engine_MAE"].mean(),
                    "equal_engine_RMSE": per_engine["engine_RMSE"].mean(),
                    "validation_engine_count": len(per_engine),
                })

                print(
                    f"Engines: {engine_count:2} | "
                    f"Repeat: {repeat + 1} | "
                    f"Equal-engine MAE: "
                    f"{per_engine['engine_MAE'].mean():.2f}"
                )

    results = pd.DataFrame(all_results)

    summary = results.groupby("training_engines").agg(
        runs=("equal_engine_MAE", "count"),
        mean_row_MAE=("row_weighted_MAE", "mean"),
        mean_equal_engine_MAE=("equal_engine_MAE", "mean"),
        std_equal_engine_MAE=("equal_engine_MAE", "std"),
        mean_row_RMSE=("row_weighted_RMSE", "mean"),
        mean_equal_engine_RMSE=("equal_engine_RMSE", "mean"),
        std_equal_engine_RMSE=("equal_engine_RMSE", "std"),
    ).reset_index()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        REPORT_DIR / "per_engine_evaluation_runs.csv", index=False
    )
    summary.to_csv(
        REPORT_DIR / "per_engine_evaluation_summary.csv", index=False
    )

    print("\n--- Per-Engine Evaluation Summary ---")
    print(summary.round(2).to_string(index=False))
    print("\nSaved results in the reports folder.")


if __name__ == "__main__":
    run_evaluation()
