
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
RUL_CAP = 125


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

    feature_columns = [
        "setting_1",
        "setting_2",
        "setting_3",
    ] + [f"sensor_{i}" for i in range(1, 22)]

    results = []

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

        original_y_val = val_data["RUL"]
        capped_y_val = original_y_val.clip(upper=RUL_CAP)

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

                for target_name in ["uncapped", "capped"]:
                    model = make_model()

                    if target_name == "uncapped":
                        y_train = subset["RUL"]
                    else:
                        y_train = subset["RUL"].clip(upper=RUL_CAP)

                    model.fit(subset[feature_columns], y_train)
                    predictions = model.predict(val_data[feature_columns])

                    if target_name == "capped":
                        capped_predictions = predictions.clip(
                            min=0, max=RUL_CAP
                        )
                        original_scale_predictions = predictions.clip(min=0)
                    else:
                        capped_predictions = predictions.clip(
                            min=0, max=RUL_CAP
                        )
                        original_scale_predictions = predictions.clip(min=0)

                    results.append({
                        "split_seed": split_seed,
                        "repeat": repeat + 1,
                        "labeled_engines": labeled_count,
                        "target": target_name,
                        "MAE_capped_target": mean_absolute_error(
                            capped_y_val, capped_predictions
                        ),
                        "RMSE_capped_target": root_mean_squared_error(
                            capped_y_val, capped_predictions
                        ),
                        "MAE_original_target": mean_absolute_error(
                            original_y_val, original_scale_predictions
                        ),
                        "RMSE_original_target": root_mean_squared_error(
                            original_y_val, original_scale_predictions
                        ),
                    })

    results_df = pd.DataFrame(results)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    detailed_path = REPORT_DIR / "rul_capping_runs.csv"
    results_df.to_csv(detailed_path, index=False)

    summary = results_df.groupby(
        ["labeled_engines", "target"]
    ).agg(
        runs=("MAE_capped_target", "count"),
        MAE_capped=("MAE_capped_target", "mean"),
        RMSE_capped=("RMSE_capped_target", "mean"),
        MAE_original=("MAE_original_target", "mean"),
        RMSE_original=("RMSE_original_target", "mean"),
    ).reset_index()

    summary_path = REPORT_DIR / "rul_capping_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n--- RUL Capping Results ---")
    print(summary.round(2).to_string(index=False))
    print(f"\nDetailed results saved to: {detailed_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    run_experiment()
