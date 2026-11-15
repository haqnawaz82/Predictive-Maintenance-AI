
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

from predictive_maintenance.config import PROCESSED_DATA_DIR, REPORT_DIR


RANDOM_SEED = 42

FEATURES = [
    "setting_1",
    "setting_2",
    "setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]


def main():
    path = PROCESSED_DATA_DIR / "train_FD001_with_RUL.csv"

    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")

    data = pd.read_csv(path)

    # First split: 80% development engines, 20% final evaluation.
    outer_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_SEED,
    )

    development_idx, evaluation_idx = next(
        outer_splitter.split(
            data[FEATURES],
            data["RUL"],
            groups=data["engine_id"],
        )
    )

    development = data.iloc[development_idx].copy()
    evaluation = data.iloc[evaluation_idx].copy()

    # Second split: 75% model training, 25% calibration.
    # This gives approximately 60/20/20 of all engines.
    inner_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=RANDOM_SEED + 1,
    )

    training_idx, calibration_idx = next(
        inner_splitter.split(
            development[FEATURES],
            development["RUL"],
            groups=development["engine_id"],
        )
    )

    training = development.iloc[training_idx].copy()
    calibration = development.iloc[calibration_idx].copy()

    # Verify that no engine is shared between groups.
    training_engines = set(training["engine_id"].unique())
    calibration_engines = set(calibration["engine_id"].unique())
    evaluation_engines = set(evaluation["engine_id"].unique())

    assert training_engines.isdisjoint(calibration_engines)
    assert training_engines.isdisjoint(evaluation_engines)
    assert calibration_engines.isdisjoint(evaluation_engines)

    # Train the model using training engines only.
    model = ExtraTreesRegressor(
        n_estimators=100,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=2,
    )

    model.fit(training[FEATURES], training["RUL"])

    # Estimate bias using separate calibration engines.
    calibration_predictions = model.predict(calibration[FEATURES])
    calibration_predictions = calibration_predictions.clip(min=0)

    calibration_actual = calibration["RUL"].to_numpy()

    correction = (
        calibration_predictions - calibration_actual
    ).mean()

    # Evaluate on untouched evaluation engines.
    actual = evaluation["RUL"].to_numpy()

    original_predictions = model.predict(evaluation[FEATURES])
    original_predictions = original_predictions.clip(min=0)

    corrected_predictions = (
        original_predictions - correction
    ).clip(min=0)

    results = pd.DataFrame([
        {
            "method": "Original Extra Trees",
            "MAE": mean_absolute_error(
                actual, original_predictions
            ),
            "RMSE": root_mean_squared_error(
                actual, original_predictions
            ),
            "mean_error": (
                original_predictions - actual
            ).mean(),
        },
        {
            "method": "Calibration bias-corrected",
            "MAE": mean_absolute_error(
                actual, corrected_predictions
            ),
            "RMSE": root_mean_squared_error(
                actual, corrected_predictions
            ),
            "mean_error": (
                corrected_predictions - actual
            ).mean(),
        },
    ])

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output = REPORT_DIR / "validation_bias_correction.csv"
    results.to_csv(output, index=False)

    print("\n--- Engine Group Counts ---")
    print("Model training engines:", len(training_engines))
    print("Calibration engines:", len(calibration_engines))
    print("Evaluation engines:", len(evaluation_engines))

    print("\nEstimated bias correction:", round(correction, 2))
    print("\n--- Independent Evaluation Results ---")
    print(results.round(2).to_string(index=False))

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
