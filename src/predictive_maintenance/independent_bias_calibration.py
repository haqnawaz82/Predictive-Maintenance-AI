import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from predictive_maintenance.config import PROCESSED_DATA_DIR, REPORT_DIR
FEATURES = ["setting_1", "setting_2", "setting_3"] + [
    f"sensor_{i}" for i in range(1, 22)
]
SEED = 42
def make_cutoff_samples(data, engine_ids, seed):
    rng = np.random.default_rng(seed)
    samples = []
    for engine_id in engine_ids:
        engine = data[data["engine_id"] == engine_id].sort_values("cycle")
        # Choose a cutoff between 50% and 90% of the engine's lifetime.
        max_cycle = int(engine["cycle"].max())
        min_cycle = max(1, int(np.ceil(max_cycle * 0.50)))
        max_cutoff = max(min_cycle, int(np.floor(max_cycle * 0.90)))
        cutoff = int(rng.integers(min_cycle, max_cutoff + 1))
        row = engine.loc[engine["cycle"] == cutoff].iloc[0].copy()
        samples.append(row)
    return pd.DataFrame(samples).reset_index(drop=True)
def main():
    data_path = PROCESSED_DATA_DIR / "train_FD001_with_RUL.csv"
    data = pd.read_csv(data_path)
    engine_ids = np.array(sorted(data["engine_id"].unique()))
    # Separate engines into training, calibration, and evaluation groups.
    first_split = GroupShuffleSplit(
        n_splits=1, test_size=0.40, random_state=SEED
    )
    train_idx, holdout_idx = next(
        first_split.split(engine_ids, groups=engine_ids)
    )
    train_engines = engine_ids[train_idx]
    holdout_engines = engine_ids[holdout_idx]
    second_split = GroupShuffleSplit(
        n_splits=1, test_size=0.50, random_state=SEED + 1
    )
    cal_idx, eval_idx = next(
        second_split.split(holdout_engines, groups=holdout_engines)
    )
    calibration_engines = holdout_engines[cal_idx]
    evaluation_engines = holdout_engines[eval_idx]
    train_rows = data[data["engine_id"].isin(train_engines)]
    calibration = make_cutoff_samples(data, calibration_engines, SEED + 2)
    evaluation = make_cutoff_samples(data, evaluation_engines, SEED + 3)
    model = ExtraTreesRegressor(
        n_estimators=100,
        min_samples_leaf=2,
        random_state=SEED,
        n_jobs=2,
    )
    model.fit(train_rows[FEATURES], train_rows["RUL"])
    cal_predictions = model.predict(calibration[FEATURES]).clip(min=0)
    correction = float(np.mean(cal_predictions - calibration["RUL"].to_numpy()))
    actual = evaluation["RUL"].to_numpy()
    original = model.predict(evaluation[FEATURES]).clip(min=0)
    corrected = np.clip(original - correction, 0, None)
    results = pd.DataFrame([
        {
            "method": "Original Extra Trees",
            "MAE": mean_absolute_error(actual, original),
            "RMSE": root_mean_squared_error(actual, original),
            "mean_error": np.mean(original - actual),
        },
        {
            "method": "Calibration bias-corrected",
            "MAE": mean_absolute_error(actual, corrected),
            "RMSE": root_mean_squared_error(actual, corrected),
            "mean_error": np.mean(corrected - actual),
        },
    ])
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = REPORT_DIR / "independent_bias_calibration_results.csv"
    predictions_path = REPORT_DIR / "independent_bias_calibration_predictions.csv"
    results.to_csv(results_path, index=False)
    prediction_table = evaluation[["engine_id", "cycle", "RUL"]].copy()
    prediction_table["original_prediction"] = original
    prediction_table["corrected_prediction"] = corrected
    prediction_table.to_csv(predictions_path, index=False)
    print("Training engines:", len(train_engines))
    print("Calibration engines:", len(calibration_engines))
    print("Evaluation engines:", len(evaluation_engines))
    print("Estimated calibration bias (cycles):", round(correction, 2))
    print("\n--- Independent Evaluation Results ---")
    print(results.round(2).to_string(index=False))
    print("\nSaved:", results_path)
    print("Saved:", predictions_path)
if __name__ == "__main__":
    main()
