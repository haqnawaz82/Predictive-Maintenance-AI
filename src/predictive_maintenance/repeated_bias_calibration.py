import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from predictive_maintenance.config import PROCESSED_DATA_DIR, REPORT_DIR
FEATURES = ["setting_1", "setting_2", "setting_3"] + [
    f"sensor_{i}" for i in range(1, 22)
]
def make_cutoff_samples(data, engine_ids, seed):
    rng = np.random.default_rng(seed)
    samples = []
    for engine_id in engine_ids:
        engine = data[data["engine_id"] == engine_id].sort_values("cycle")
        max_cycle = int(engine["cycle"].max())
        low = max(1, int(np.ceil(max_cycle * 0.50)))
        high = max(low, int(np.floor(max_cycle * 0.90)))
        cutoff = int(rng.integers(low, high + 1))
        samples.append(engine.loc[engine["cycle"] == cutoff].iloc[0].copy())
    return pd.DataFrame(samples).reset_index(drop=True)
def main():
    data = pd.read_csv(
        PROCESSED_DATA_DIR / "train_FD001_with_RUL.csv"
    )
    engine_ids = np.array(sorted(data["engine_id"].unique()))
    trial_results = []
    for seed in range(100, 110):
        first = GroupShuffleSplit(
            n_splits=1, test_size=0.40, random_state=seed
        )
        train_idx, holdout_idx = next(
            first.split(engine_ids, groups=engine_ids)
        )
        train_engines = engine_ids[train_idx]
        holdout_engines = engine_ids[holdout_idx]
        second = GroupShuffleSplit(
            n_splits=1, test_size=0.50, random_state=seed + 1000
        )
        cal_idx, eval_idx = next(
            second.split(holdout_engines, groups=holdout_engines)
        )
        cal_engines = holdout_engines[cal_idx]
        eval_engines = holdout_engines[eval_idx]
        train = data[data["engine_id"].isin(train_engines)]
        calibration = make_cutoff_samples(data, cal_engines, seed + 2000)
        evaluation = make_cutoff_samples(data, eval_engines, seed + 3000)
        model = ExtraTreesRegressor(
            n_estimators=100,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=2,
        )
        model.fit(train[FEATURES], train["RUL"])
        cal_pred = model.predict(calibration[FEATURES]).clip(min=0)
        correction = float(
            np.mean(cal_pred - calibration["RUL"].to_numpy())
        )
        actual = evaluation["RUL"].to_numpy()
        original = model.predict(evaluation[FEATURES]).clip(min=0)
        corrected = np.clip(original - correction, 0, None)
        trial_results.append({
            "seed": seed,
            "training_engines": len(train_engines),
            "calibration_engines": len(cal_engines),
            "evaluation_engines": len(eval_engines),
            "correction_cycles": correction,
            "original_MAE": mean_absolute_error(actual, original),
            "corrected_MAE": mean_absolute_error(actual, corrected),
            "original_RMSE": root_mean_squared_error(actual, original),
            "corrected_RMSE": root_mean_squared_error(actual, corrected),
            "original_mean_error": np.mean(original - actual),
            "corrected_mean_error": np.mean(corrected - actual),
        })
        print(
            f"Trial {seed}: "
            f"MAE {mean_absolute_error(actual, original):.2f} -> "
            f"{mean_absolute_error(actual, corrected):.2f}; "
            f"RMSE {root_mean_squared_error(actual, original):.2f} -> "
            f"{root_mean_squared_error(actual, corrected):.2f}"
        )
    results = pd.DataFrame(trial_results)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    runs_path = REPORT_DIR / "repeated_bias_calibration_runs.csv"
    summary_path = REPORT_DIR / "repeated_bias_calibration_summary.csv"
    results.to_csv(runs_path, index=False)
    summary = pd.DataFrame({
        "metric": [
            "MAE",
            "RMSE",
            "Mean signed error",
            "Estimated correction (cycles)",
        ],
        "original_mean": [
            results["original_MAE"].mean(),
            results["original_RMSE"].mean(),
            results["original_mean_error"].mean(),
            np.nan,
        ],
        "corrected_mean": [
            results["corrected_MAE"].mean(),
            results["corrected_RMSE"].mean(),
            results["corrected_mean_error"].mean(),
            results["correction_cycles"].mean(),
        ],
        "original_std": [
            results["original_MAE"].std(),
            results["original_RMSE"].std(),
            results["original_mean_error"].std(),
            np.nan,
        ],
        "corrected_std": [
            results["corrected_MAE"].std(),
            results["corrected_RMSE"].std(),
            results["corrected_mean_error"].std(),
            results["correction_cycles"].std(),
        ],
    })
    summary.to_csv(summary_path, index=False)
    print("\n--- Ten-Trial Summary: Mean and Standard Deviation ---")
    print(summary.round(2).to_string(index=False))
    print("\nSaved:", runs_path)
    print("Saved:", summary_path)
if __name__ == "__main__":
    main()
