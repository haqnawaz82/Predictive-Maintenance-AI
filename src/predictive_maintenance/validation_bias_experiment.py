import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from predictive_maintenance.config import PROCESSED_DATA_DIR, REPORT_DIR
FEATURES = ["setting_1", "setting_2", "setting_3"] + [
    f"sensor_{i}" for i in range(1, 22)
]
def main():
    path = PROCESSED_DATA_DIR / "train_FD001_with_RUL.csv"
    data = pd.read_csv(path)
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=42
    )
    train_idx, val_idx = next(
        splitter.split(data[FEATURES], data["RUL"], groups=data["engine_id"])
    )
    train = data.iloc[train_idx]
    val = data.iloc[val_idx]
    model = ExtraTreesRegressor(
        n_estimators=100,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=2,
    )
    model.fit(train[FEATURES], train["RUL"])
    actual = val["RUL"].to_numpy()
    predicted = model.predict(val[FEATURES])
    correction = (predicted - actual).mean()
    corrected = (predicted - correction).clip(min=0)
    results = pd.DataFrame([
        {
            "method": "Original Extra Trees",
            "MAE": mean_absolute_error(actual, predicted),
            "RMSE": root_mean_squared_error(actual, predicted),
            "mean_error": (predicted - actual).mean(),
        },
        {
            "method": "Validation bias-corrected",
            "MAE": mean_absolute_error(actual, corrected),
            "RMSE": root_mean_squared_error(actual, corrected),
            "mean_error": (corrected - actual).mean(),
        },
    ])
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "validation_bias_correction.csv"
    results.to_csv(output, index=False)
    print("Validation engines:", val["engine_id"].nunique())
    print("Estimated correction (cycles):", round(correction, 2))
    print(results.round(2).to_string(index=False))
    print("Saved:", output)
if __name__ == "__main__":
    main()
