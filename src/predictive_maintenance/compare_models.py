
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from predictive_maintenance.config import PROCESSED_DATA_DIR, REPORT_DIR

RANDOM_SEED = 42


def compare_models():
    data_path = PROCESSED_DATA_DIR / "train_FD001_with_RUL.csv"
    df = pd.read_csv(data_path)

    feature_columns = [
        "setting_1",
        "setting_2",
        "setting_3",
    ] + [f"sensor_{i}" for i in range(1, 22)]

    X = df[feature_columns]
    y = df["RUL"]
    groups = df["engine_id"]

    # Keep engines separate between training and validation.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_SEED,
    )

    train_idx, val_idx = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
            n_jobs=2,
        ),
        "Linear Regression": make_pipeline(
            StandardScaler(),
            LinearRegression(),
        ),
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)

        results.append({
            "model": name,
            "MAE_cycles": mean_absolute_error(
                y_val, predictions
            ),
            "RMSE_cycles": root_mean_squared_error(
                y_val, predictions
            ),
        })

    # Add the simple median baseline.
    baseline_predictions = [y_train.median()] * len(y_val)

    results.append({
        "model": "Median baseline",
        "MAE_cycles": mean_absolute_error(
            y_val, baseline_predictions
        ),
        "RMSE_cycles": root_mean_squared_error(
            y_val, baseline_predictions
        ),
    })

    results_df = pd.DataFrame(results).sort_values("MAE_cycles")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / "model_comparison.csv"
    results_df.to_csv(output_path, index=False)

    print("\n--- Model Comparison ---")
    print(f"Training engines: {groups.iloc[train_idx].nunique()}")
    print(f"Validation engines: {groups.iloc[val_idx].nunique()}")
    print(results_df.round(2).to_string(index=False))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    compare_models()
