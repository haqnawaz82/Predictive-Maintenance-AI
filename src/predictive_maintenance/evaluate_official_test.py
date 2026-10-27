
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from predictive_maintenance.config import PROCESSED_DATA_DIR, REPORT_DIR

FEATURE_COLUMNS = [
    "setting_1",
    "setting_2",
    "setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]


def evaluate_official_test():
    train_path = PROCESSED_DATA_DIR / "train_FD001_with_RUL.csv"
    test_path = PROCESSED_DATA_DIR / "test_FD001_official.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Official test data not found: {test_path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X_train = train[FEATURE_COLUMNS]
    y_train = train["RUL"]
    X_test = test[FEATURE_COLUMNS]
    y_test = test["RUL"]

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=2,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=100,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=2,
        ),
    }

    results = []
    prediction_table = test[["engine_id", "cycle", "RUL"]].copy()

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        predictions = predictions.clip(min=0)

        mae = mean_absolute_error(y_test, predictions)
        rmse = root_mean_squared_error(y_test, predictions)

        results.append({
            "model": name,
            "training_engines": train["engine_id"].nunique(),
            "test_engines": test["engine_id"].nunique(),
            "MAE": mae,
            "RMSE": rmse,
        })

        prediction_table[f"{name}_prediction"] = predictions

    results_df = pd.DataFrame(results)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results_path = REPORT_DIR / "official_test_results.csv"
    predictions_path = REPORT_DIR / "official_test_predictions.csv"

    results_df.to_csv(results_path, index=False)
    prediction_table.to_csv(predictions_path, index=False)

    print("\n--- Official C-MAPSS FD001 Test Results ---")
    print(results_df.round(2).to_string(index=False))
    print(f"\nResults saved to: {results_path}")
    print(f"Predictions saved to: {predictions_path}")


if __name__ == "__main__":
    evaluate_official_test()
