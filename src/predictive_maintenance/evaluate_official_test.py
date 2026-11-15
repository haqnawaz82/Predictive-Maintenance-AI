
import json
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from predictive_maintenance.config import (
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    REPORT_DIR,
)

FEATURE_COLUMNS = [
    "setting_1",
    "setting_2",
    "setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]


MODEL_CONFIGS = {
    "Random Forest": {
        "n_estimators": 100,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": 2,
    },
    "Extra Trees": {
        "n_estimators": 100,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": 2,
    },
}


def nasa_score(y_true, y_pred):
    """Calculate the asymmetric NASA C-MAPSS scoring function."""
    error = np.asarray(y_pred) - np.asarray(y_true)

    penalties = np.where(
        error < 0,
        np.exp(-error / 13) - 1,
        np.exp(error / 10) - 1,
    )

    return float(np.sum(penalties))


def evaluate_official_test(save_reports=True):
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
        "Random Forest": RandomForestRegressor(**MODEL_CONFIGS["Random Forest"]),
        "Extra Trees": ExtraTreesRegressor(**MODEL_CONFIGS["Extra Trees"]),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    prediction_table = test[["engine_id", "cycle", "RUL"]].copy()

    model_files = {
        "Random Forest": "random_forest_fd001.joblib",
        "Extra Trees": "extra_trees_fd001.joblib",
    }

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        # Save the fitted model.
        model_path = MODEL_DIR / model_files[name]
        joblib.dump(model, model_path)
        print(f"Model saved to: {model_path}")

        predictions = model.predict(X_test)
        predictions = predictions.clip(min=0)

        mae = mean_absolute_error(y_test, predictions)
        rmse = root_mean_squared_error(y_test, predictions)
        score = nasa_score(y_test, predictions)

        results.append({
            "model": name,
            "training_engines": train["engine_id"].nunique(),
            "test_engines": test["engine_id"].nunique(),
            "MAE": mae,
            "RMSE": rmse,
            "NASA_score": score,
        })

        prediction_table[f"{name}_prediction"] = predictions

    results_df = pd.DataFrame(results)

    # Save feature information.
    feature_path = MODEL_DIR / "feature_columns.json"

    with open(feature_path, "w", encoding="utf-8") as file:
        json.dump(FEATURE_COLUMNS, file, indent=4)

    # Save model and environment metadata.
    metadata = {
        "project": "AI-Based Machine Fault Detection with Limited Failure Data",
        "dataset": "NASA C-MAPSS FD001",
        "prediction_target": "Remaining Useful Life (RUL)",
        "training_engines": int(train["engine_id"].nunique()),
        "test_engines": int(test["engine_id"].nunique()),
        "training_rows": int(len(train)),
        "test_rows": int(len(test)),
        "target_column": "RUL",
        "feature_columns_file": "feature_columns.json",
        "models": {
            "Random Forest": {
                "file": model_files["Random Forest"],
                "configuration": MODEL_CONFIGS["Random Forest"],
            },
            "Extra Trees": {
                "file": model_files["Extra Trees"],
                "configuration": MODEL_CONFIGS["Extra Trees"],
            },
        },
        "software_versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }

    metadata_path = MODEL_DIR / "model_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    # Save evaluation reports only when requested.
    if save_reports:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)

        results_path = REPORT_DIR / "official_test_results.csv"
        predictions_path = REPORT_DIR / "official_test_predictions.csv"

        results_df.to_csv(results_path, index=False)
        prediction_table.to_csv(predictions_path, index=False)

        print(f"\nResults saved to: {results_path}")
        print(f"Predictions saved to: {predictions_path}")
    else:
        print("\nExisting evaluation reports were not modified.")

    print("\n--- Official C-MAPSS FD001 Test Results ---")
    print(results_df.round(2).to_string(index=False))
    print(f"\nFeature information saved to: {feature_path}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    evaluate_official_test()
