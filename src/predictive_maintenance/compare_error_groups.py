import pandas as pd
data = pd.read_csv("reports/official_test_predictions.csv")
data["RUL_group"] = pd.cut(
    data["RUL"],
    bins=[-1, 30, 60, 90, 120, 200],
    labels=["0-30", "31-60", "61-90", "91-120", "121+"],
)
models = {
    "Random Forest": "Random Forest_prediction",
    "Extra Trees": "Extra Trees_prediction",
}
results = []
for group, group_data in data.groupby("RUL_group", observed=True):
    for model_name, column in models.items():
        errors = group_data[column] - group_data["RUL"]
        results.append({
            "RUL_group": group,
            "model": model_name,
            "engines": len(group_data),
            "mean_error": errors.mean(),
            "MAE": errors.abs().mean(),
        })
summary = pd.DataFrame(results)
print("\n--- Model Error Comparison by Actual RUL Group ---")
print(summary.round(2).to_string(index=False))
