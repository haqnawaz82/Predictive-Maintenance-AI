
import pandas as pd
from predictive_maintenance.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

COLUMNS = [
    "engine_id", "cycle", "setting_1", "setting_2", "setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]


def prepare_official_test_data():
    test_path = RAW_DATA_DIR / "test_FD001.txt"
    rul_path = RAW_DATA_DIR / "RUL_FD001.txt"

    test = pd.read_csv(
        test_path, sep=r"\s+", header=None, names=COLUMNS
    )
    official_rul = pd.read_csv(
        rul_path, sep=r"\s+", header=None, names=["RUL"]
    )

    last_rows = (
        test.sort_values(["engine_id", "cycle"])
        .groupby("engine_id", as_index=False)
        .tail(1)
        .sort_values("engine_id")
        .reset_index(drop=True)
    )

    if len(last_rows) != len(official_rul):
        raise ValueError(
            f"Test engines ({len(last_rows)}) do not match "
            f"RUL labels ({len(official_rul)})."
        )

    last_rows["RUL"] = official_rul["RUL"].to_numpy()

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output = PROCESSED_DATA_DIR / "test_FD001_official.csv"
    last_rows.to_csv(output, index=False)

    print(f"Test rows loaded: {len(test):,}")
    print(f"Test engines: {len(last_rows)}")
    print(f"Missing values: {int(last_rows.isna().sum().sum())}")
    print(
        f"Official RUL range: {last_rows['RUL'].min()} "
        f"to {last_rows['RUL'].max()}"
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    prepare_official_test_data()
