
from pathlib import Path

import pandas as pd

from predictive_maintenance.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

COLUMNS = [
    "engine_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]


def prepare_training_data() -> pd.DataFrame:
    """Load NASA FD001 training data and calculate raw RUL labels."""
    source = RAW_DATA_DIR / "train_FD001.txt"

    if not source.exists():
        raise FileNotFoundError(f"Training data not found: {source}")

    df = pd.read_csv(source, sep=r"\s+", header=None, names=COLUMNS)

    last_cycle = df.groupby("engine_id")["cycle"].transform("max")
    df["RUL"] = last_cycle - df["cycle"]

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output = PROCESSED_DATA_DIR / "train_FD001_with_RUL.csv"
    df.to_csv(output, index=False)

    print(f"Loaded rows: {len(df):,}")
    print(f"Unique engines: {df['engine_id'].nunique()}")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    print(f"Saved processed dataset: {output}")

    return df


if __name__ == "__main__":
    prepare_training_data()
