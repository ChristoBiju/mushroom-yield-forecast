"""
src/clean.py
Day 4: Data Cleaning - Missing value handling, duplicate removal, and outlier filtering.
"""

from pathlib import Path
import pandas as pd

# Paths
INPUT_PATH = Path("data/interim/01_loaded.csv")  # Change to .parquet if available
OUTPUT_PATH = Path("data/interim/02_cleaned.parquet")


def clean_polyhouse_data(input_file: Path, output_file: Path):
    # 1. Load Day 3 interim data
    if not input_file.exists():
        input_file = Path("data/interim/01_loaded.parquet")

    df = pd.read_csv(input_file, parse_dates=["timestamp"])
    initial_rows = len(df)
    print(f"Loaded {initial_rows} rows from {input_file}\n" + "=" * 50)

    # 2. Missing Value Report (Before Cleaning)
    print("--- Missing Value Report (Before Fixes) ---")
    missing_count = df.isna().sum()
    missing_pct = (df.isna().sum() / len(df)) * 100
    report_df = pd.DataFrame({"Missing Count": missing_count, "Missing %": missing_pct})
    print(report_df)
    print("\n" + "=" * 50)

    # 3. Apply Validity Threshold Rules
    valid_mask = (
        df["humidity_pct"].between(50, 100)
        & df["temperature_c"].between(10, 35)
        & df["co2_ppm"].between(400, 2000)
    )
    df = df[valid_mask].copy()

    # 4. Short-gap forward fill for sensor columns only
    sensor_cols = ["temperature_c", "humidity_pct", "co2_ppm"]
    df[sensor_cols] = df[sensor_cols].ffill(limit=2)

    # 5. Drop rows with null target variable (harvested_yield_kg)
    df = df.dropna(subset=["harvested_yield_kg"])

    # 6. Remove duplicate timestamps
    df = df.drop_duplicates(subset=["timestamp"], keep="last")

    # 7. Final Verification & Logging
    final_rows = len(df)
    dropped_rows = initial_rows - final_rows
    print(f"Cleaning Complete!")
    print(f"Initial Rows: {initial_rows} | Final Rows: {final_rows} | Dropped: {dropped_rows}")
    print(f"Target column ('harvested_yield_kg') null count: {df['harvested_yield_kg'].isna().sum()}")

    # 8. Export Cleaned Dataset
    output_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(output_file, index=False)
        print(f"Saved cleaned snapshot to: {output_file}")
    except ImportError:
        csv_fallback = output_file.with_suffix(".csv")
        df.to_csv(csv_fallback, index=False)
        print(f"`pyarrow` missing. Saved CSV fallback to: {csv_fallback}")


if __name__ == "__main__":
    clean_polyhouse_data(INPUT_PATH, OUTPUT_PATH)