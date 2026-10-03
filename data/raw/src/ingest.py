from pathlib import Path
import pandas as pd

# Define paths
RAW_DATA_PATH = Path("data/raw/polyhouse_sensors.csv")
INTERIM_DATA_DIR = Path("data/interim")
INTERIM_FILE_PATH = INTERIM_DATA_DIR / "01_loaded.parquet"


def ingest_sensor_data(filepath: Path) -> pd.DataFrame:
    """Ingests polyhouse CSV logs with date parsing and structural verification."""
    print(f"Loading data from: {filepath}\n" + "=" * 50)

    # 1. Read CSV with date parsing
    df = pd.read_csv(
    filepath, parse_dates=["timestamp"]
)

    # 2. Print Shape & Info
    print(f"Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")

    print("--- Data Types & Non-Null Counts ---")
    df.info()
    print("\n" + "=" * 50)

    # 3. Inspect Head
    print("--- First 5 Rows ---")
    print(df.head())
    print("\n" + "=" * 50)

    # 4. Statistical Summary (Unit & Range Check)
    print("--- Statistical Summary (Check for anomalous ranges) ---")
    print(df.describe().T)
    print("\n" + "=" * 50)

    return df


def save_interim_snapshot(df: pd.DataFrame, output_path: Path) -> None:
    """Saves the un-transformed interim snapshot preserving data types."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to Parquet to preserve schema and speed up future reads
    try:
        df.to_parquet(output_path, index=False)
        print(f"Successfully saved snapshot to: {output_path}")
    except ImportError:
        # Fallback to CSV if pyarrow/fastparquet is not installed
        fallback_csv = output_path.with_suffix(".csv")
        df.to_csv(fallback_csv, index=False)
        print(f"`pyarrow` missing. Saved CSV fallback to: {fallback_csv}")


if __name__ == "__main__":
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw file not found at {RAW_DATA_PATH}. "
            "Please place your CSV in data/raw/ or run the generator script."
        )

    raw_df = ingest_sensor_data(RAW_DATA_PATH)
    save_interim_snapshot(raw_df, INTERIM_FILE_PATH)