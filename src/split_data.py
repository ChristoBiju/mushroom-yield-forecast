"""
src/split_data.py
Day 8: Chronological Train/Test Split & Data Leakage Prevention.
Splits data temporally (80/20), fits MinMaxScaler strictly on X_train,
and saves isolated train/test arrays under data/processed/.
"""

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# File paths
INPUT_PATH = Path("data/processed/03_features.parquet")
PROCESSED_DIR = Path("data/processed")
SCALER_PATH = Path("models/minmax_scaler.joblib")


def split_and_leakage_prevention():
    # 1. Load processed dataset
    if not INPUT_PATH.exists():
        fallback_csv = INPUT_PATH.with_suffix(".csv")
        df = pd.read_csv(fallback_csv, parse_dates=["timestamp"])
    else:
        df = pd.read_parquet(INPUT_PATH)

    print(f"Loaded {len(df)} rows for chronological splitting.\n" + "=" * 50)

    # 2. Ensure chronological order by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    target_col = "harvested_yield_kg" if "harvested_yield_kg" in df.columns else "yield_kg"

    # Separate features and target
    # Exclude timestamp and target from feature list
    feature_cols = [c for c in df.columns if c not in ["timestamp", target_col]]
    
    X = df[feature_cols]
    y = df[target_col]
    timestamps = df["timestamp"]

    # 3. Chronological 80/20 Split
    split_index = int(len(df) * 0.80)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    train_dates = (timestamps.iloc[0], timestamps.iloc[split_index - 1])
    test_dates = (timestamps.iloc[split_index], timestamps.iloc[-1])

    print(f"Train Period: {train_dates[0].date()} to {train_dates[1].date()} ({len(X_train)} rows)")
    print(f"Test Period:  {test_dates[0].date()} to {test_dates[1].date()} ({len(X_test)} rows)\n" + "=" * 50)

    # Verify no test rows appear before train cutoff date
    assert timestamps.iloc[split_index] > timestamps.iloc[split_index - 1], "Chronological order violation!"

    # 4. Prevent Data Leakage: Fit Scaler on Train ONLY, Transform Test
    scaler = MinMaxScaler()
    
    # Fit and transform training features
    X_train_scaled_array = scaler.fit_transform(X_train)
    
    # Transform test features using training bounds
    X_test_scaled_array = scaler.transform(X_test)

    # Rebuild DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=feature_cols)
    X_test_scaled = pd.DataFrame(X_test_scaled_array, columns=feature_cols)

    # Save the leak-free scaler artifact
    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Leak-free MinMaxScaler artifact saved to: {SCALER_PATH}")

    # 5. Persist Split Artifacts under data/processed/
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    X_train_scaled.assign(timestamp=timestamps.iloc[:split_index].values, target=y_train.values).to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    X_test_scaled.assign(timestamp=timestamps.iloc[split_index:].values, target=y_test.values).to_parquet(PROCESSED_DIR / "test.parquet", index=False)
    
    print(f"Saved train split to: {PROCESSED_DIR / 'train.parquet'}")
    print(f"Saved test split to: {PROCESSED_DIR / 'test.parquet'}")
    print("=" * 50 + "\nDay 8 Split & Data Leakage Prevention Complete!")


if __name__ == "__main__":
    split_and_leakage_prevention()