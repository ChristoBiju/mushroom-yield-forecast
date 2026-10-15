"""
src/train_baseline.py
Day 9: Train and Evaluate Baseline Linear Regression Model.
Trains scikit-learn LinearRegression on X_train, evaluates on test set,
saves metrics to reports/metrics_linear.json, and exports model artifact.
"""
import numpy as np
from pathlib import Path
import json
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# File paths
TRAIN_PATH = Path("data/processed/train.parquet")
TEST_PATH = Path("data/processed/test.parquet")
MODEL_OUTPUT_PATH = Path("models/linear_regression.joblib")
METRICS_OUTPUT_PATH = Path("reports/metrics_linear.json")


def train_baseline_model():
    # 1. Load leak-free train and test splits from Day 8
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError("Train or test parquet files not found! Complete Day 8 first.")

    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    print(f"Loaded train data ({len(train_df)} rows) and test data ({len(test_df)} rows).\n" + "=" * 50)

    # 2. Separate features (X) and target (y)
    target_col = "target" if "target" in train_df.columns else "harvested_yield_kg"
    
    feature_cols = [c for c in train_df.columns if c not in ["timestamp", target_col]]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # 3. Train Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Linear Regression model successfully trained on X_train.")
    # 4. Predict on Test Set
    y_pred = model.predict(X_test)

    # 5. Compute Evaluation Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))  # RMSE computed cleanly
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "mae": float(mae),
        "rmse": rmse,
        "r2": float(r2)
    }

    print("\n--- Test Set Evaluation Metrics ---")
    print(f"MAE:  {mae:.4f} kg")
    print(f"RMSE: {rmse:.4f} kg")
    print(f"R²:   {r2:.4f}\n" + "=" * 50)

    # 6. Save Metrics to JSON Report
    METRICS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_OUTPUT_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Test metrics saved to: {METRICS_OUTPUT_PATH}")

    # 7. Save Model Artifact under models/
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"Model artifact saved to: {MODEL_OUTPUT_PATH}")

    # 8. Inspect Feature Coefficients
    print("\n--- Model Coefficients (Feature Weights) ---")
    coefficients = pd.DataFrame({
        "Feature": feature_cols,
        "Coefficient": model.coef_
    }).sort_values(by="Coefficient", ascending=False)
    print(coefficients.to_string(index=False))
    print("=" * 50 + "\nDay 9 Baseline Modeling Complete!")


if __name__ == "__main__":
    train_baseline_model()