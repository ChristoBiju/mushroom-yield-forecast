"""
src/train_rf.py
Day 11: Train and Evaluate Random Forest Regressor.
Trains scikit-learn RandomForestRegressor, evaluates on test set,
plots feature importances, saves metrics report, and exports model artifact.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# File paths
TRAIN_PATH = Path("data/processed/train.parquet")
TEST_PATH = Path("data/processed/test.parquet")
MODEL_OUTPUT_PATH = Path("models/random_forest.joblib")
METRICS_OUTPUT_PATH = Path("reports/metrics_rf.json")
FIGURES_DIR = Path("reports/figures")


def train_random_forest():
    # 1. Load leak-free training and testing datasets
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError("Train or test parquet files not found! Complete Day 8 first.")

    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    target_col = "target" if "target" in train_df.columns else "harvested_yield_kg"
    feature_cols = [c for c in train_df.columns if c not in ["timestamp", target_col]]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # 2. Train Random Forest Regressor with default sensible settings
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    print("Random Forest model successfully trained on X_train.")

    # 3. Predict on Test Set
    y_pred = rf_model.predict(X_test)

    # 4. Compute Evaluation Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "mae": float(mae),
        "rmse": rmse,
        "r2": float(r2)
    }

    print("\n--- Random Forest Test Metrics ---")
    print(f"MAE:  {mae:.4f} kg")
    print(f"RMSE: {rmse:.4f} kg")
    print(f"R²:   {r2:.4f}\n" + "=" * 50)

    # Save metrics report
    METRICS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_OUTPUT_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    # 5. Save Model Artifact
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_model, MODEL_OUTPUT_PATH)
    print(f"Model artifact saved to: {MODEL_OUTPUT_PATH}")

    # 6. Plot and Save Feature Importances
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=True)

    plt.figure(figsize=(10, 6))
    importances.plot(kind="barh", color="forestgreen")
    plt.title("Random Forest Feature Importances")
    plt.xlabel("Mean Decrease in Impurity")
    plt.tight_layout()
    
    importance_plot_path = FIGURES_DIR / "rf_feature_importances.png"
    plt.savefig(importance_plot_path, dpi=300)
    plt.close()
    print(f"Feature importance plot saved to: {importance_plot_path}")

    print("=" * 50 + "\nDay 11 Random Forest Modeling Complete!")


if __name__ == "__main__":
    train_random_forest()