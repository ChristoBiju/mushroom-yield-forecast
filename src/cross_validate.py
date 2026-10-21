"""
src/cross_validate.py
Day 12: Cross-Validation & Overfitting Analysis using TimeSeriesSplit.
Evaluates Linear Regression and Random Forest across chronological folds
and writes out reports/cv_results.md.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

# File paths
TRAIN_PATH = Path("data/processed/train.parquet")
REPORT_PATH = Path("reports/cv_results.md")


def run_cross_validation():
    # 1. Load training dataset (Never use test set during CV!)
    if not TRAIN_PATH.exists():
        raise FileNotFoundError("Train parquet file not found! Complete Day 8 first.")

    train_df = pd.read_parquet(TRAIN_PATH)

    target_col = "target" if "target" in train_df.columns else "harvested_yield_kg"
    feature_cols = [c for c in train_df.columns if c not in ["timestamp", target_col]]

    X = train_df[feature_cols]
    y = train_df[target_col]

    # 2. Setup TimeSeriesSplit (e.g., 3 splits)
    tscv = TimeSeriesSplit(n_splits=3)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }

    cv_results = {}

    print("Running TimeSeriesSplit Cross-Validation...\n" + "=" * 50)

    for name, model in models.items():
        fold_maes = []
        fold_rmses = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)

            fold_maes.append(mean_absolute_error(y_val, preds))
            fold_rmses.append(np.sqrt(mean_squared_error(y_val, preds)))

        cv_results[name] = {
            "mean_mae": np.mean(fold_maes),
            "std_mae": np.std(fold_maes),
            "mean_rmse": np.mean(fold_rmses),
            "std_rmse": np.std(fold_rmses),
            "fold_maes": fold_maes
        }

        print(f"[{name}]")
        print(f"  Mean CV MAE:  {np.mean(fold_maes):.4f} (±{np.std(fold_maes):.4f})")
        print(f"  Mean CV RMSE: {np.mean(fold_rmses):.4f} (±{np.std(fold_rmses):.4f})\n" + "-" * 40)

    # 3. Write Cross-Validation Report
    report_content = f"""# Cross-Validation & Overfitting Analysis Report

## Overview
This report documents the temporal cross-validation results using scikit-learn's `TimeSeriesSplit` ($n\_splits=3$) on the training dataset for Day 12. 

## Methodology
- **Validator:** `TimeSeriesSplit` (ensures training data strictly precedes validation data chronologically, preventing future data leakage)[cite: 15, 16].
- **Data Source:** `data/processed/train.parquet` (Test data remains untouched and locked away).

## Cross-Validation Results Summary
| Model | Mean CV MAE | Std CV MAE | Mean CV RMSE | Std CV RMSE |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | `{cv_results['Linear Regression']['mean_mae']:.4f}` | `{cv_results['Linear Regression']['std_mae']:.4f}` | `{cv_results['Linear Regression']['mean_rmse']:.4f}` | `{cv_results['Linear Regression']['std_rmse']:.4f}` |
| **Random Forest** | `{cv_results['Random Forest']['mean_mae']:.4f}` | `{cv_results['Random Forest']['std_mae']:.4f}` | `{cv_results['Random Forest']['mean_rmse']:.4f}` | `{cv_results['Random Forest']['std_rmse']:.4f}` |

## Overfitting & Variance Notes
- **Stability Across Folds:** Standard deviations across folds reflect how sensitive models are to shifting seasonal or environmental sensor windows.
- **Overfitting Risk:** Random Forest exhibits higher training capacity, which can lead to larger train-vs-validation error gaps if tree depth is unconstrained. Day 13 hyperparameter tuning will address regularization.
"""

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"Cross-validation report saved successfully to: {REPORT_PATH}[cite: 15, 17]")
    print("=" * 50 + "\nDay 12 Cross-Validation Complete!")


if __name__ == "__main__":
    run_cross_validation()