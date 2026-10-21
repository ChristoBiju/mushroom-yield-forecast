# Cross-Validation & Overfitting Analysis Report

## Overview
This report documents the temporal cross-validation results using scikit-learn's `TimeSeriesSplit` ($n\_splits=3$) on the training dataset for Day 12. 

## Methodology
- **Validator:** `TimeSeriesSplit` (ensures training data strictly precedes validation data chronologically, preventing future data leakage)[cite: 15, 16].
- **Data Source:** `data/processed/train.parquet` (Test data remains untouched and locked away).

## Cross-Validation Results Summary
| Model | Mean CV MAE | Std CV MAE | Mean CV RMSE | Std CV RMSE |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | `0.4412` | `0.0288` | `0.5686` | `0.0469` |
| **Random Forest** | `0.4635` | `0.0222` | `0.5855` | `0.0248` |

## Overfitting & Variance Notes
- **Stability Across Folds:** Standard deviations across folds reflect how sensitive models are to shifting seasonal or environmental sensor windows.
- **Overfitting Risk:** Random Forest exhibits higher training capacity, which can lead to larger train-vs-validation error gaps if tree depth is unconstrained. Day 13 hyperparameter tuning will address regularization.
