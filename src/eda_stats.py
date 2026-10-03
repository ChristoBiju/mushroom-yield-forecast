"""
src/eda_stats.py
Day 5: Generate descriptive statistics, calculate Coefficient of Variation (CV),
and export a clean Data Quality Report markdown file for stakeholders.
"""

from pathlib import Path
import pandas as pd

# File paths
INPUT_PATH = Path("data/interim/02_cleaned.parquet")
REPORT_PATH = Path("reports/data_quality.md")


def generate_data_quality_report():
    # 1. Fallback check for CSV if parquet is missing
    if not INPUT_PATH.exists():
        input_file = Path("data/interim/02_cleaned.csv")
        df = pd.read_csv(input_file, parse_dates=["timestamp"])
    else:
        df = pd.read_parquet(INPUT_PATH)

    print(f"Loaded {len(df)} rows for descriptive analysis.\n" + "=" * 50)

    # 2. Select numerical columns & calculate summary statistics
    target_col = "harvested_yield_kg" if "harvested_yield_kg" in df.columns else "yield_kg"
    num_cols = ["temperature_c", "humidity_pct", "co2_ppm", target_col]

    # .describe().T gives count, mean, std, min, 25%, 50%, 75%, max
    summary = df[num_cols].describe().T

    # Compute Coefficient of Variation (CV = std / mean)
    summary["cv"] = summary["std"] / summary["mean"]

    # Compute median explicitly for comparison against mean
    summary["median"] = df[num_cols].median()

    # Round figures to 2 decimal places for clean reporting
    summary = summary.round(2)

    # 3. Extract metadata
    min_date = df["timestamp"].min()
    max_date = df["timestamp"].max()
    total_rows = len(df)

    # 4. Build Markdown Report Content
    report_lines = [
        "# Polyhouse Data Quality & Descriptive Statistics Report\n",
        f"**Total Observation Rows:** {total_rows}\n",
        f"**Date Range:** {min_date} to {max_date}\n",
        "## 1. Summary Statistics & Coefficient of Variation (CV)\n",
        summary[["count", "mean", "median", "std", "cv", "min", "max"]].to_markdown(),
        "\n## 2. Key Insights & Agritech Findings\n",
        f"- **Yield Skewness:** Mean yield is `{summary.loc[target_col, 'mean']}` kg, whereas median yield is `{summary.loc[target_col, 'median']}` kg. The difference highlights occasional high-yield flush days.",
        f"- **Relative Variability (CV):** CO2 exhibits a CV of `{summary.loc['co2_ppm', 'cv']}`, reflecting active ventilation control within the polyhouse environment.",
        f"- **Humidity Range:** Relative humidity remains tightly bound between `{summary.loc['humidity_pct', 'min']}%` and `{summary.loc['humidity_pct', 'max']}%`, optimal for mushroom pinhead growth.",
    ]

    # 5. Export Report to reports/data_quality.md
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Data quality report successfully written to: {REPORT_PATH}")


if __name__ == "__main__":
    generate_data_quality_report()