"""
src/eda_plots.py
Day 6: EDA - Correlation Heatmap and Scatter Plots for Polyhouse Sensor Data.
Generates publication-ready figures saved under reports/figures/.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
INPUT_PATH = Path("data/interim/02_cleaned.parquet")
FIGURES_DIR = Path("reports/figures")


def generate_eda_visualizations():
    # 1. Load Day 4/5 cleaned interim dataset
    if not INPUT_PATH.exists():
        input_file = Path("data/interim/02_cleaned.csv")
        df = pd.read_csv(input_file, parse_dates=["timestamp"])
    else:
        df = pd.read_parquet(INPUT_PATH)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(df)} rows for EDA visualization.\n" + "=" * 50)

    target_col = "harvested_yield_kg" if "harvested_yield_kg" in df.columns else "yield_kg"
    num_cols = ["temperature_c", "humidity_pct", "co2_ppm", target_col]

    # Clean feature names for plot display
    display_labels = {
        "temperature_c": "Temperature (°C)",
        "humidity_pct": "Humidity (%)",
        "co2_ppm": "CO2 (PPM)",
        target_col: "Yield (kg)"
    }

    # -------------------------------------------------------------
    # Plot 1: Correlation Heatmap
    # -------------------------------------------------------------
    plt.figure(figsize=(7, 5))
    corr_matrix = df[num_cols].corr()
    
    # Rename index/columns for plot hygiene
    plot_corr = corr_matrix.rename(index=display_labels, columns=display_labels)
    
    sns.heatmap(
        plot_corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5
    )
    plt.title("Polyhouse Sensor Correlation Heatmap", fontsize=12, pad=12)
    plt.tight_layout()
    heatmap_path = FIGURES_DIR / "correlation_heatmap.png"
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"Saved: {heatmap_path}")

    # -------------------------------------------------------------
    # Plot 2: 3-Panel Scatter Plots (Features vs Yield)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    features = [
        ("temperature_c", "Temperature (°C)", "tab:red"),
        ("humidity_pct", "Humidity (%)", "tab:blue"),
        ("co2_ppm", "CO2 (PPM)", "tab:green")
    ]

    for ax, (col, label, color) in zip(axes, features):
        ax.scatter(df[col], df[target_col], alpha=0.4, color=color, edgecolors="none")
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("Yield (kg)", fontsize=10)
        ax.set_title(f"{label} vs. Yield", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Sensor Variables vs. Mushroom Harvest Yield", fontsize=13, y=1.02)
    plt.tight_layout()
    scatter_path = FIGURES_DIR / "sensor_scatter_plots.png"
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"Saved: {scatter_path}")

    print("=" * 50 + "\nEDA Visualizations successfully generated!")


if __name__ == "__main__":
    generate_eda_visualizations()