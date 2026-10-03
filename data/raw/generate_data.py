# generate_data.py
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 365

data = {
    "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
    "temperature_c": rng.normal(22, 1.5, n).round(2),
    "humidity_pct": np.clip(rng.normal(87, 3, n), 75, 98).round(2),
    "co2_ppm": rng.normal(900, 80, n).round(0),
}

# Simple synthetic yield relationship
data["harvested_yield_kg"] = (
    8 + 0.3 * data["temperature_c"] + 0.05 * data["humidity_pct"] - 0.002 * data["co2_ppm"] + rng.normal(0, 0.5, n)
).round(2)

df = pd.DataFrame(data)
df.to_csv("data/raw/polyhouse_sensors.csv", index=False)
print("Saved raw dataset to data/raw/polyhouse_sensors.csv")