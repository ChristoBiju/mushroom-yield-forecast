# Data Cleaning Log - Polyhouse Sensor Data

## 1. Missing Values Strategy
- **Sensors (`temperature_c`, `humidity_pct`, `co2_ppm`):** Forward-fill (`ffill`) for short gaps up to 2 consecutive missing hours.
- **Target (`harvested_yield_kg`):** Drop rows where yield is missing. Never impute target values.

## 2. Duplicate Strategy
- Drop duplicate timestamps, keeping the latest reading (`keep='last'`).

## 3. Validity Threshold Rules
- `humidity_pct`: 50% to 100%
- `temperature_c`: 10°C to 35°C
- `co2_ppm`: 400 PPM to 2000 PPM
- `harvested_yield_kg`: Non-null