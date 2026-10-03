# Polyhouse Data Quality & Descriptive Statistics Report

**Total Observation Rows:** 365

**Date Range:** 2024-01-01 00:00:00 to 2024-12-30 00:00:00

## 1. Summary Statistics & Coefficient of Variation (CV)

|                    |   count |   mean |   median |   std |   cv |    min |     max |
|:-------------------|--------:|-------:|---------:|------:|-----:|-------:|--------:|
| temperature_c      |     365 |  21.99 |    21.97 |  1.41 | 0.06 |  18.15 |   26.37 |
| humidity_pct       |     365 |  86.74 |    86.74 |  3.07 | 0.04 |  78.11 |   94.79 |
| co2_ppm            |     365 | 901.16 |   904    | 78.27 | 0.09 | 608    | 1154    |
| harvested_yield_kg |     365 |  17.14 |    17.13 |  0.68 | 0.04 |  15.31 |   18.85 |

## 2. Key Insights & Agritech Findings

- **Yield Skewness:** Mean yield is `17.14` kg, whereas median yield is `17.13` kg. The difference highlights occasional high-yield flush days.
- **Relative Variability (CV):** CO2 exhibits a CV of `0.09`, reflecting active ventilation control within the polyhouse environment.
- **Humidity Range:** Relative humidity remains tightly bound between `78.11%` and `94.79%`, optimal for mushroom pinhead growth.