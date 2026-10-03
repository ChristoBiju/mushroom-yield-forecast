# Day 6: Exploratory Data Analysis (EDA) Notes

## 1. Visualizations
- **Heatmap:** Saved to `reports/figures/correlation_heatmap.png`
- **Scatter Plots:** Saved to `reports/figures/sensor_scatter_plots.png`

## 2. Key Agritech Takeaways & Mushroom Biology Links
1. **Humidity Sweet Spot:** Yield peaks in high humidity ranges (~80–90%), aligned with mushroom pinhead formation and fruiting moisture requirements.
2. **CO2 Impact:** CO2 levels display moderate correlation with yield. High CO2 levels favor vegetative mycelium growth, while lower CO2 triggers fruiting.
3. **Non-Linear Relationships:** Scatter plots reveal curvature near boundary conditions (too cold or too hot reduces yield growth), indicating that simple linear regression may underfit non-linear environmental responses.
4. **Modeling Implications:** Feature engineering (e.g., non-linear terms or decision-tree based models like Random Forest) should be evaluated in Phase 2 to capture curved sensor response boundaries.