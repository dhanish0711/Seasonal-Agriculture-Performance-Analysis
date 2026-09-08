import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Set styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['figure.titlesize'] = 15
plt.rcParams['figure.titleweight'] = 'bold'

os.makedirs('output_visualizations', exist_ok=True)
os.makedirs('data_cleaned', exist_ok=True)

# 1. Load Data
df = pd.read_csv('seasonal_agriculture_performance_dataset.csv')
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# 2. Data Cleaning & Imputation
# Impute Yield_Tonnes_Ha using Production_Tonnes / Farm_Area_Hectares
yield_null_idx = df['Yield_Tonnes_Ha'].isnull()
df.loc[yield_null_idx, 'Yield_Tonnes_Ha'] = (
    df.loc[yield_null_idx, 'Production_Tonnes'] / df.loc[yield_null_idx, 'Farm_Area_Hectares']
).round(2)

# Impute Rainfall_mm using (Season, State) median
rainfall_group = df.groupby(['Season', 'State'])['Rainfall_mm'].transform('median')
df['Rainfall_mm'] = df['Rainfall_mm'].fillna(rainfall_group).fillna(df['Rainfall_mm'].median())

# Impute Soil_Moisture_pct using (Season, Irrigation_Method) median
moisture_group = df.groupby(['Season', 'Irrigation_Method'])['Soil_Moisture_pct'].transform('median')
df['Soil_Moisture_pct'] = df['Soil_Moisture_pct'].fillna(moisture_group).fillna(df['Soil_Moisture_pct'].median())

# 3. Feature Engineering
df['Profit_Margin_pct'] = (df['Profit_INR'] / df['Revenue_INR'] * 100).round(2)
df['Cost_per_Hectare'] = (df['Total_Cost_INR'] / df['Farm_Area_Hectares']).round(2)
df['Revenue_per_Hectare'] = (df['Revenue_INR'] / df['Farm_Area_Hectares']).round(2)
df['Profit_per_Hectare'] = (df['Profit_INR'] / df['Farm_Area_Hectares']).round(2)
df['Water_Productivity_INR_m3'] = (df['Profit_INR'] / df['Water_Used_m3']).round(2)
df['Is_Profitable'] = (df['Profit_INR'] > 0).astype(int)

# Save cleaned dataset
df.to_csv('data_cleaned/cleaned_seasonal_agriculture_performance.csv', index=False)
print("Saved cleaned data to data_cleaned/cleaned_seasonal_agriculture_performance.csv")

# 4. Statistical Tests
# ANOVA for macro metrics across Season
stats_results = {}
for col in ['Yield_Tonnes_Ha', 'Profit_INR', 'Water_Efficiency_t_per_1000m3', 'Disease_Pest_Risk_pct', 'Total_Cost_INR', 'Water_Used_m3']:
    k_grp = df[df['Season'] == 'Kharif'][col].dropna()
    r_grp = df[df['Season'] == 'Rabi'][col].dropna()
    z_grp = df[df['Season'] == 'Zaid'][col].dropna()
    f_stat, p_val = stats.f_oneway(k_grp, r_grp, z_grp)
    kw_stat, kw_p = stats.kruskal(k_grp, r_grp, z_grp)
    stats_results[col] = {
        'anova_f': float(f_stat),
        'anova_p': float(p_val),
        'kruskal_h': float(kw_stat),
        'kruskal_p': float(kw_p)
    }

# Crop-specific Yield ANOVAs
crop_yield_anova = {}
for crop in df['Crop'].unique():
    c_df = df[df['Crop'] == crop]
    groups = [c_df[c_df['Season'] == s]['Yield_Tonnes_Ha'].dropna() for s in ['Kharif', 'Rabi', 'Zaid']]
    f_stat, p_val = stats.f_oneway(*groups)
    crop_yield_anova[crop] = {'f_stat': float(f_stat), 'p_value': float(p_val)}

# Welch's t-test for Zaid Drip vs Flood profit
zaid_drip = df[(df['Season'] == 'Zaid') & (df['Irrigation_Method'] == 'Drip')]['Profit_INR']
zaid_flood = df[(df['Season'] == 'Zaid') & (df['Irrigation_Method'] == 'Flood')]['Profit_INR']
t_stat, t_pval = stats.ttest_ind(zaid_drip, zaid_flood, equal_var=False)

stats_results['zaid_drip_vs_flood_ttest'] = {
    't_stat': float(t_stat),
    'p_value': float(t_pval),
    'drip_mean_profit': float(zaid_drip.mean()),
    'flood_mean_profit': float(zaid_flood.mean())
}

# Chi-Square test: Irrigation Method vs Season
contingency_irr_season = pd.crosstab(df['Irrigation_Method'], df['Season'])
chi2, p_chi2, dof, _ = stats.chi2_contingency(contingency_irr_season)
stats_results['chi2_irrigation_season'] = {
    'chi2': float(chi2),
    'p_value': float(p_chi2),
    'dof': int(dof)
}

# 5. Machine Learning Models
feature_cols = [
    'Farm_Area_Hectares', 'Rainfall_mm', 'Avg_Temperature_C', 'Humidity_pct',
    'Sunlight_Hours_Day', 'Soil_pH', 'Soil_Moisture_pct', 'Nitrogen_kg_ha',
    'Phosphorus_kg_ha', 'Potassium_kg_ha', 'Fertilizer_kg_ha', 'Pesticide_Litre_ha',
    'Seed_Quality_Score', 'Water_Used_m3', 'Disease_Pest_Risk_pct'
]

# Add dummy encoding for categorical features
X = pd.get_dummies(df[feature_cols + ['Crop', 'Season', 'Irrigation_Method', 'State']], drop_first=True)
y_yield = df['Yield_Tonnes_Ha']
y_profit = df['Profit_INR']

X_train, X_test, y_y_train, y_y_test = train_test_split(X, y_yield, test_size=0.2, random_state=42)
_, _, y_p_train, y_p_test = train_test_split(X, y_profit, test_size=0.2, random_state=42)

# Yield Model
rf_yield = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_yield.fit(X_train, y_y_train)
y_y_pred = rf_yield.predict(X_test)
r2_yield = r2_score(y_y_test, y_y_pred)
rmse_yield = np.sqrt(mean_squared_error(y_y_test, y_y_pred))
mae_yield = mean_absolute_error(y_y_test, y_y_pred)

# Profit Model
rf_profit = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_profit.fit(X_train, y_p_train)
y_p_pred = rf_profit.predict(X_test)
r2_profit = r2_score(y_p_test, y_p_pred)
rmse_profit = np.sqrt(mean_squared_error(y_p_test, y_p_pred))
mae_profit = mean_absolute_error(y_p_test, y_p_pred)

# Feature Importance
importances_profit = pd.Series(rf_profit.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
importances_yield = pd.Series(rf_yield.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)

ml_results = {
    'yield_model': {'r2': float(r2_yield), 'rmse': float(rmse_yield), 'mae': float(mae_yield)},
    'profit_model': {'r2': float(r2_profit), 'rmse': float(rmse_profit), 'mae': float(mae_profit)},
    'top_profit_features': importances_profit.to_dict(),
    'top_yield_features': importances_yield.to_dict()
}

with open('data_cleaned/statistical_ml_summary.json', 'w') as f:
    json.dump({'stats': stats_results, 'crop_yield_anova': crop_yield_anova, 'ml': ml_results}, f, indent=2)
print("Saved summary statistics and ML metrics.")

# 6. Generate 5 High-Resolution Publication Figures
palette_season = {'Kharif': '#1f77b4', 'Rabi': '#2ca02c', 'Zaid': '#d62728'}

# FIGURE 1: Seasonal Macro Metrics
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Seasonal Agricultural Dynamics: Environmental & Economic Divergence', fontsize=16, fontweight='bold')

# Panel 1: Rainfall vs Temperature
sns.scatterplot(data=df, x='Avg_Temperature_C', y='Rainfall_mm', hue='Season', palette=palette_season, alpha=0.6, ax=axes[0, 0], s=35)
axes[0, 0].set_title('Seasonal Climatic Clusters: Temperature vs Rainfall')
axes[0, 0].set_xlabel('Average Temperature (°C)')
axes[0, 0].set_ylabel('Rainfall (mm)')

# Panel 2: Profit by Season (Violin + Boxplot)
sns.boxplot(data=df, x='Season', y='Profit_INR', palette=palette_season, ax=axes[0, 1], width=0.4, showfliers=False)
axes[0, 1].axhline(0, color='black', linestyle='--', linewidth=1.2, label='Break-even (₹0)')
axes[0, 1].set_title('Distribution of Net Profit by Season (₹ INR)')
axes[0, 1].set_ylabel('Net Profit (INR)')
axes[0, 1].legend(loc='upper right')

# Panel 3: Pest Risk by Season
sns.barplot(data=df, x='Season', y='Disease_Pest_Risk_pct', palette=palette_season, ax=axes[1, 0], ci=95, capsize=0.1)
axes[1, 0].set_title('Disease & Pest Risk (%) by Season (with 95% CI)')
axes[1, 0].set_ylabel('Pest Risk (%)')
for p in axes[1, 0].patches:
    axes[1, 0].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 4), textcoords='offset points')

# Panel 4: Water Usage
sns.barplot(data=df, x='Season', y='Water_Used_m3', palette=palette_season, ax=axes[1, 1], ci=None)
axes[1, 1].set_title('Mean Irrigation Water Consumption (m³) by Season')
axes[1, 1].set_ylabel('Water Volume (m³)')
for p in axes[1, 1].patches:
    axes[1, 1].annotate(f"{p.get_height():.0f} m³", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 4), textcoords='offset points')

plt.tight_layout()
fig1_path = 'output_visualizations/fig1_seasonal_macro_metrics.png'
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved {fig1_path}")

# FIGURE 2: Crop Seasonal Economics
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Crop-Level Performance: Economic Disparity & Seasonal Sensitivity', fontsize=16, fontweight='bold')

crop_profit_pivot = df.pivot_table(index='Crop', columns='Season', values='Profit_INR', aggfunc='mean')[['Kharif', 'Rabi', 'Zaid']]
crop_profit_pivot.plot(kind='bar', ax=axes[0], color=['#1f77b4', '#2ca02c', '#d62728'], width=0.75)
axes[0].axhline(0, color='black', linestyle='-', linewidth=0.8)
axes[0].set_title('Mean Net Profit (INR) by Crop across Seasons')
axes[0].set_ylabel('Net Profit (INR)')
axes[0].tick_params(axis='x', rotation=30)
axes[0].legend(title='Season')

crop_yield_pivot = df.pivot_table(index='Crop', columns='Season', values='Yield_Tonnes_Ha', aggfunc='mean')[['Kharif', 'Rabi', 'Zaid']]
sns.heatmap(crop_yield_pivot, annot=True, fmt='.2f', cmap='YlGnBu', cbar_kws={'label': 'Yield (Tonnes/Ha)'}, ax=axes[1])
axes[1].set_title('Mean Crop Yield Heatmap (Tonnes/Ha)')
axes[1].set_ylabel('Crop Variety')

plt.tight_layout()
fig2_path = 'output_visualizations/fig2_crop_seasonal_economics.png'
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved {fig2_path}")

# FIGURE 3: Irrigation Resilience
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Irrigation Technology as a Climate Resilience Lever', fontsize=16, fontweight='bold')

irr_profit_pivot = df.pivot_table(index='Irrigation_Method', columns='Season', values='Profit_INR', aggfunc='mean')[['Kharif', 'Rabi', 'Zaid']]
irr_profit_pivot.plot(kind='bar', ax=axes[0], color=['#1f77b4', '#2ca02c', '#d62728'], width=0.7)
axes[0].axhline(0, color='red', linestyle='--', linewidth=1, label='Break-even (₹0)')
axes[0].set_title('Mean Profit (INR) by Irrigation Method across Seasons')
axes[0].set_ylabel('Mean Profit (INR)')
axes[0].tick_params(axis='x', rotation=0)
axes[0].legend(title='Season')

sns.barplot(data=df, x='Irrigation_Method', y='Water_Efficiency_t_per_1000m3', hue='Season', palette=palette_season, ax=axes[1], ci=None)
axes[1].set_title('Water Efficiency (Tonnes per 1000 m³) by Irrigation Method')
axes[1].set_ylabel('Efficiency (Tonnes / 1000 m³)')
axes[1].legend(title='Season')

plt.tight_layout()
fig3_path = 'output_visualizations/fig3_irrigation_resilience.png'
plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved {fig3_path}")

# FIGURE 4: ML Feature Importance & Predictions
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Machine Learning Drivers of Agricultural Performance (Random Forest)', fontsize=16, fontweight='bold')

importances_profit.sort_values().plot(kind='barh', color='#1f77b4', ax=axes[0])
axes[0].set_title(f'Top 10 Predictors of Farm Net Profit (R² = {r2_profit:.3f})')
axes[0].set_xlabel('Relative Feature Importance')

importances_yield.sort_values().plot(kind='barh', color='#2ca02c', ax=axes[1])
axes[1].set_title(f'Top 10 Predictors of Crop Yield (R² = {r2_yield:.3f})')
axes[1].set_xlabel('Relative Feature Importance')

plt.tight_layout()
fig4_path = 'output_visualizations/fig4_ml_feature_importance.png'
plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved {fig4_path}")

# FIGURE 5: Regional / State Dynamics
fig, ax = plt.subplots(figsize=(12, 6))
state_profit_pivot = df.pivot_table(index='State', columns='Season', values='Profit_INR', aggfunc='mean')[['Kharif', 'Rabi', 'Zaid']]
sns.heatmap(state_profit_pivot / 1000, annot=True, fmt='.1f', cmap='coolwarm', center=0, cbar_kws={'label': 'Mean Profit (₹ in Thousands)'}, ax=ax)
ax.set_title('State-wise Seasonal Profitability Heatmap (₹ in Thousands)', fontsize=14, fontweight='bold')
ax.set_ylabel('State')
ax.set_xlabel('Season')

plt.tight_layout()
fig5_path = 'output_visualizations/fig5_state_seasonal_dynamics.png'
plt.savefig(fig5_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved {fig5_path}")

print("All 5 publication figures successfully generated!")
