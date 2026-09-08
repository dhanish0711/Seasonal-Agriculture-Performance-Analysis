import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

os.makedirs('models', exist_ok=True)

df = pd.read_csv('data_cleaned/cleaned_seasonal_agriculture_performance.csv')

# Numerical features
num_cols = [
    'Farm_Area_Hectares', 'Rainfall_mm', 'Avg_Temperature_C', 'Humidity_pct',
    'Sunlight_Hours_Day', 'Soil_pH', 'Soil_Moisture_pct', 'Nitrogen_kg_ha',
    'Phosphorus_kg_ha', 'Potassium_kg_ha', 'Fertilizer_kg_ha', 'Pesticide_Litre_ha',
    'Seed_Quality_Score', 'Water_Used_m3', 'Disease_Pest_Risk_pct'
]
cat_cols = ['Crop', 'Season', 'Irrigation_Method', 'State']

# 1. Prepare ML training matrix
X = pd.get_dummies(df[num_cols + cat_cols], drop_first=True)
y_yield = df['Yield_Tonnes_Ha']
y_profit = df['Profit_INR']

feature_names = X.columns.tolist()

# Fit Random Forest Regressors
rf_yield = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_yield.fit(X, y_yield)

rf_profit = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_profit.fit(X, y_profit)

# 2. Unsupervised Clustering: K-Means on Economic & Performance Profiles
cluster_features = ['Yield_Tonnes_Ha', 'Total_Cost_INR', 'Revenue_INR', 'Profit_INR', 'Water_Used_m3', 'Soil_pH']
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(df[cluster_features])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_cluster_scaled)

# Map clusters to meaningful business names based on profit
cluster_means = df.groupby('Cluster')['Profit_INR'].mean()
cluster_ranking = cluster_means.sort_values(ascending=False).index.tolist()
cluster_map = {
    cluster_ranking[0]: 'Cluster A: High-Margin Commercial Leaders',
    cluster_ranking[1]: 'Cluster B: Moderate-Margin Resilient Farms',
    cluster_ranking[2]: 'Cluster C: Climate & Cost-Vulnerable Farms'
}
df['Cluster_Name'] = df['Cluster'].map(cluster_map)

pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(X_cluster_scaled)
df['PCA1'] = pca_coords[:, 0]
df['PCA2'] = pca_coords[:, 1]

# Save updated dataset with clusters
df.to_csv('data_cleaned/cleaned_seasonal_agriculture_performance.csv', index=False)

# Save models bundle
bundle = {
    'rf_yield': rf_yield,
    'rf_profit': rf_profit,
    'kmeans': kmeans,
    'scaler': scaler,
    'pca': pca,
    'cluster_map': cluster_map,
    'feature_names': feature_names,
    'num_cols': num_cols,
    'cat_cols': cat_cols
}
joblib.dump(bundle, 'models/model_bundle.joblib')
print("Successfully trained and saved model bundle to models/model_bundle.joblib")
print(f"Cluster profiles:\n{df.groupby('Cluster_Name')[['Profit_INR', 'Yield_Tonnes_Ha', 'Water_Used_m3']].mean()}")
