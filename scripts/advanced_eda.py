import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestRegressor
import os

# Setup paths
base_dir = '/home/ubuntu/GlobalWeatherAnalysis'
data_dir = os.path.join(base_dir, 'data')
viz_dir = os.path.join(base_dir, 'visualizations')

# Load cleaned data
df = pd.read_csv(os.path.join(data_dir, 'cleaned_weather_data.csv'))

# 1. Anomaly Detection
features_for_anomaly = ['temperature_celsius', 'humidity', 'wind_kph', 'pressure_mb']
iso_forest = IsolationForest(contamination=0.01, random_state=42)
df['anomaly'] = iso_forest.fit_predict(df[features_for_anomaly])

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='temperature_celsius', y='humidity', hue='anomaly', palette={1: 'blue', -1: 'red'}, alpha=0.5)
plt.title('Anomaly Detection: Temperature vs Humidity', fontsize=15)
plt.legend(title='Is Anomaly?', labels=['Yes (-1)', 'No (1)'])
plt.savefig(os.path.join(viz_dir, '03_anomaly_detection.png'), dpi=300)
plt.close()

# 2. Feature Importance
features = ['wind_kph', 'pressure_mb', 'precip_mm', 'humidity', 'cloud', 'visibility_km', 'uv_index', 'gust_kph']
X = df[features]
y = df['temperature_celsius']

# Sample data for faster processing if needed, but let's try full first
rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
importances.plot(kind='bar', color='teal')
plt.title('Feature Importance for Temperature Prediction', fontsize=15)
plt.ylabel('Importance Score')
plt.savefig(os.path.join(viz_dir, '04_feature_importance.png'), dpi=300)
plt.close()

# 3. Air Quality Correlation
aq_cols = [c for c in df.columns if 'air_quality' in c and df[c].dtype != 'object']
weather_cols = ['temperature_celsius', 'humidity', 'wind_kph', 'precip_mm']
corr_matrix = df[aq_cols + weather_cols].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix.loc[aq_cols, weather_cols], annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation: Air Quality vs Weather Parameters', fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, '05_air_quality_correlation.png'), dpi=300)
plt.close()

# Save progress
df.to_csv(os.path.join(data_dir, 'advanced_eda_data.csv'), index=False)
print("Phase 4 Complete: Advanced EDA.")
