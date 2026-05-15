import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Setup paths
base_dir = '/home/ubuntu/GlobalWeatherAnalysis'
data_dir = os.path.join(base_dir, 'data')
viz_dir = os.path.join(base_dir, 'visualizations')

# Load data
df = pd.read_csv(os.path.join(data_dir, 'advanced_eda_data.csv'))

# 1. Spatial Analysis: Temperature by Latitude
plt.figure(figsize=(10, 6))
sns.regplot(data=df.sample(5000), x='latitude', y='temperature_celsius', scatter_kws={'alpha':0.1}, line_kws={'color':'red'})
plt.title('Temperature vs Latitude (Spatial Pattern)', fontsize=15)
plt.xlabel('Latitude')
plt.ylabel('Temperature (°C)')
plt.savefig(os.path.join(viz_dir, '08_temp_vs_latitude.png'), dpi=300)
plt.close()

# 2. Climate Analysis: Average Temperature by Timezone (Region Proxy)
tz_avg_temp = df.groupby('timezone')['temperature_celsius'].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 15))
tz_avg_temp.head(30).plot(kind='barh', color='orange')
plt.title('Top 30 Warmest Regions (by Timezone)', fontsize=15)
plt.xlabel('Average Temperature (°C)')
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, '09_regional_climate.png'), dpi=300)
plt.close()

# 3. Environmental Impact: Air Quality (PM2.5) vs Cloud Cover
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x=pd.cut(df['cloud'], bins=5), y='air_quality_PM2.5')
plt.title('Air Quality (PM2.5) across Cloud Cover Levels', fontsize=15)
plt.xlabel('Cloud Cover Range (%)')
plt.ylabel('PM2.5 Concentration')
plt.savefig(os.path.join(viz_dir, '10_aq_vs_cloud.png'), dpi=300)
plt.close()

# 4. Geographical Patterns: Average Humidity by Country (Top 20)
country_humidity = df.groupby('country')['humidity'].mean().sort_values(ascending=False).head(20)
plt.figure(figsize=(12, 6))
country_humidity.plot(kind='bar', color='blue')
plt.title('Top 20 Most Humid Countries (Average)', fontsize=15)
plt.ylabel('Avg Humidity (%)')
plt.savefig(os.path.join(viz_dir, '11_country_humidity.png'), dpi=300)
plt.close()

print("Phase 6 Complete: Unique Analyses.")
