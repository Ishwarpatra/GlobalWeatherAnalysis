import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler

# Setup paths
base_dir = '/home/ubuntu/GlobalWeatherAnalysis'
data_dir = os.path.join(base_dir, 'data')
viz_dir = os.path.join(base_dir, 'visualizations')
os.makedirs(viz_dir, exist_ok=True)

# Load data
file_path = os.path.join(data_dir, 'GlobalWeatherRepository.csv')
df = pd.read_csv(file_path, encoding='latin1')

# 1. Data Cleaning
# Convert last_updated to datetime
df['last_updated'] = pd.to_datetime(df['last_updated'])

# Handle missing values (if any)
missing_count = df.isnull().sum().sum()
df = df.dropna() # In this dataset, we'll drop if any, though inspection showed 0

# Normalization
scaler = MinMaxScaler()
df['temp_normalized'] = scaler.fit_transform(df[['temperature_celsius']])

# 2. Basic EDA
plt.style.use('seaborn-v0_8-whitegrid')

# Temperature Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['temperature_celsius'], kde=True, color='skyblue')
plt.title('Global Temperature Distribution', fontsize=15)
plt.xlabel('Temperature (°C)')
plt.ylabel('Frequency')
plt.savefig(os.path.join(viz_dir, '01_temp_distribution.png'), dpi=300)
plt.close()

# Precipitation Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df[df['precip_mm'] > 0]['precip_mm'], kde=True, color='salmon')
plt.title('Global Precipitation Distribution (Excluding 0)', fontsize=15)
plt.xlabel('Precipitation (mm)')
plt.ylabel('Frequency')
plt.savefig(os.path.join(viz_dir, '02_precip_distribution.png'), dpi=300)
plt.close()

# Save cleaned data
cleaned_file = os.path.join(data_dir, 'cleaned_weather_data.csv')
df.to_csv(cleaned_file, index=False)

print(f"Phase 3 Complete: Data Cleaning & Basic EDA.")
print(f"Cleaned dataset saved to {cleaned_file}")
print(f"Visualizations saved to {viz_dir}")
