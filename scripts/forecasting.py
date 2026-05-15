import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

# Setup paths
base_dir = '/home/ubuntu/GlobalWeatherAnalysis'
data_dir = os.path.join(base_dir, 'data')
viz_dir = os.path.join(base_dir, 'visualizations')

# Load data
df = pd.read_csv(os.path.join(data_dir, 'advanced_eda_data.csv'))
df['last_updated'] = pd.to_datetime(df['last_updated'])

# Prepare Time Series features
df = df.sort_values('last_updated')
df['hour'] = df['last_updated'].dt.hour
df['day'] = df['last_updated'].dt.day
df['month'] = df['last_updated'].dt.month
df['dayofweek'] = df['last_updated'].dt.dayofweek

# Lag features (previous temperature)
df['temp_lag1'] = df.groupby('location_name')['temperature_celsius'].shift(1)
df = df.dropna(subset=['temp_lag1'])

# Select features
features = ['hour', 'day', 'month', 'dayofweek', 'temp_lag1', 'humidity', 'pressure_mb', 'cloud']
X = df[features]
y = df['temperature_celsius']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=42)
}

results = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'R2': r2}
    predictions[name] = y_pred
    print(f"{name} - MSE: {mse:.4f}, R2: {r2:.4f}")

# Ensemble (Simple Average)
ensemble_pred = (predictions['Random Forest'] + predictions['Gradient Boosting']) / 2
ensemble_mse = mean_squared_error(y_test, ensemble_pred)
ensemble_r2 = r2_score(y_test, ensemble_pred)
results['Ensemble (RF+GB)'] = {'MSE': ensemble_mse, 'R2': ensemble_r2}
print(f"Ensemble - MSE: {ensemble_mse:.4f}, R2: {ensemble_r2:.4f}")

# Visualization of Results
res_df = pd.DataFrame(results).T
plt.figure(figsize=(10, 6))
res_df['R2'].plot(kind='bar', color='plum')
plt.title('Model Comparison (R2 Score)', fontsize=15)
plt.ylabel('R2 Score')
plt.ylim(0, 1)
plt.savefig(os.path.join(viz_dir, '06_model_comparison_r2.png'), dpi=300)
plt.close()

# Actual vs Predicted (Ensemble)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, ensemble_pred, alpha=0.1, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Actual vs Predicted Temperature (Ensemble)', fontsize=15)
plt.xlabel('Actual (°C)')
plt.ylabel('Predicted (°C)')
plt.savefig(os.path.join(viz_dir, '07_actual_vs_predicted.png'), dpi=300)
plt.close()

print("Phase 5 Complete: Forecasting Models.")
