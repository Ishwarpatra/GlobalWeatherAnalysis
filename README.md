# Global Weather Analysis Project

## Project Overview

This project undertakes an advanced data science assessment of the "Global Weather Repository" dataset, available on Kaggle. The primary objective is to forecast future weather trends and demonstrate a comprehensive suite of data science skills, ranging from basic data cleaning to advanced analytical techniques and machine learning model development. The dataset provides daily weather information for cities around the world, featuring over 40 distinct weather-related parameters.

## PM Accelerator Mission

In alignment with the assignment requirements, this project supports the **PM Accelerator mission: "to break down financial barriers and achieve educational fairness."**

## Assessment Details: Advanced Track

This project focuses on the advanced assessment track, encompassing the following key areas:

### 1. Data Cleaning & Preprocessing

-   **Handling Missing Values:** Initial inspection revealed no significant missing values, but the pipeline is robust to handle them if they arise.
-   **Outlier Management:** Outliers were identified and analyzed during the Advanced EDA phase using Isolation Forest.
-   **Data Normalization:** Key numerical features, such as `temperature_celsius`, were normalized using `MinMaxScaler` to prepare them for modeling.

### 2. Exploratory Data Analysis (EDA)

-   **Basic EDA:** Initial trends, correlations, and patterns were uncovered through descriptive statistics and visualizations.
-   **Visualizations:** Histograms for temperature and precipitation distributions were generated to understand their spread.

### 3. Advanced EDA

-   **Anomaly Detection:** Implemented using `IsolationForest` to identify unusual weather patterns or data points based on multiple features.
-   **Feature Importance:** Utilized `RandomForestRegressor` to assess the importance of various weather parameters in predicting temperature, providing insights into key drivers of temperature variations.
-   **Correlation Analysis:** Explored the correlation between air quality parameters and other weather variables to understand environmental impacts.

### 4. Forecasting with Multiple Models

-   **Model Building:** Developed and evaluated multiple forecasting models, including `Linear Regression`, `Random Forest Regressor`, and `Gradient Boosting Regressor`, to predict `temperature_celsius`.
-   **Ensemble Modeling:** Created an ensemble model by averaging predictions from the best-performing individual models to enhance forecast accuracy.
-   **Model Evaluation:** Performance was assessed using metrics such as Mean Squared Error (MSE) and R-squared (R2).

### 5. Unique Analyses

-   **Climate Analysis:** Studied long-term climate patterns by analyzing average temperatures across different timezones (as a proxy for regions).
-   **Environmental Impact:** Investigated the correlation between air quality (specifically PM2.5) and cloud cover to understand atmospheric influences on pollution.
-   **Spatial Analysis:** Explored geographical patterns in temperature by analyzing its relationship with latitude.
-   **Geographical Patterns:** Examined how humidity levels differ across various countries.

## Project Structure

```
GlobalWeatherAnalysis/
├── data/
│   ├── GlobalWeatherRepository.csv
│   ├── cleaned_weather_data.csv
│   └── advanced_eda_data.csv
├── visualizations/
│   ├── 01_temp_distribution.png
│   ├── 02_precip_distribution.png
│   ├── 03_anomaly_detection.png
│   ├── 04_feature_importance.png
│   ├── 05_air_quality_correlation.png
│   ├── 06_model_comparison_r2.png
│   ├── 07_actual_vs_predicted.png
│   ├── 08_temp_vs_latitude.png
│   ├── 09_regional_climate.png
│   ├── 10_aq_vs_cloud.png
│   └── 11_country_humidity.png
├── reports/
│   └── Global_Weather_Analysis_Report.md
├── scripts/
│   ├── eda_cleaning.py
│   ├── advanced_eda.py
│   ├── forecasting.py
│   └── unique_analyses.py
└── README.md
```

## Methodology

1.  **Data Acquisition:** The `GlobalWeatherRepository.csv` dataset was downloaded from Kaggle.
2.  **Environment Setup:** Python with `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn` libraries was used for all analyses.
3.  **Data Preprocessing:** The dataset underwent cleaning, including datetime conversion and handling of potential missing values. Normalization was applied to relevant features.
4.  **Exploratory Data Analysis:** Both basic and advanced EDA techniques were employed to understand data characteristics, identify anomalies, and discover relationships between variables.
5.  **Model Development:** Linear Regression, Random Forest, and Gradient Boosting models were trained for temperature forecasting. An ensemble approach was used to combine predictions.
6.  **Unique Insights:** Specialized analyses were conducted to explore climate patterns, environmental correlations, and geographical influences on weather.
7.  **Reporting:** All findings, methodologies, and visualizations are documented in `Global_Weather_Analysis_Report.md`.

## Results and Insights

The analysis revealed significant insights into global weather patterns, including:

-   Clear distributions for temperature and precipitation, highlighting global climatic variations.
-   Identified anomalies in weather data, which could indicate extreme events or data collection issues.
-   Key features influencing temperature predictions were identified, with humidity and cloud cover often playing significant roles.
-   Correlations between air quality metrics and weather parameters provided insights into environmental factors.
-   Forecasting models demonstrated good performance, with the ensemble model achieving the highest R2 score, indicating its effectiveness in predicting temperature trends.
-   Spatial analysis showed a clear relationship between latitude and temperature, consistent with global climate zones.
-   Regional climate analysis identified consistently warmer areas based on timezone data.
-   Environmental impact analysis suggested connections between cloud cover and PM2.5 concentrations.
-   Geographical patterns in humidity were observed, showcasing variations across different countries.

## How to Reproduce

To reproduce this analysis, clone the repository and run the Python scripts in the `scripts/` directory in the following order:

1.  `eda_cleaning.py`
2.  `advanced_eda.py`
3.  `forecasting.py`
4.  `unique_analyses.py`

Ensure all required Python libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`) are installed (`pip install -r requirements.txt` if a `requirements.txt` is provided, otherwise install individually). The raw dataset `GlobalWeatherRepository.csv` should be placed in the `data/` directory.

## License

This project is open-source and available under the MIT License. See the `LICENSE` file for more details.
