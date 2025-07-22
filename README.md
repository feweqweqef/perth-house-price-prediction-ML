# perth-house-price-prediction-ML
My first ML project - A Machine Learning Approach to Forecasting Real Estate Prices in Perth, Western Australia

## Project Overview

Perth's housing market is shaped by many factors—location, land size, amenities, proximity to the CBD, and more. Using a dataset of over 30,000 entries, this project builds and evaluates various regression models to predict house prices based on property features and location. The final model is deployed using Streamlit for public use.

## 📁 Dataset Description

- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/syuzai/perth-house-prices)  
- **Original Data Source**: [house.speakingsame.com](http://house.speakingsame.com/)
- **Suburbs Covered**: 322  
- **Average Entries/Suburb**: ~100  
- **Total Columns**: 19 (before cleaning)

### Target Column:
- `PRICE`: Final sale price of the property

### Key Features:
- `BEDROOMS`, `BATHROOMS`, `GARAGE`
- `LAND_AREA`, `FLOOR_AREA`, `BUILD_YEAR`
- `CBD_DIST`, `NEAREST_STN_DIST`, `NEAREST_SCH_DIST`
- `SUBURB`, `LATITUDE`, `LONGITUDE`, etc.

##  Data Preprocessing

- **Missing Values**: Handled by dropping rows or imputing where applicable
- **Outliers**: Removed using Z-score for numeric columns
- **Unit Consistency**: Converted distances (e.g., km → meters)
- **Feature Scaling**: Applied `StandardScaler` for algorithms sensitive to scale

##  Final Model Comparison

| Model                 | R² Score (Test) | R² Score (Validation) | Notes                                 |
|----------------------|-----------------|------------------------|---------------------------------------|
| **LightGBM Regressor** | **0.8348**       | 0.8281                 | Best overall performer after tuning   |
| Random Forest         | 0.8313          | 0.8250 (approx)        | Strong baseline; robust & stable      |
| CatBoost Regressor    | 0.8295          | 0.8220 (approx)        | Performed well with minimal tuning    |
| XGBoost Regressor     | 0.8232          | 0.8160 (approx)        | Slightly less accurate than CatBoost  |
| Decision Tree         | 0.6446          | 0.6780 (approx)        | Overfit the training data             |
| SVR / Linear / Ridge  | < 0.50 or poor  | -                      | Rejected due to underperformance      |

📂 **For a full breakdown of model metrics, tuning parameters, and visualizations, please refer to the [`report/`](./report/) folder.**



