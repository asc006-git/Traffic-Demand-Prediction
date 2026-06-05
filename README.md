# Traffic Demand Prediction Using Geospatial, Temporal, and Environmental Features

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?style=for-the-badge)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)

</div>

<div align="center">

### Geospatial Analytics • Traffic Forecasting • Machine Learning • Feature Engineering

</div>

---

## Overview

This project presents an end-to-end machine learning solution for forecasting traffic demand using geospatial, temporal, environmental, and road infrastructure features.

The objective is to predict traffic intensity at a given location and timestamp by learning hidden patterns from historical transportation data. The solution combines advanced feature engineering with gradient boosting techniques to capture nonlinear relationships influencing traffic flow and demand.

---

## Project Summary

<div align="center">

| Metric | Value |
|----------|----------|
| Problem Type | Regression |
| Model | CatBoost Regressor |
| Validation Strategy | 3-Fold K-Fold Cross Validation |
| Mean R² Score | **0.9395** |
| Training Records | 77,299 |
| Test Records | 41,778 |
| Total Engineered Features | 19 |
| Target Variable | demand |

</div>

---

## Dataset Overview

<div align="center">

| Dataset | Records | Features |
|----------|----------|----------|
| Training Set | 77,299 | 10 |
| Test Set | 41,778 | 10 |

</div>

### Available Features

<div align="center">

| Feature | Description |
|----------|-------------|
| geohash | Encoded geographical location |
| day | Day identifier |
| timestamp | Time of observation |
| RoadType | Road category |
| NumberofLanes | Number of lanes |
| LargeVehicles | Vehicle restrictions |
| Landmarks | Nearby landmark presence |
| Temperature | Environmental temperature |
| Weather | Weather condition |
| demand | Traffic demand score (Target) |

</div>

---

## Feature Engineering

<div align="center">

| Category | Features Generated |
|-----------|-------------------|
| Temporal | hour, minute, hour_sin, hour_cos, time_period |
| Spatial | geohash, gh3, gh4 |
| Infrastructure | RoadType, NumberofLanes, LargeVehicles, Landmarks |
| Environmental | Temperature, Weather |
| Interaction Features | veh_lanes, road_lanes, land_veh, weather_temp |

</div>

---

## Model Configuration

<div align="center">

| Hyperparameter | Value |
|----------------|--------|
| Algorithm | CatBoostRegressor |
| Iterations | 1500 |
| Learning Rate | 0.07 |
| Depth | 8 |
| L2 Regularization | 5 |
| Subsample | 0.8 |
| Early Stopping | 80 |
| Loss Function | RMSE |

</div>

---

## Validation Results

### Fold-wise Performance

<div align="center">

| Fold | R² Score |
|--------|----------|
| Fold 1 | 0.9409 |
| Fold 2 | 0.9387 |
| Fold 3 | 0.9389 |

</div>

### Overall Performance

<div align="center">

| Metric | Value |
|---------|---------|
| Mean R² Score | **0.9395** |
| Standard Deviation | **0.0010** |

</div>

---

## Top Contributing Features

<div align="center">

| Rank | Feature |
|---------|---------|
| 1 | RoadType |
| 2 | geohash |
| 3 | gh4 |
| 4 | NumberofLanes |
| 5 | road_lanes |
| 6 | time_period |
| 7 | veh_lanes |
| 8 | RoadType_enc |
| 9 | hour_sin |
| 10 | hour_cos |

</div>

---

<details>
<summary><b>Exploratory Data Analysis</b></summary>

### Key Findings

- More than 1200 unique geohash locations indicating high geographical diversity.
- Missing values primarily observed in RoadType, Temperature, and Weather.
- Demand distribution exhibited strong positive skewness.
- Significant variation in traffic demand across locations and road types.
- Road infrastructure and spatial features contributed most to predictive performance.

</details>

---

<details>
<summary><b>Data Preprocessing Pipeline</b></summary>

### Missing Value Treatment

**Numerical Features**
- Median Imputation

**Categorical Features**
- Mode Imputation

### Timestamp Processing

Timestamp values were decomposed into:

- Hour
- Minute

### Cyclical Encoding

```python
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
```

This preserves the cyclical nature of time.

### Geospatial Engineering

Hierarchical location features were extracted using:

```text
geohash
├── gh3
└── gh4
```

These features capture regional traffic patterns and location-based demand behavior.

### Target Transformation

```python
y = np.log1p(demand)
```

Logarithmic transformation was applied to reduce skewness and improve model learning stability.

</details>

---

<details>
<summary><b>Model Selection Rationale</b></summary>

Several machine learning algorithms were considered:

- Linear Regression
- Random Forest Regressor
- XGBoost
- LightGBM
- CatBoost

CatBoost was selected because it provides:

- Native support for categorical variables
- Strong performance on structured tabular data
- Effective handling of high-cardinality features
- Minimal preprocessing requirements
- Robust generalization performance

</details>

---

## Project Structure

```text
traffic-demand-prediction/
│
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── eda.py
├── train.py
├── full_pipeline.py
├── submission.csv
├── README.md
├── test.py
│
├── catboost_info/
│   ├── learn_error.tsv
│   ├── test_error.tsv
│   ├── time_left.tsv
│   └── catboost_training.json
│
└── outputs/
```

---

## Future Enhancements

- Hyperparameter optimization using Optuna
- Ensemble learning with LightGBM and XGBoost
- Advanced geospatial target encoding
- Automated feature selection techniques
- Bayesian optimization
- Temporal sequence modeling
- Spatial clustering methods
- Real-time deployment using REST APIs

---

## Technologies Used

<div align="center">

| Category | Technologies |
|-----------|-------------|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-Learn, CatBoost |
| Version Control | Git, GitHub |
| Development Environment | Jupyter Notebook, VS Code |

</div>

---

## Author

**Adhyatma Singh Chauhan**

Machine Learning • Data Science • Software Development

This project demonstrates the application of feature engineering, geospatial analytics, and gradient boosting techniques for large-scale traffic demand forecasting and predictive modeling.