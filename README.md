# WIDS Project: Temperature Prediction

## Overview
This project focuses on predicting 14-day temperature values using machine learning. It involves a full workflow from data preprocessing to model training, hyperparameter tuning, evaluation, and deployment via a command-line interface (CLI).

The dataset was obtained from **Kaggle (WiDS Datathon)**, and only the **training file** was used for this project. No external or hidden test files were included.

The goal is to build accurate regression models using:

- RandomForestRegressor (baseline)
- XGBoost Regressor
- CatBoost Regressor
- LightGBM Regressor

Hyperparameter tuning is done using **Optuna**.

---

## Libraries Used
- **Data manipulation & numerical computing:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `plotly`
- **Machine learning:** `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `feature-engine`
- **Hyperparameter tuning:** `optuna`
- **Model export & CLI inference:** `joblib`, `sys`

---

## Deliverables
The project includes the following completed pipelines:

- **Data Preprocessing Pipeline** (pandas, sklearn)
- **Feature Transformation Pipeline** (sklearn, feature-engine)
- **Model Training Pipeline** (sklearn, XGBoost, LightGBM, CatBoost)
- **Model Evaluation Pipeline** (sklearn)
- **Hyperparameter Search Pipeline** (Optuna)
- **Export of the final trained models** and code for inference via CLI
