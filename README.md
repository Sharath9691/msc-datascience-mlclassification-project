# Health Impact ML Classification Project

## Overview
This project applies **machine learning classification techniques** to analyze the impact of air quality on public health outcomes.  
The workflow includes **data exploration, preprocessing, feature selection, model development, and performance comparison** using multiple algorithms (LightGBM, XGBoost, Random Forest).  

The dataset contains health-related indicators (e.g., respiratory cases, cardiovascular cases, hospital admissions) alongside air quality metrics, with the target variable being **HealthImpactClass**.

---

## Project Structure
- **Data Exploration**  
  - Summary statistics, missing values, and correlation analysis  
  - Distribution plots, boxplots, and histograms  
  - Outlier detection and handling using the IQR method  

- **Data Preprocessing**  
  - Duplicate removal  
  - Outlier capping  
  - Feature scaling with `StandardScaler`  
  - Dimensionality reduction using **PCA**  
  - Feature importance analysis via PCA loadings and Random Forest  

- **Feature Selection**  
  - Combined ranking of PCA and Random Forest importance  
  - Top 12 features selected for model training  

- **Model Development**  
  - Train-test split with stratification  
  - Class imbalance handled using **SMOTE**  
  - Models implemented:
    - LightGBM  
    - XGBoost  
    - Random Forest  

- **Hyperparameter Tuning**  
  - GridSearchCV applied to LightGBM  
  - Optimized parameters selected based on weighted F1-score  

- **Model Evaluation**  
  - Accuracy, classification report, and confusion matrix  
  - Comparison of model performance across algorithms  

---

## Requirements
Install the following dependencies before running the notebook:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn lightgbm xgboost
