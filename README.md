# Predicting Machine Learning Model Performance from Dataset Characteristics

A framework to predict the performance of various machine learning models—linear, non-linear, and tree-based—based on key characteristics of classification datasets.

---

## Overview

Selecting the best machine learning model for a classification task can be challenging and resource-intensive. This framework streamlines the process by analyzing dataset characteristics and predicting model performance metrics of accuracy, recall, and F1 score. This project provides a practical tool to identify optimal models without exhaustive trial-and-error.

---

## Features

- **Dataset Metadata Analysis**: Utilizes sample size, feature count, class distribution, and more.
- **Performance Metrics Prediction**: Predictions for linear, non-linear, and tree-based models.
- **Synthetic Data Integration**: Generates synthetic samples to expand smaller datasets.
- **Feature Engineering**: Includes PCA, dimensionality estimates, and class balancing techniques (*SMOTE*, *ADASYN*).
- **Stacked Regression Models**: Combines `RandomForestRegressor`, `GradientBoostingRegressor`, and `SVR` for robust performance prediction.

---

## Technologies Used

- **Python Libraries**:
  - Data Processing: `pandas`, `numpy`, `scikit-learn`
  - Visualization: `matplotlib`, `seaborn`
  - Optimization: `optuna`
- **Machine Learning Models**: `CatBoostClassifier`, `ExtraTreesClassifier`, `SGDClassifier`, `RidgeClassifier`, `GaussianNB`, `LinearSVC`
- **Synthetic Data Techniques**: *Gaussian Mixture Models (GMM)*, *SMOTE*, *ADASYN*

---
