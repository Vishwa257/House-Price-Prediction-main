# Ames Housing Price Prediction(End-to-End Regression Pipeline)

This project implements a complete regression pipeline to predict house prices in Ames, Iowa, using the Ames Housing dataset from Kaggle:  
https://www.kaggle.com/datasets/prevek18/ames-housing-dataset/data

The process includes data cleaning, feature engineering, multicollinearity handling, categorical encoding, feature scaling, and model building with Lasso hyperparameter tuning.

---

## Objective

Build a predictive model that estimates the sale price of homes in Ames, Iowa based on features like overall quality, area, year built, garage condition, and more.

---

## Dataset Summary

- Source: Ames Housing Dataset (Kaggle)
- Shape: 2,930 rows × 80 columns
- Target Variable: SalePrice
- Mix of numerical and categorical features

---

## Workflow Overview

### 1. Data Cleaning
- Dropped high-null columns: Pool QC, Misc Feature, Fence, etc.
- Used domain-aware imputation:
  - 'none' for features like Mas Vnr Type, Fireplace Qu (absence of feature)
  - 'Ng' (no garage) for garage-related columns
  - Median for numerical columns like Lot Frontage
- Dropped rows with leftover missing values

### 2. Feature Selection
- Removed low-correlation features (Pearson corr < 0.1 with SalePrice)
- Removed multicollinear features using Variance Inflation Factor (VIF)

### 3. Categorical Consolidation
- Grouped rare categorical values into 'Other'
- Dropped near-constant features like Utilities

### 4. Target Transformation
- Applied log1p transformation to SalePrice to reduce right skew

### 5. Encoding and Scaling
- One-hot encoding on categorical features
- StandardScaler applied to numerical features after train-test split

### 6. Model Training and Evaluation
- Trained and compared multiple models including linear and ensemble-based methods

---

## Model Performance

| Model                    | RMSE          | R² Score             |
|--------------------------|---------------|-----------------------|
| Linear Regression        | 28649.88      | 0.8830                |
| Ridge Regression         | 28164.16      | 0.8659                |
| Lasso Regression         | 85460.91      | -2.1556e+30 (failed)  |
| LassoCV (alpha tuning)   | 27264.20      | 0.8941                |
| Random Forest Regressor  | 32725.91      | 0.7866                |

---

## Tools and Libraries

- pandas, numpy, matplotlib, seaborn
- scikit-learn: LinearRegression, Ridge, Lasso, LassoCV, RandomForestRegressor, GridSearchCV, train_test_split, StandardScaler
- statsmodels for multicollinearity (VIF)

---

## Key Learnings

- Importance of using domain knowledge for missing value handling
- How log transformation improves model performance on skewed data
- LassoCV allows regularization + feature selection
- Lasso requires careful alpha tuning to avoid underfitting

---

## Next Steps

- Deploy this model via Streamlit for interactive use
- Use SHAP or LIME for feature explainability
- Test with other ensemble models like XGBoost or LightGBM

---
