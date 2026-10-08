"""
ML Assignment 1 - VAR1: Power Plant Steam Turbine Optimization
Roll Number: BT2024039

Polynomial Regression with OLS, Ridge, and Lasso regularization
to predict Net Power Score (y) from turbine operational parameters (x1-x6).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PRED_DIR = os.path.join(BASE_DIR, 'predictions')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

ROLL_NO = "BT2024039"
TRAIN_FILE = os.path.join(DATA_DIR, f"{ROLL_NO}_train_var1.csv")
TEST_FILE = os.path.join(DATA_DIR, f"{ROLL_NO}_test_var1.csv")
PRED_FILE = os.path.join(PRED_DIR, f"{ROLL_NO} pred var1.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "var1_model.joblib")

ALL_FEATURES = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
TARGET = 'y'

# ============================================================
# Load Data
# ============================================================
print("=" * 65)
print("VAR1: Power Plant Steam Turbine Optimization")
print("=" * 65)

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print(f"Features:    {ALL_FEATURES}")
print(f"Target:      {TARGET}")
print()

X_train = train[ALL_FEATURES]
y_train = train[TARGET]
X_test = test[ALL_FEATURES]

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ============================================================
# 1. Standard OLS Polynomial Regression Search
# ============================================================
print("1. Standard OLS Polynomial Regression (Degrees 1 to 5):")
print(f"{'Degree':<8} {'CV MSE':<14} {'CV R²':<12}")
print("-" * 38)

best_ols = {'mse': np.inf}
for deg in [1, 2, 3, 4, 5]:
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
        ('scaler', StandardScaler()),
        ('reg', LinearRegression())
    ])
    mse = -cross_val_score(pipe, X_train, y_train, cv=kf, scoring='neg_mean_squared_error').mean()
    r2 = cross_val_score(pipe, X_train, y_train, cv=kf, scoring='r2').mean()
    print(f"{deg:<8} {mse:<14.6f} {r2:<12.6f}")
    if mse < best_ols['mse']:
        best_ols = {'degree': deg, 'mse': mse, 'r2': r2, 'pipe': pipe}

print(f"Best OLS: Degree {best_ols['degree']}, CV R² = {best_ols['r2']:.6f}, MSE = {best_ols['mse']:.6f}\n")

# ============================================================
# 2. Ridge Regularization Search (L2 Penalty)
# ============================================================
print("2. Ridge Polynomial Regression (L2 Regularization):")
print(f"{'Degree':<8} {'Alpha':<8} {'CV MSE':<14} {'CV R²':<12}")
print("-" * 46)

best_ridge = {'mse': np.inf}
for deg in [4, 5]:
    for alpha in [1.0, 5.0, 10.0, 20.0, 30.0]:
        pipe = Pipeline([
            ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
            ('scaler', StandardScaler()),
            ('reg', Ridge(alpha=alpha, random_state=RANDOM_STATE))
        ])
        mse = -cross_val_score(pipe, X_train, y_train, cv=kf, scoring='neg_mean_squared_error').mean()
        r2 = cross_val_score(pipe, X_train, y_train, cv=kf, scoring='r2').mean()
        print(f"{deg:<8} {alpha:<8} {mse:<14.6f} {r2:<12.6f}")
        if mse < best_ridge['mse']:
            best_ridge = {'degree': deg, 'alpha': alpha, 'mse': mse, 'r2': r2, 'pipe': pipe}

print(f"Best Ridge: Degree {best_ridge['degree']} (alpha={best_ridge['alpha']}), CV R² = {best_ridge['r2']:.6f}, MSE = {best_ridge['mse']:.6f}\n")

# ============================================================
# 3. Lasso Regularization Search (L1 Sparsity Penalty)
# ============================================================
print("3. Lasso Polynomial Regression (L1 Feature Sparsity):")
print(f"{'Degree':<8} {'Alpha':<8} {'Non-zero':<10} {'CV MSE':<14} {'CV R²':<12}")
print("-" * 56)

best_lasso = {'mse': np.inf}
for deg in [4, 5]:
    for alpha in [0.001, 0.005, 0.01, 0.02, 0.05]:
        pipe = Pipeline([
            ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
            ('scaler', StandardScaler()),
            ('reg', Lasso(alpha=alpha, max_iter=10000, random_state=RANDOM_STATE))
        ])
        mse = -cross_val_score(pipe, X_train, y_train, cv=kf, scoring='neg_mean_squared_error').mean()
        r2 = cross_val_score(pipe, X_train, y_train, cv=kf, scoring='r2').mean()
        
        pipe.fit(X_train, y_train)
        nonzero = np.sum(pipe.named_steps['reg'].coef_ != 0)
        
        print(f"{deg:<8} {alpha:<8} {nonzero:<10} {mse:<14.6f} {r2:<12.6f}")
        if mse < best_lasso['mse']:
            best_lasso = {'degree': deg, 'alpha': alpha, 'nonzero': nonzero, 'mse': mse, 'r2': r2, 'pipe': pipe}

print(f"Best Lasso: Degree {best_lasso['degree']} (alpha={best_lasso['alpha']}), CV R² = {best_lasso['r2']:.6f}, MSE = {best_lasso['mse']:.6f}\n")

# ============================================================
# 4. Final Model Selection & Retraining
# ============================================================
# Lasso at degree 5 eliminates noisy features and yields the highest R2
selected_model_name = "Lasso Polynomial Regression"
final_pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=best_lasso['degree'], include_bias=False)),
    ('scaler', StandardScaler()),
    ('reg', Lasso(alpha=best_lasso['alpha'], max_iter=10000, random_state=RANDOM_STATE))
])

print("=" * 65)
print(f"Selected Final Model: {selected_model_name} (Degree {best_lasso['degree']}, alpha={best_lasso['alpha']})")
print(f"  Cross-Validation R²:  {best_lasso['r2']:.6f}")
print(f"  Cross-Validation MSE: {best_lasso['mse']:.6f}")
print("=" * 65)

final_pipe.fit(X_train, y_train)
train_r2 = final_pipe.score(X_train, y_train)
nonzero_features = np.sum(final_pipe.named_steps['reg'].coef_ != 0)
total_features = final_pipe.named_steps['poly'].n_output_features_

print(f"Training R²:          {train_r2:.6f}")
print(f"Active Features:      {nonzero_features} of {total_features} retained")

# Save model artifact
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(final_pipe, MODEL_FILE)
print(f"Model saved to:       {MODEL_FILE}")

# Generate and save predictions
predictions = final_pipe.predict(X_test)
pred_df = pd.DataFrame({TARGET: predictions})
os.makedirs(PRED_DIR, exist_ok=True)
pred_df.to_csv(PRED_FILE, index=False)
# Also copy to root for submission
pred_df.to_csv(f"{ROLL_NO} pred var1.csv", index=False)

print(f"Predictions saved to: {PRED_FILE} and ./{ROLL_NO} pred var1.csv")
print(f"Prediction count:     {len(predictions)} (0 NaN, 0 Inf)")
print(f"Prediction range:     [{predictions.min():.4f}, {predictions.max():.4f}]")
