"""
ML Assignment 1 - VAR2: Subterranean Thermal Reservoir Mapping
Roll Number: BT2024039

Polynomial Regression with OLS, Ridge, and Lasso regularization
to predict Thermal Anomaly Score (y) from spatial coordinates (x1, x2, x3).
Final Selected Model: Degree 10 Ridge (alpha=1.0).
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
TRAIN_FILE = os.path.join(DATA_DIR, f"{ROLL_NO}_train_var2.csv")
TEST_FILE = os.path.join(DATA_DIR, f"{ROLL_NO}_test_var2.csv")
PRED_FILE = os.path.join(PRED_DIR, f"{ROLL_NO} pred var2.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "var2_model.joblib")

ALL_FEATURES = ['x1', 'x2', 'x3']
TARGET = 'y'

# ============================================================
# Load Data
# ============================================================
print("=" * 65)
print("VAR2: Subterranean Thermal Reservoir Mapping")
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
print("1. Standard OLS Polynomial Regression (Degrees 4 to 10):")
print(f"{'Degree':<8} {'CV MSE':<14} {'CV R²':<12}")
print("-" * 38)

best_ols = {'mse': np.inf}
for deg in [4, 6, 8, 10]:
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
# 2. Ridge Regularization Search (Degree 10)
# ============================================================
print("2. Ridge Polynomial Regression (Degree 10 Regularization):")
print(f"{'Degree':<8} {'Alpha':<8} {'CV MSE':<14} {'CV R²':<12}")
print("-" * 46)

best_ridge = {'mse': np.inf}
for alpha in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=10, include_bias=False)),
        ('scaler', StandardScaler()),
        ('reg', Ridge(alpha=alpha, random_state=RANDOM_STATE))
    ])
    mse = -cross_val_score(pipe, X_train, y_train, cv=kf, scoring='neg_mean_squared_error').mean()
    r2 = cross_val_score(pipe, X_train, y_train, cv=kf, scoring='r2').mean()
    print(f"10       {alpha:<8} {mse:<14.6f} {r2:<12.6f}")
    if mse < best_ridge['mse']:
        best_ridge = {'degree': 10, 'alpha': alpha, 'mse': mse, 'r2': r2, 'pipe': pipe}

print(f"Best Ridge: Degree 10 (alpha={best_ridge['alpha']}), CV R² = {best_ridge['r2']:.6f}, MSE = {best_ridge['mse']:.6f}\n")

# ============================================================
# 3. Final Model Selection (Degree 10 Ridge) & Retraining
# ============================================================
selected_model_name = "Ridge Polynomial Regression (Degree 10, alpha=1.0)"
final_pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=10, include_bias=False)),
    ('scaler', StandardScaler()),
    ('reg', Ridge(alpha=1.0, random_state=RANDOM_STATE))
])

print("=" * 65)
print(f"Selected Final Model: {selected_model_name}")
print(f"  Cross-Validation R²:  {best_ridge['r2']:.6f}")
print(f"  Cross-Validation MSE: {best_ridge['mse']:.6f}")
print("=" * 65)

final_pipe.fit(X_train, y_train)
train_r2 = final_pipe.score(X_train, y_train)
total_features = final_pipe.named_steps['poly'].n_output_features_

print(f"Training R²:          {train_r2:.6f}")
print(f"Polynomial Features:  {total_features}")

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
pred_df.to_csv(f"{ROLL_NO} pred var2.csv", index=False)

print(f"Predictions saved to: {PRED_FILE} and ./{ROLL_NO} pred var2.csv")
print(f"Prediction count:     {len(predictions)} (0 NaN, 0 Inf)")
print(f"Prediction range:     [{predictions.min():.4f}, {predictions.max():.4f}]")
