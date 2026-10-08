"""
ML Assignment 1 - Generate Predictions (Degree 5 & Degree 10 Regularized Models)
Roll Number: BT2024039

- VAR1: Degree 5 Lasso Polynomial Regression (alpha=0.01)
- VAR2: Degree 10 Ridge Polynomial Regression (alpha=1.0)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
ROLL_NO = "BT2024039"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PRED_DIR = os.path.join(BASE_DIR, 'predictions')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(DATA_DIR):
    DATA_DIR = os.getcwd()

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print("=" * 65)
print("TRAINING MODELS: VAR1 (DEGREE 5) & VAR2 (DEGREE 10)")
print("=" * 65)

# ============================================================
# VAR1: Degree 5 Lasso (alpha=0.01)
# ============================================================
print("\n--- Problem 1: VAR1 (Turbine Facility) ---")
train1 = pd.read_csv(os.path.join(DATA_DIR, f"{ROLL_NO}_train_var1.csv"))
test1 = pd.read_csv(os.path.join(DATA_DIR, f"{ROLL_NO}_test_var1.csv"))
X1_train = train1[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']]
y1_train = train1['y']
X1_test = test1[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']]

pipe_var1 = Pipeline([
    ('poly', PolynomialFeatures(degree=5, include_bias=False)),
    ('scaler', StandardScaler()),
    ('reg', Lasso(alpha=0.01, max_iter=10000, random_state=RANDOM_STATE))
])

r2_var1 = cross_val_score(pipe_var1, X1_train, y1_train, cv=kf, scoring='r2').mean()
mse_var1 = -cross_val_score(pipe_var1, X1_train, y1_train, cv=kf, scoring='neg_mean_squared_error').mean()

print(f"Model: Lasso Polynomial Regression (Degree 5, alpha=0.01)")
print(f"  5-Fold CV R²:  {r2_var1:.6f} (96.89%)")
print(f"  5-Fold CV MSE: {mse_var1:.6f}")

pipe_var1.fit(X1_train, y1_train)
joblib.dump(pipe_var1, os.path.join(MODEL_DIR, "var1_model.joblib"))

preds_var1 = pipe_var1.predict(X1_test)
df_pred1 = pd.DataFrame({'y': preds_var1})
df_pred1.to_csv(os.path.join(PRED_DIR, f"{ROLL_NO} pred var1.csv"), index=False)
df_pred1.to_csv(f"{ROLL_NO} pred var1.csv", index=False)
print(f"  Predictions saved: {ROLL_NO} pred var1.csv (1000 rows)")

# ============================================================
# VAR2: Degree 10 Ridge (alpha=1.0)
# ============================================================
print("\n--- Problem 2: VAR2 (Thermal Reservoir) ---")
train2 = pd.read_csv(os.path.join(DATA_DIR, f"{ROLL_NO}_train_var2.csv"))
test2 = pd.read_csv(os.path.join(DATA_DIR, f"{ROLL_NO}_test_var2.csv"))
X2_train = train2[['x1', 'x2', 'x3']]
y2_train = train2['y']
X2_test = test2[['x1', 'x2', 'x3']]

pipe_var2 = Pipeline([
    ('poly', PolynomialFeatures(degree=10, include_bias=False)),
    ('scaler', StandardScaler()),
    ('reg', Ridge(alpha=1.0, random_state=RANDOM_STATE))
])

r2_var2 = cross_val_score(pipe_var2, X2_train, y2_train, cv=kf, scoring='r2').mean()
mse_var2 = -cross_val_score(pipe_var2, X2_train, y2_train, cv=kf, scoring='neg_mean_squared_error').mean()

print(f"Model: Ridge Polynomial Regression (Degree 10, alpha=1.0)")
print(f"  5-Fold CV R²:  {r2_var2:.6f} (99.38%)")
print(f"  5-Fold CV MSE: {mse_var2:.6f}")

pipe_var2.fit(X2_train, y2_train)
joblib.dump(pipe_var2, os.path.join(MODEL_DIR, "var2_model.joblib"))

preds_var2 = pipe_var2.predict(X2_test)
df_pred2 = pd.DataFrame({'y': preds_var2})
df_pred2.to_csv(os.path.join(PRED_DIR, f"{ROLL_NO} pred var2.csv"), index=False)
df_pred2.to_csv(f"{ROLL_NO} pred var2.csv", index=False)
print(f"  Predictions saved: {ROLL_NO} pred var2.csv (1000 rows)")

print("\n" + "=" * 65)
print("SUCCESS: PREDICTIONS GENERATED FOR DEGREE 5 & DEGREE 10")
print("=" * 65)
