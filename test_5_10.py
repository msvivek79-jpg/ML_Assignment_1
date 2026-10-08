import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')

train1 = pd.read_csv('data/BT2024039_train_var1.csv')
X1 = train1[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']]
y1 = train1['y']

train2 = pd.read_csv('data/BT2024039_train_var2.csv')
X2 = train2[['x1', 'x2', 'x3']]
y2 = train2['y']

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("=" * 60)
print("VAR1 AT DEGREE 5 (Total Features: 461)")
print("=" * 60)
# OLS
pipe_ols1 = Pipeline([('poly', PolynomialFeatures(degree=5, include_bias=False)), ('scaler', StandardScaler()), ('reg', LinearRegression())])
r2_ols1 = cross_val_score(pipe_ols1, X1, y1, cv=kf, scoring='r2').mean()
mse_ols1 = -cross_val_score(pipe_ols1, X1, y1, cv=kf, scoring='neg_mean_squared_error').mean()
print(f"OLS (deg 5):               CV R2 = {r2_ols1:.6f}, MSE = {mse_ols1:.6f}")

# Ridge
for alpha in [1.0, 5.0, 10.0, 20.0, 30.0, 50.0]:
    pipe = Pipeline([('poly', PolynomialFeatures(degree=5, include_bias=False)), ('scaler', StandardScaler()), ('reg', Ridge(alpha=alpha, random_state=42))])
    r2 = cross_val_score(pipe, X1, y1, cv=kf, scoring='r2').mean()
    mse = -cross_val_score(pipe, X1, y1, cv=kf, scoring='neg_mean_squared_error').mean()
    print(f"Ridge (deg 5, alpha={alpha:<4}):  CV R2 = {r2:.6f}, MSE = {mse:.6f}")

# Lasso
for alpha in [0.001, 0.005, 0.01, 0.02, 0.05]:
    pipe = Pipeline([('poly', PolynomialFeatures(degree=5, include_bias=False)), ('scaler', StandardScaler()), ('reg', Lasso(alpha=alpha, max_iter=10000, random_state=42))])
    r2 = cross_val_score(pipe, X1, y1, cv=kf, scoring='r2').mean()
    mse = -cross_val_score(pipe, X1, y1, cv=kf, scoring='neg_mean_squared_error').mean()
    print(f"Lasso (deg 5, alpha={alpha:<5}):  CV R2 = {r2:.6f}, MSE = {mse:.6f}")

print("\n" + "=" * 60)
print("VAR2 AT DEGREE 10 (Total Features: 285)")
print("=" * 60)
# OLS
pipe_ols2 = Pipeline([('poly', PolynomialFeatures(degree=10, include_bias=False)), ('scaler', StandardScaler()), ('reg', LinearRegression())])
r2_ols2 = cross_val_score(pipe_ols2, X2, y2, cv=kf, scoring='r2').mean()
mse_ols2 = -cross_val_score(pipe_ols2, X2, y2, cv=kf, scoring='neg_mean_squared_error').mean()
print(f"OLS (deg 10):              CV R2 = {r2_ols2:.6f}, MSE = {mse_ols2:.6f}")

# Ridge
for alpha in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
    pipe = Pipeline([('poly', PolynomialFeatures(degree=10, include_bias=False)), ('scaler', StandardScaler()), ('reg', Ridge(alpha=alpha, random_state=42))])
    r2 = cross_val_score(pipe, X2, y2, cv=kf, scoring='r2').mean()
    mse = -cross_val_score(pipe, X2, y2, cv=kf, scoring='neg_mean_squared_error').mean()
    print(f"Ridge (deg 10, alpha={alpha:<5}): CV R2 = {r2:.6f}, MSE = {mse:.6f}")

# Lasso
for alpha in [0.0001, 0.0005, 0.001, 0.005]:
    pipe = Pipeline([('poly', PolynomialFeatures(degree=10, include_bias=False)), ('scaler', StandardScaler()), ('reg', Lasso(alpha=alpha, max_iter=10000, random_state=42))])
    r2 = cross_val_score(pipe, X2, y2, cv=kf, scoring='r2').mean()
    mse = -cross_val_score(pipe, X2, y2, cv=kf, scoring='neg_mean_squared_error').mean()
    print(f"Lasso (deg 10, alpha={alpha:<6}): CV R2 = {r2:.6f}, MSE = {mse:.6f}")
