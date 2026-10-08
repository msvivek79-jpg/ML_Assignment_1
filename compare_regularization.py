import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
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
print("VAR1 COMPARISON: OLS vs Ridge vs Lasso")
print("=" * 60)
# VAR1 OLS Baseline
pipe_ols1 = Pipeline([('poly', PolynomialFeatures(degree=4, include_bias=False)), ('scaler', StandardScaler()), ('reg', LinearRegression())])
r2_ols1 = cross_val_score(pipe_ols1, X1, y1, cv=kf, scoring='r2').mean()
mse_ols1 = -cross_val_score(pipe_ols1, X1, y1, cv=kf, scoring='neg_mean_squared_error').mean()
print(f"VAR1 OLS (deg 4):               CV R2 = {r2_ols1:.6f}, MSE = {mse_ols1:.6f}")

# VAR1 Ridge
best_ridge1 = None
best_r2_r1 = -1
for deg in [4, 5]:
    for alpha in [0.1, 1.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0]:
        pipe = Pipeline([('poly', PolynomialFeatures(degree=deg, include_bias=False)), ('scaler', StandardScaler()), ('reg', Ridge(alpha=alpha, random_state=42))])
        r2 = cross_val_score(pipe, X1, y1, cv=kf, scoring='r2').mean()
        mse = -cross_val_score(pipe, X1, y1, cv=kf, scoring='neg_mean_squared_error').mean()
        if r2 > best_r2_r1:
            best_r2_r1 = r2
            best_ridge1 = (deg, alpha, r2, mse)
print(f"VAR1 Ridge (deg {best_ridge1[0]}, alpha={best_ridge1[1]}):   CV R2 = {best_ridge1[2]:.6f}, MSE = {best_ridge1[3]:.6f}")

# VAR1 Lasso
best_lasso1 = None
best_r2_l1 = -1
for deg in [4, 5]:
    for alpha in [0.0005, 0.001, 0.005, 0.01, 0.02, 0.05]:
        pipe = Pipeline([('poly', PolynomialFeatures(degree=deg, include_bias=False)), ('scaler', StandardScaler()), ('reg', Lasso(alpha=alpha, max_iter=10000, random_state=42))])
        r2 = cross_val_score(pipe, X1, y1, cv=kf, scoring='r2').mean()
        mse = -cross_val_score(pipe, X1, y1, cv=kf, scoring='neg_mean_squared_error').mean()
        if r2 > best_r2_l1:
            best_r2_l1 = r2
            best_lasso1 = (deg, alpha, r2, mse)
print(f"VAR1 Lasso (deg {best_lasso1[0]}, alpha={best_lasso1[1]}):   CV R2 = {best_lasso1[2]:.6f}, MSE = {best_lasso1[3]:.6f}")

print("\n" + "=" * 60)
print("VAR2 COMPARISON: OLS vs Ridge vs Lasso")
print("=" * 60)
# VAR2 OLS Baseline
pipe_ols2 = Pipeline([('poly', PolynomialFeatures(degree=8, include_bias=False)), ('scaler', StandardScaler()), ('reg', LinearRegression())])
r2_ols2 = cross_val_score(pipe_ols2, X2, y2, cv=kf, scoring='r2').mean()
mse_ols2 = -cross_val_score(pipe_ols2, X2, y2, cv=kf, scoring='neg_mean_squared_error').mean()
print(f"VAR2 OLS (deg 8):               CV R2 = {r2_ols2:.6f}, MSE = {mse_ols2:.6f}")

# VAR2 Ridge
best_ridge2 = None
best_r2_r2 = -1
for deg in [7, 8, 9]:
    for alpha in [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0]:
        pipe = Pipeline([('poly', PolynomialFeatures(degree=deg, include_bias=False)), ('scaler', StandardScaler()), ('reg', Ridge(alpha=alpha, random_state=42))])
        r2 = cross_val_score(pipe, X2, y2, cv=kf, scoring='r2').mean()
        mse = -cross_val_score(pipe, X2, y2, cv=kf, scoring='neg_mean_squared_error').mean()
        if r2 > best_r2_r2:
            best_r2_r2 = r2
            best_ridge2 = (deg, alpha, r2, mse)
print(f"VAR2 Ridge (deg {best_ridge2[0]}, alpha={best_ridge2[1]}):  CV R2 = {best_ridge2[2]:.6f}, MSE = {best_ridge2[3]:.6f}")

# VAR2 Lasso
best_lasso2 = None
best_r2_l2 = -1
for deg in [7, 8, 9]:
    for alpha in [0.0001, 0.0005, 0.001, 0.005, 0.01]:
        pipe = Pipeline([('poly', PolynomialFeatures(degree=deg, include_bias=False)), ('scaler', StandardScaler()), ('reg', Lasso(alpha=alpha, max_iter=10000, random_state=42))])
        r2 = cross_val_score(pipe, X2, y2, cv=kf, scoring='r2').mean()
        mse = -cross_val_score(pipe, X2, y2, cv=kf, scoring='neg_mean_squared_error').mean()
        if r2 > best_r2_l2:
            best_r2_l2 = r2
            best_lasso2 = (deg, alpha, r2, mse)
print(f"VAR2 Lasso (deg {best_lasso2[0]}, alpha={best_lasso2[1]}):  CV R2 = {best_lasso2[2]:.6f}, MSE = {best_lasso2[3]:.6f}")
