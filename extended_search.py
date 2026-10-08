import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
import warnings
warnings.filterwarnings('ignore')

train2 = pd.read_csv('BT2024039_train_var2.csv')
X2 = train2[['x1', 'x2', 'x3']]
y2 = train2['y']

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Extend VAR2 search to degrees 9-12 (all 3 features) to check stability
print("VAR2 Extended search (all 3 features, degrees 9-12)")
print(f"{'Degree':<8} {'CV MSE':<15} {'CV R2':<12} {'Poly features':<15}")
print("-" * 55)

for deg in range(4, 13):
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
        ('scaler', StandardScaler()),
        ('reg', LinearRegression())
    ])
    mse_scores = -cross_val_score(pipe, X2, y2, cv=kf, scoring='neg_mean_squared_error')
    r2_scores = cross_val_score(pipe, X2, y2, cv=kf, scoring='r2')
    
    n_feat = PolynomialFeatures(degree=deg, include_bias=False).fit_transform(X2.values[:1]).shape[1]
    print(f"{deg:<8} {mse_scores.mean():<15.6f} {r2_scores.mean():<12.6f} {n_feat:<15}")

# Also test VAR1 with degree 6 to check
print()
print("VAR1 Extended (all 6, degrees 3-7)")
train1 = pd.read_csv('BT2024039_train_var1.csv')
X1 = train1[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']]
y1 = train1['y']

for deg in range(3, 8):
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
        ('scaler', StandardScaler()),
        ('reg', LinearRegression())
    ])
    mse_scores = -cross_val_score(pipe, X1, y1, cv=kf, scoring='neg_mean_squared_error')
    r2_scores = cross_val_score(pipe, X1, y1, cv=kf, scoring='r2')
    n_feat = PolynomialFeatures(degree=deg, include_bias=False).fit_transform(X1.values[:1]).shape[1]
    print(f"deg={deg}: CV MSE={mse_scores.mean():.6f}, CV R2={r2_scores.mean():.6f}, poly_features={n_feat}")

# Check fold-wise stability for VAR2 deg 7 vs 8
print()
print("VAR2 fold-wise stability check (all 3 features)")
for deg in [7, 8, 9]:
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
        ('scaler', StandardScaler()),
        ('reg', LinearRegression())
    ])
    mse_scores = -cross_val_score(pipe, X2, y2, cv=kf, scoring='neg_mean_squared_error')
    r2_scores = cross_val_score(pipe, X2, y2, cv=kf, scoring='r2')
    print(f"deg={deg}: MSE per fold={np.round(mse_scores, 4)}, R2 per fold={np.round(r2_scores, 4)}")
