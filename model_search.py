import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# LOAD DATA
# ============================================================
train1 = pd.read_csv('BT2024039_train_var1.csv')
test1 = pd.read_csv('BT2024039_test_var1.csv')
train2 = pd.read_csv('BT2024039_train_var2.csv')
test2 = pd.read_csv('BT2024039_test_var2.csv')

print("=== VAR1 CORRELATIONS ===")
print(train1.corr()['y'].drop('y').sort_values(key=abs, ascending=False))
print()
print("=== VAR2 CORRELATIONS ===")
print(train2.corr()['y'].drop('y').sort_values(key=abs, ascending=False))
print()

# ============================================================
# VAR1: MODEL SEARCH
# ============================================================
print("=" * 70)
print("VAR1 MODEL SEARCH")
print("=" * 70)

X1_full = train1[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']]
y1 = train1['y']

kf = KFold(n_splits=5, shuffle=True, random_state=42)

feature_configs_var1 = [
    (['x1'], 'x1 only'),
    (['x1', 'x2'], 'x1,x2'),
    (['x1', 'x2', 'x3'], 'x1,x2,x3'),
    (['x1', 'x2', 'x3', 'x4'], 'x1-x4'),
    (['x1', 'x2', 'x3', 'x4', 'x5', 'x6'], 'all 6'),
]

degrees_var1 = [1, 2, 3, 4, 5]

print(f"{'Features':<20} {'Degree':<8} {'CV MSE':<15} {'CV R2':<12}")
print("-" * 60)

best_var1 = {'mse': np.inf}
all_results_var1 = []

for feat_list, feat_name in feature_configs_var1:
    for deg in degrees_var1:
        X_sub = X1_full[feat_list]
        pipe = Pipeline([
            ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
            ('scaler', StandardScaler()),
            ('reg', LinearRegression())
        ])

        mse_scores = -cross_val_score(pipe, X_sub, y1, cv=kf, scoring='neg_mean_squared_error')
        r2_scores = cross_val_score(pipe, X_sub, y1, cv=kf, scoring='r2')

        mean_mse = mse_scores.mean()
        mean_r2 = r2_scores.mean()

        print(f"{feat_name:<20} {deg:<8} {mean_mse:<15.6f} {mean_r2:<12.6f}")

        all_results_var1.append({
            'features': feat_name, 'degree': deg, 'mse': mean_mse, 'r2': mean_r2
        })

        if mean_mse < best_var1['mse']:
            best_var1 = {
                'features': feat_list, 'name': feat_name,
                'degree': deg, 'mse': mean_mse, 'r2': mean_r2
            }

print()
print(f"BEST VAR1: {best_var1['name']}, degree={best_var1['degree']}, "
      f"MSE={best_var1['mse']:.6f}, R2={best_var1['r2']:.6f}")

# ============================================================
# VAR2: MODEL SEARCH
# ============================================================
print()
print("=" * 70)
print("VAR2 MODEL SEARCH")
print("=" * 70)

X2_full = train2[['x1', 'x2', 'x3']]
y2 = train2['y']

feature_configs_var2 = [
    (['x1'], 'x1 only'),
    (['x1', 'x2'], 'x1,x2'),
    (['x1', 'x2', 'x3'], 'all 3'),
]

degrees_var2 = [1, 2, 3, 4, 5, 6, 7, 8]

print(f"{'Features':<20} {'Degree':<8} {'CV MSE':<15} {'CV R2':<12}")
print("-" * 60)

best_var2 = {'mse': np.inf}
all_results_var2 = []

for feat_list, feat_name in feature_configs_var2:
    for deg in degrees_var2:
        X_sub = X2_full[feat_list]
        pipe = Pipeline([
            ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
            ('scaler', StandardScaler()),
            ('reg', LinearRegression())
        ])

        mse_scores = -cross_val_score(pipe, X_sub, y2, cv=kf, scoring='neg_mean_squared_error')
        r2_scores = cross_val_score(pipe, X_sub, y2, cv=kf, scoring='r2')

        mean_mse = mse_scores.mean()
        mean_r2 = r2_scores.mean()

        print(f"{feat_name:<20} {deg:<8} {mean_mse:<15.6f} {mean_r2:<12.6f}")

        all_results_var2.append({
            'features': feat_name, 'degree': deg, 'mse': mean_mse, 'r2': mean_r2
        })

        if mean_mse < best_var2['mse']:
            best_var2 = {
                'features': feat_list, 'name': feat_name,
                'degree': deg, 'mse': mean_mse, 'r2': mean_r2
            }

print()
print(f"BEST VAR2: {best_var2['name']}, degree={best_var2['degree']}, "
      f"MSE={best_var2['mse']:.6f}, R2={best_var2['r2']:.6f}")

# ============================================================
# FINAL MODELS: RETRAIN ON ALL TRAINING DATA & PREDICT
# ============================================================
print()
print("=" * 70)
print("FINAL MODELS - RETRAIN & PREDICT")
print("=" * 70)

# VAR1 Final
final_feat_var1 = best_var1['features']
final_deg_var1 = best_var1['degree']

pipe_var1 = Pipeline([
    ('poly', PolynomialFeatures(degree=final_deg_var1, include_bias=False)),
    ('scaler', StandardScaler()),
    ('reg', LinearRegression())
])
pipe_var1.fit(X1_full[final_feat_var1], y1)
preds_var1 = pipe_var1.predict(test1[final_feat_var1])

# VAR2 Final
final_feat_var2 = best_var2['features']
final_deg_var2 = best_var2['degree']

pipe_var2 = Pipeline([
    ('poly', PolynomialFeatures(degree=final_deg_var2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('reg', LinearRegression())
])
pipe_var2.fit(X2_full[final_feat_var2], y2)
preds_var2 = pipe_var2.predict(test2[final_feat_var2])

# Save predictions
pred_df_var1 = pd.DataFrame({'y': preds_var1})
pred_df_var2 = pd.DataFrame({'y': preds_var2})

pred_df_var1.to_csv('BT2024039 pred var1.csv', index=False)
pred_df_var2.to_csv('BT2024039 pred var2.csv', index=False)

print(f"VAR1 predictions saved: shape={pred_df_var1.shape}")
print(f"VAR2 predictions saved: shape={pred_df_var2.shape}")

# ============================================================
# VALIDATION OF PREDICTION FILES
# ============================================================
print()
print("=" * 70)
print("PREDICTION FILE VALIDATION")
print("=" * 70)

sample = pd.read_csv('sample_submission.csv')
p1 = pd.read_csv('BT2024039 pred var1.csv')
p2 = pd.read_csv('BT2024039 pred var2.csv')

print(f"Sample submission: columns={list(sample.columns)}, shape={sample.shape}")
print()

for name, pred, test_df in [
    ('VAR1', p1, test1),
    ('VAR2', p2, test2)
]:
    print(f"--- {name} ---")
    print(f"  Rows match test: {len(pred)} == {len(test_df)} -> {len(pred) == len(test_df)}")
    print(f"  Columns: {list(pred.columns)}")
    print(f"  Columns match sample: {list(pred.columns) == list(sample.columns)}")
    print(f"  NaN count: {pred.isnull().sum().sum()}")
    print(f"  Inf count: {np.isinf(pred.values).sum()}")
    print(f"  Dtype: {pred['y'].dtype}")
    print(f"  Prediction range: [{pred['y'].min():.4f}, {pred['y'].max():.4f}]")
    print()

# Training R2 for reference
train_r2_var1 = pipe_var1.score(X1_full[final_feat_var1], y1)
train_r2_var2 = pipe_var2.score(X2_full[final_feat_var2], y2)
print(f"VAR1 Training R2: {train_r2_var1:.6f}")
print(f"VAR2 Training R2: {train_r2_var2:.6f}")

# Print coefficients info
n_poly_feat_var1 = pipe_var1.named_steps['poly'].n_output_features_
n_poly_feat_var2 = pipe_var2.named_steps['poly'].n_output_features_
print(f"VAR1 polynomial features count: {n_poly_feat_var1}")
print(f"VAR2 polynomial features count: {n_poly_feat_var2}")
