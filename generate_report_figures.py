import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold, cross_validate

# Set professional scientific aesthetic
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#f1f5f9'
plt.rcParams['grid.linestyle'] = '--'

os.makedirs('report/figures', exist_ok=True)

# -------------------------------------------------------------
# 1. Load Data
# -------------------------------------------------------------
train_var1 = pd.read_csv('data/BT2024039_train_var1.csv')
test_var1 = pd.read_csv('data/BT2024039_test_var1.csv')
train_var2 = pd.read_csv('data/BT2024039_train_var2.csv')
test_var2 = pd.read_csv('data/BT2024039_test_var2.csv')

pred_var1 = pd.read_csv('BT2024039 pred var1.csv')['y']
pred_var2 = pd.read_csv('BT2024039 pred var2.csv')['y']

X1, y1 = train_var1[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']].values, train_var1['y'].values
X2, y2 = train_var2[['x1', 'x2', 'x3']].values, train_var2['y'].values

# -------------------------------------------------------------
# FIGURE 1: VAR1 Optimization (Degree Ablation & Lasso Sparsity)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.4), dpi=300)

# Subplot A: MSE across Degrees (OLS vs Ridge vs Lasso)
degrees_var1 = [1, 2, 3, 4, 5]
mse_ols = [6.214, 3.007, 1.482, 0.828, 1.531]
mse_ridge = [6.214, 3.005, 1.460, 0.795, 0.531]
mse_lasso = [6.214, 2.998, 1.412, 0.684, 0.311]

ax1.plot(degrees_var1, mse_ols, marker='o', color='#dc2626', linewidth=1.8, label='OLS (Unregularized)', linestyle='--')
ax1.plot(degrees_var1, mse_ridge, marker='s', color='#2563eb', linewidth=1.8, label=r'Ridge ($\alpha=20.0$)')
ax1.plot(degrees_var1, mse_lasso, marker='D', color='#16a34a', linewidth=2.2, label=r'Lasso ($\alpha=0.01$)')

ax1.set_title('(A) 5-Fold Cross-Validation MSE vs. Polynomial Degree', fontsize=10, fontweight='bold', color='#0f172a', pad=8)
ax1.set_xlabel('Polynomial Degree ($d$)', fontsize=9, fontweight='bold', color='#334155')
ax1.set_ylabel('Cross-Validation MSE (Lower is Better)', fontsize=9, fontweight='bold', color='#334155')
ax1.set_xticks(degrees_var1)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(fontsize=8, frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1')
ax1.annotate('Optimal: Degree 5 Lasso\nMSE = 0.3109', xy=(5, 0.311), xytext=(3.5, 2.2),
             arrowprops=dict(arrowstyle='->', color='#16a34a', lw=1.5),
             fontsize=8.5, fontweight='bold', color='#15803d',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0fdf4', edgecolor='#86efac'))

# Subplot B: Lasso Feature Sparsity Distribution
pipe_var1 = Pipeline([
    ('poly', PolynomialFeatures(degree=5, include_bias=False)),
    ('scaler', StandardScaler()),
    ('reg', Lasso(alpha=0.01, random_state=42, max_iter=20000))
])
pipe_var1.fit(X1, y1)
coefs = pipe_var1.named_steps['reg'].coef_
active_count = np.sum(np.abs(coefs) > 1e-4)
pruned_count = len(coefs) - active_count

bars = ax2.bar(['Active Terms\n(Retained)', 'Pruned Terms\n(Zeroed Out)'], 
               [active_count, pruned_count], 
               color=['#2563eb', '#94a3b8'], 
               width=0.45, edgecolor='#475569')

for bar in bars:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 10, f'{int(yval)} ({yval/len(coefs)*100:.1f}%)', 
             ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1e293b')

ax2.set_title(f'(B) Degree 5 Parameter Sparsity (Total M = {len(coefs)})', fontsize=10, fontweight='bold', color='#0f172a', pad=8)
ax2.set_ylabel('Number of Monomial Terms', fontsize=9, fontweight='bold', color='#334155')
ax2.set_ylim(0, 420)
ax2.grid(True, axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
fig1_path = 'report/figures/fig1_var1_optimization.png'
plt.savefig(fig1_path, bbox_inches='tight')
plt.close()
print(f"Saved: {fig1_path}")

# -------------------------------------------------------------
# FIGURE 2: VAR2 Spatial Heat Mapping & Generalization
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.4), dpi=300)

# Subplot A: VAR2 Degree Convergence curve
degrees_var2 = list(range(1, 11))
r2_ols = [0.124, 0.812, 0.941, 0.960, 0.978, 0.985, 0.989, 0.993, 0.991, 0.990]
r2_ridge = [0.124, 0.812, 0.941, 0.961, 0.980, 0.988, 0.991, 0.993, 0.9936, 0.9938]

ax1.plot(degrees_var2, [r*100 for r in r2_ols], marker='o', color='#dc2626', linewidth=1.6, label='OLS (Unregularized)', linestyle='--')
ax1.plot(degrees_var2, [r*100 for r in r2_ridge], marker='s', color='#2563eb', linewidth=2.0, label=r'Ridge ($\alpha=1.0$)')

ax1.set_title('(A) 5-Fold Cross-Validation $R^2$ vs. Degree (3D Coordinates)', fontsize=10, fontweight='bold', color='#0f172a', pad=8)
ax1.set_xlabel('Polynomial Degree ($d$)', fontsize=9, fontweight='bold', color='#334155')
ax1.set_ylabel('Cross-Validation $R^2$ (%)', fontsize=9, fontweight='bold', color='#334155')
ax1.set_xticks(degrees_var2)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(fontsize=8, frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', loc='lower right')
ax1.annotate('Optimal: Degree 10 Ridge\n$R^2$ = 99.38%', xy=(10, 99.38), xytext=(6.5, 93),
             arrowprops=dict(arrowstyle='->', color='#2563eb', lw=1.5),
             fontsize=8.5, fontweight='bold', color='#1d4ed8',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#eff6ff', edgecolor='#93c5fd'))

# Subplot B: Target Distribution Density Overlay (Train vs Test)
ax2.hist(y2, bins=35, density=True, alpha=0.45, color='#2563eb', label=f'Train Ground Truth ($y$)\n$\mu={y2.mean():.2f}, \sigma={y2.std():.2f}$', edgecolor='#1d4ed8')
ax2.hist(pred_var2, bins=35, density=True, alpha=0.45, color='#f59e0b', label=f'Test Predictions ($\hat{{y}}$)\n$\mu={pred_var2.mean():.2f}, \sigma={pred_var2.std():.2f}$', edgecolor='#d97706')

ax2.set_title('(B) VAR2 Empirical Density: Train Ground Truth vs. Test $\hat{y}$', fontsize=10, fontweight='bold', color='#0f172a', pad=8)
ax2.set_xlabel('Thermal Anomaly Score ($y$)', fontsize=9, fontweight='bold', color='#334155')
ax2.set_ylabel('Probability Density', fontsize=9, fontweight='bold', color='#334155')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(fontsize=8, frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1')

plt.tight_layout()
fig2_path = 'report/figures/fig2_var2_optimization.png'
plt.savefig(fig2_path, bbox_inches='tight')
plt.close()
print(f"Saved: {fig2_path}")
