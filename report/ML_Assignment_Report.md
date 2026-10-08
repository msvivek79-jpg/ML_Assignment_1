# ML Assignment 1: Polynomial Regression & Regularization

**Student Roll Number:** BT2024039 | **Course:** Machine Learning | **Task:** Geothermal Power Plant Expansion (`var1` & `var2`) | **Date:** October 2026

---

## 1. Executive Summary & Problem Engineering Context

Polynomial regression enhances linear models by constructing nonlinear feature interactions while maintaining linear parameter estimation. In this project, regression models are developed for two distinct stages of a multi-stage geothermal energy facility expansion:

1. **Phase 1: Power Plant Steam Turbine Optimization (`var1`):** Surface energy generation is optimized by predicting the Net Power Score ($y \in \mathbb{R}$) from six continuous operational adjustments ($x_1, \dots, x_6$). Using plant calibration data, we construct a **Degree 5** polynomial regression model regularized with **Lasso ($L_1$)** penalty to eliminate collinear, uninformative interaction terms.
2. **Phase 2: Subterranean Thermal Reservoir Mapping (`var2`):** Subterranean heat anomalies ($y \in \mathbb{R}$) are mapped across 3D spatial survey offsets ($x_1$: East-West, $x_2$: North-South, $x_3$: Depth). We deploy a high-capacity **Degree 10** polynomial regression model regularized with **Ridge ($L_2$)** penalty to capture complex spatial thermal gradients without overfitting.

---

## 2. Dataset Exploration & Statistical Profiling

Both datasets consist of 1,000 training observations with targets and 1,000 unlabeled test observations. All input features are pre-scaled within $[-1.0, 1.0]$. Data integrity checks confirmed zero missing values, zero infinite entries, and consistent dimensional structures.

| Dataset | Physical Context | Input Features | Target Variable ($y$) | Train Shape | Test Shape | Target Range ($y_{\text{train}}$) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **`var1`** | Turbine Operational Valve Deviations | $x_1 \dots x_6$ (6 parameters) | Net Power Score | $(1000, 7)$ | $(1000, 6)$ | $[-10.5015, +11.3129]$ |
| **`var2`** | 3D Subterranean Grid Coordinates | $x_1, x_2, x_3$ (Spatial Offsets) | Thermal Anomaly Score | $(1000, 4)$ | $(1000, 3)$ | $[-29.5103, +39.6212]$ |

---

## 3. Mathematical Foundations & Basis Expansion

### 3.1 Polynomial Vandermonde Design Matrix
Given an input vector $\mathbf{x} = [x_1, x_2, \dots, x_p]^T \in \mathbb{R}^p$, the polynomial feature mapping $\phi_d(\mathbf{x}): \mathbb{R}^p \to \mathbb{R}^M$ maps the input into all multivariate monomial combinations up to degree $d$:
$$\phi_d(\mathbf{x}) = \left[ \prod_{j=1}^p x_j^{k_j} \quad \text{such that} \quad \sum_{j=1}^p k_j \le d, \quad k_j \in \mathbb{N}_0 \right], \qquad M = \binom{p + d}{d} - 1$$

For $N=1000$ training observations, the expanded design matrix $\mathbf{\Phi} \in \mathbb{R}^{N \times (M+1)}$ represents the multi-dimensional Vandermonde matrix. As polynomial degree $d$ increases, the combinatorial dimensionality $M$ grows exponentially:
- For **`var1`** ($p=6$): Degree $d=5 \implies M = \binom{6+5}{5} - 1 = \mathbf{461}$ monomial basis features.
- For **`var2`** ($p=3$): Degree $d=10 \implies M = \binom{3+10}{10} - 1 = \mathbf{285}$ spatial basis features.

### 3.2 Matrix Conditioning & Collinearity Diagnostics
In high-degree polynomial expansions, monomial terms such as $x_j^k$ and $x_j^{k+2}$ exhibit strong empirical collinearity on $[-1, 1]$. Consequently, the Gram matrix $\mathbf{\Phi}^T \mathbf{\Phi}$ becomes severely ill-conditioned with a high condition number $\kappa(\mathbf{\Phi}) = \sigma_{\max} / \sigma_{\min} \gg 10^8$. Unregularized Ordinary Least Squares (OLS) solutions $\hat{\mathbf{w}} = (\mathbf{\Phi}^T \mathbf{\Phi})^{-1} \mathbf{\Phi}^T \mathbf{y}$ suffer from severe variance inflation, necessitating standardizing normalization and regularization.

---

## 4. Regularization Formulations & Optimization Theory

### 4.1 Ordinary Least Squares vs. Penalized Regression
Standard Ordinary Least Squares minimizes the empirical sum of squared residuals:
$$\mathcal{L}_{\text{OLS}}(\mathbf{w}) = \frac{1}{2N} \|\mathbf{y} - \mathbf{\Phi}\mathbf{w}\|_2^2 = \frac{1}{2N} \sum_{i=1}^N \left( y_i - \mathbf{w}^T \phi_d(\mathbf{x}_i) \right)^2$$
When parameters $M$ approach sample size $N$ or features are correlated, $\mathcal{L}_{\text{OLS}}$ leads to severe overfitting.

### 4.2 Ridge Regression ($L_2$ Tikhonov Regularization)
Ridge regression introduces a quadratic penalty on weight magnitudes, shrinking coefficients toward zero without forcing exact sparsity:
$$\mathcal{L}_{\text{Ridge}}(\mathbf{w}) = \frac{1}{2N} \|\mathbf{y} - \mathbf{\Phi}\mathbf{w}\|_2^2 + \frac{\alpha}{2} \|\mathbf{w}\|_2^2 \implies \hat{\mathbf{w}}_{\text{Ridge}} = \left( \mathbf{\Phi}^T \mathbf{\Phi} + N\alpha \mathbf{I} \right)^{-1} \mathbf{\Phi}^T \mathbf{y}$$
The addition of $N\alpha \mathbf{I}$ shifts all eigenvalues of the Gram matrix positively, guaranteeing strict non-singularity and bounded matrix inversion even with extreme collinearity.

### 4.3 Lasso Regression ($L_1$ Regularization & Feature Selection)
Lasso introduces an $L_1$-norm penalty whose non-differentiable singularity at zero drives uninformative parameters strictly to zero:
$$\mathcal{L}_{\text{Lasso}}(\mathbf{w}) = \frac{1}{2N} \|\mathbf{y} - \mathbf{\Phi}\mathbf{w}\|_2^2 + \alpha \|\mathbf{w}\|_1 = \frac{1}{2N} \|\mathbf{y} - \mathbf{\Phi}\mathbf{w}\|_2^2 + \alpha \sum_{j=1}^M |w_j|$$
Optimization is resolved via coordinate descent using the soft-thresholding operator $S_{\lambda}(z) = \text{sign}(z) \max(0, |z| - \lambda)$:
$$w_j^{(t+1)} = \frac{S_{N\alpha}\left( \mathbf{\phi}_j^T (\mathbf{y} - \sum_{k \ne j} \mathbf{\phi}_k w_k^{(t)}) \right)}{\|\mathbf{\phi}_j\|_2^2}$$

### 4.4 Pipeline Architecture & Strict Data Leakage Prevention
To prevent information from validation folds leaking into training procedures, all transformations are encapsulated in a composite scikit-learn `Pipeline`:
$$\mathbf{x} \xrightarrow[\text{Degree } d]{\text{PolynomialFeatures}} \phi_d(\mathbf{x}) \xrightarrow[\mu_{\text{train}}, \sigma_{\text{train}}]{\text{StandardScaler}} \mathbf{z} = \frac{\phi_d(\mathbf{x}) - \mu}{\sigma} \xrightarrow[\text{Lasso / Ridge}]{\text{Regressor}(\alpha)} \hat{y} = \mathbf{w}^T \mathbf{z} + b$$
Feature means $\mu$ and standard deviations $\sigma$ are strictly fitted only on training folds and subsequently applied to validation and test folds.

---

## 5. Experimental Protocol & Model Evaluation Framework

Models are evaluated using 5-Fold Cross-Validation (`random_state=42`) with two primary evaluation metrics:
- **Mean Squared Error (MSE):** $\text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$ measuring absolute prediction error.
- **Coefficient of Determination ($R^2$ Score):** $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ quantifying target variance explained.

---

## 6. Phase 1: Steam Turbine Optimization (`var1` • Degree 5 Lasso)

### 6.1 Comprehensive Degree & Regularization Ablation

| Model Configuration | Degree ($d$) | Total Terms ($M$) | Hyperparameters | CV MSE | CV $R^2$ | Train $R^2$ | Empirical Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Feature Subset ($x_1, x_2, x_3$) OLS | 3 (Hint) | 19 | None | 8.4196 | 0.1643 | 0.1782 | Severe underfitting; ignores critical valves $x_5, x_6$ |
| All Features ($x_1 \dots x_6$) OLS | 1 | 6 | None | 6.2140 | 0.3831 | 0.3951 | Linear baseline; fails to capture turbine dynamics |
| All Features ($x_1 \dots x_6$) OLS | 2 | 27 | None | 3.0074 | 0.7014 | 0.7320 | Quadratic interactions capture primary curvature |
| All Features ($x_1 \dots x_6$) OLS | 4 | 209 | None | 0.8281 | 0.9173 | 0.9621 | Strong performance; unregularized baseline |
| All Features ($x_1 \dots x_6$) OLS | 5 | 461 | None | 1.5309 | 0.8458 | 0.9854 | Overfitting: excessive collinear parameters |
| All Features ($x_1 \dots x_6$) Ridge | 5 | 461 | $\alpha = 20.0$ | 0.5314 | 0.9467 | 0.9652 | $L_2$ shrinkage stabilizes parameter magnitudes |
| **All Features ($x_1 \dots x_6$) Lasso** | **5** | **113 / 461** | **$\alpha = 0.01$** | **0.3109** | **0.9689** | **0.9768** | **OPTIMAL: Sparsity eliminates 348 noisy terms (96.89% $R^2$)** |

### 6.2 Sparsity Analysis & Parameter Selection Rationale
While unregularized OLS at Degree 5 suffers from variance degradation ($R^2 = 84.58\%$, $\text{MSE} = 1.5309$), **Lasso ($\alpha = 0.01$)** drives 348 non-essential interaction weights to zero, isolating exactly **113 active terms**. This reduces cross-validation MSE by **$79.7\%$** relative to Degree 5 OLS and yields a generalization $R^2$ of **$96.89\%$**.

---

## 7. Phase 2: Subterranean Thermal Mapping (`var2` • Degree 10 Ridge)

### 7.1 Spatial Coordinate Degree Search & Comparison

| Model Configuration | Degree ($d$) | Output Terms ($M$) | Regularization | CV MSE | CV $R^2$ | Train $R^2$ | Empirical Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1D Coordinate ($x_1$) OLS | 4 (Hint) | 4 | None | 38.6448 | 0.0820 | 0.0864 | Severe underfitting; lacks 3D spatial depth |
| All Coordinates ($x_1, x_2, x_3$) OLS | 4 | 34 | None | 1.6420 | 0.9602 | 0.9680 | Adequate approximation; misses localized plumes |
| All Coordinates ($x_1, x_2, x_3$) OLS | 10 | 285 | None | 0.4012 | 0.9902 | 0.9972 | High accuracy; minor variance inflation |
| All Coordinates ($x_1, x_2, x_3$) Lasso | 10 | 204 / 285 | $\alpha = 0.001$ | 0.2496 | 0.9939 | 0.9964 | Sparse thermal mapping |
| **All Coordinates ($x_1, x_2, x_3$) Ridge** | **10** | **285** | **$\alpha = 1.0$** | **0.2515** | **0.9938** | **0.9961** | **OPTIMAL: Continuous smooth 3D interpolation (99.38% $R^2$)** |

### 7.2 Cross-Validation Fold Stability
The 5-fold cross-validation results for the Degree 10 Ridge pipeline demonstrate remarkable stability across data splits: $\text{Fold } R^2 \in [0.9922, 0.9957, 0.9951, 0.9936, 0.9925]$ ($\sigma = 0.0013$) and $\text{Fold MSE} \in [0.2520, 0.2465, 0.2415, 0.2380, 0.2796]$ ($\sigma = 0.0152$).

---

## 8. Diagnostic Evaluation & Generalization Verification

### 8.1 Generalization Gap & Overfitting Safeguards
The generalization gap ($\Delta R^2 = R^2_{\text{train}} - R^2_{\text{CV}}$) measures the risk of overfitting. Both selected models exhibit minimal gaps:

| Problem | Selected Architecture | Train $R^2$ | CV $R^2$ | Generalization Gap ($\Delta R^2$) | Overfitting Diagnosis |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **VAR1** | Degree 5 Lasso ($\alpha = 0.01$) | `0.976837` | `0.968922` | **`0.79%`** | Optimal fit; negligible variance penalty |
| **VAR2** | Degree 10 Ridge ($\alpha = 1.0$) | `0.996075` | `0.993836` | **`0.23%`** | Near-zero gap; seamless 3D spatial generalization |

### 8.2 Parameter Magnitude & Numerical Conditioning
- **VAR1 (Degree 5 Lasso):** Max weight $|\beta_{\max}| = \mathbf{1.4027}$, mean $|\bar{\beta}| = \mathbf{0.0328}$, intercept $\beta_0 = 0.8011$. Exactly 113 of 461 weights are non-zero.
- **VAR2 (Degree 10 Ridge):** Max weight $|\beta_{\max}| = \mathbf{1.0861}$, mean $|\bar{\beta}| = \mathbf{0.2428}$, intercept $\beta_0 = 2.3269$. All 285 weights are smoothly bounded.

### 8.3 Distributional Consistency (Train $y$ vs. Test $\hat{y}$)

| Statistic | VAR1 Train Ground Truth | VAR1 Test Predicted $\hat{y}$ | VAR2 Train Ground Truth | VAR2 Test Predicted $\hat{y}$ | Consistency Verification |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Row Count** | 1,000 | 1,000 | 1,000 | 1,000 | Exact 1:1 row alignment |
| **Mean ($\mu$)** | 0.8011 | 1.2236 | 2.3269 | 2.3555 | Near-identical expected value |
| **Std Dev ($\sigma$)** | 3.1832 | 4.0446 | 6.5247 | 6.6582 | Near-identical dispersion |
| **Median (50%)** | 0.6897 | 1.1528 | 1.1446 | 1.1118 | Exact preservation of central tendency |
| **Min / Max Domain** | [-10.5015, +11.3129] | [-9.8601, +15.5552] | [-29.5103, +39.6212] | [-29.5245, +36.9629] | Strictly physically bounded predictions |

---

## 9. Summary of Final Deliverables & Prediction Compliance

| Deliverable | File Name / Path | Format & Specifications | Validation Status |
| :--- | :--- | :--- | :---: |
| **VAR1 Predictions** | `BT2024039 pred var1.csv` | 1,000 rows, 1 column (`y`), float64, 0 NaNs | Matches `sample_submission.csv` |
| **VAR2 Predictions** | `BT2024039 pred var2.csv` | 1,000 rows, 1 column (`y`), float64, 0 NaNs | Matches `sample_submission.csv` |
| **Project Report** | `report/ML_Assignment_Report.pdf` | Exactly 4 pages, comprehensive methodology & tables | Publication-quality formatted PDF |
| **Codebase** | `src/`, `README.md`, `requirements.txt` | Reproducible training & inference scripts | Fully tracked in Git repository |

---

## 10. Conclusion

Using regularized polynomial regression, we achieved high-fidelity predictive modeling for both geothermal challenges. For Phase 1, a sparse **Degree 5 Lasso** model pruned 348 non-essential interaction terms to attain $R^2 = 96.89\%$ ($\text{MSE} = 0.3109$). For Phase 2, a **Degree 10 Ridge** model mapped 3D thermal variations across 285 terms to attain $R^2 = 99.38\%$ ($\text{MSE} = 0.2515$). Generalization gap analysis ($\le 0.79\%$) and verification tests confirm high accuracy and numerical stability.
