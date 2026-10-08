# ML Assignment 1 — Polynomial Regression

## Student
**Roll Number:** BT2024039

## Objective
Develop regularized polynomial regression models to predict a continuous target variable `y` for two independent geothermal engineering problems using personalized datasets.

## Dataset

| Problem | Train File | Test File | Features | Train Rows | Test Rows |
|---------|-----------|-----------|----------|------------|----------|
| VAR1 | `BT2024039_train_var1.csv` | `BT2024039_test_var1.csv` | x1–x6 (6 features) | 1000 | 1000 |
| VAR2 | `BT2024039_train_var2.csv` | `BT2024039_test_var2.csv` | x1–x3 (3 features) | 1000 | 1000 |

All features are normalized within [-1, 1]. Target variable is `y`.

## Problems & Configured Degrees

### VAR1 — Power Plant Steam Turbine Optimization
Predict the Net Power Score from six turbine operational parameters ($x_1 \dots x_6$).
- **Configured Degree:** **Degree 5**
- **Model:** Lasso Polynomial Regression ($L_1$ Sparsity, $\alpha = 0.01$)
- **Active Terms:** 113 of 461 terms retained
- **CV MSE:** 0.3109
- **CV $R^2$:** **0.9689** (96.89%)

### VAR2 — Subterranean Thermal Reservoir Mapping
Predict the Thermal Anomaly Score from three spatial coordinate offsets ($x_1, x_2, x_3$).
- **Configured Degree:** **Degree 10**
- **Model:** Ridge Polynomial Regression ($L_2$ Regularization, $\alpha = 1.0$)
- **Terms:** 285 terms
- **CV MSE:** 0.2515
- **CV $R^2$:** **0.9938** (99.38%)

## Results Summary

| Problem | Model Architecture | Polynomial Degree | Hyperparameters | CV MSE | CV $R^2$ | Train $R^2$ |
|---------|-------------------|:-----------------:|:---------------:|:------:|:--------:|:-----------:|
| **VAR1** | **Lasso Polynomial Regression** | **5** | **$\alpha = 0.01$** | **0.3109** | **0.9689** | **0.9768** |
| **VAR2** | **Ridge Polynomial Regression** | **10** | **$\alpha = 1.0$** | **0.2515** | **0.9938** | **0.9961** |

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run training scripts
python src/train_var1.py
python src/train_var2.py

# Or run the all-in-one script to regenerate predictions
python src/generate_predictions.py
```

## Generated Outputs

- `predictions/BT2024039 pred var1.csv` (and `./BT2024039 pred var1.csv`)
- `predictions/BT2024039 pred var2.csv` (and `./BT2024039 pred var2.csv`)
- `models/var1_model.joblib`
- `models/var2_model.joblib`

## Reproducibility

All random seeds are fixed (`random_state=42`). Running the scripts on the provided datasets reproduces exact results. The codebase uses relative paths and strictly prevents data leakage.

## Project Structure

```
ML-Assignment-1/
├── README.md
├── requirements.txt
├── data/
│   ├── BT2024039_train_var1.csv
│   ├── BT2024039_test_var1.csv
│   ├── BT2024039_train_var2.csv
│   └── BT2024039_test_var2.csv
├── src/
│   ├── train_var1.py
│   ├── train_var2.py
│   └── generate_predictions.py
├── models/
│   ├── var1_model.joblib
│   └── var2_model.joblib
├── predictions/
│   ├── BT2024039 pred var1.csv
│   └── BT2024039 pred var2.csv
└── report/
    ├── ML_Assignment_Report.pdf
    ├── ML_Assignment_Report.html
    └── ML_Assignment_Report.md
```
