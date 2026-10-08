import pandas as pd
import numpy as np
import os

print("=" * 60)
print("FINAL VERIFICATION")
print("=" * 60)

sample = pd.read_csv('sample_submission.csv')
p1 = pd.read_csv('BT2024039 pred var1.csv')
p2 = pd.read_csv('BT2024039 pred var2.csv')
test1 = pd.read_csv('BT2024039_test_var1.csv')
test2 = pd.read_csv('BT2024039_test_var2.csv')

checks = []

print("Sample submission: columns={}, shape={}, dtype={}".format(
    list(sample.columns), sample.shape, sample['y'].dtype))
print()

for name, pred, test_df in [('VAR1', p1, test1), ('VAR2', p2, test2)]:
    print("--- {} ---".format(name))
    c1 = len(pred) == len(test_df)
    c2 = list(pred.columns) == list(sample.columns)
    c3 = len(pred.columns) == 1
    c4 = pred.isnull().sum().sum() == 0
    c5 = np.isfinite(pred.values).all()
    c6 = pred['y'].dtype == np.float64

    status = lambda c: "PASS" if c else "FAIL"
    print("  [{}] Row count: {} == {}".format(status(c1), len(pred), len(test_df)))
    print("  [{}] Columns match sample: {}".format(status(c2), list(pred.columns)))
    print("  [{}] Single column".format(status(c3)))
    print("  [{}] No NaN".format(status(c4)))
    print("  [{}] No infinite values".format(status(c5)))
    print("  [{}] Numeric dtype: {}".format(status(c6), pred['y'].dtype))
    print("  Prediction stats: min={:.4f}, max={:.4f}, mean={:.4f}".format(
        pred['y'].min(), pred['y'].max(), pred['y'].mean()))
    print()
    checks.extend([c1, c2, c3, c4, c5, c6])

files_to_check = [
    'BT2024039 pred var1.csv',
    'BT2024039 pred var2.csv',
    'predictions/BT2024039 pred var1.csv',
    'predictions/BT2024039 pred var2.csv',
    'src/train_var1.py',
    'src/train_var2.py',
    'src/generate_predictions.py',
    'requirements.txt',
    'README.md',
    'report/ML_Assignment_Report.md',
]

print("--- File Existence ---")
for f in files_to_check:
    exists = os.path.exists(f)
    status = "PASS" if exists else "FAIL"
    print("  [{}] {}".format(status, f))
    checks.append(exists)

print()
if all(checks):
    print("ALL CHECKS PASSED")
else:
    print("SOME CHECKS FAILED")
