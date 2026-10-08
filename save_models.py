import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression

os.makedirs('models', exist_ok=True)

# Train and save VAR1
train1 = pd.read_csv('data/BT2024039_train_var1.csv')
X1 = train1[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']]
y1 = train1['y']
model1 = Pipeline([
    ('poly', PolynomialFeatures(degree=4, include_bias=False)),
    ('scaler', StandardScaler()),
    ('reg', LinearRegression())
])
model1.fit(X1, y1)
joblib.dump(model1, 'models/var1_model.joblib')

# Train and save VAR2
train2 = pd.read_csv('data/BT2024039_train_var2.csv')
X2 = train2[['x1', 'x2', 'x3']]
y2 = train2['y']
model2 = Pipeline([
    ('poly', PolynomialFeatures(degree=8, include_bias=False)),
    ('scaler', StandardScaler()),
    ('reg', LinearRegression())
])
model2.fit(X2, y2)
joblib.dump(model2, 'models/var2_model.joblib')

print("Models successfully saved to 'models/' directory:")
for f in os.listdir('models'):
    path = os.path.join('models', f)
    print(f" - {path} ({os.path.getsize(path)} bytes)")
