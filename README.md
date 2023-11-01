# Life_Expectancy
```python
import pandas as pd
import seaborn as sns
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures

# Read your dataset (assuming you have already loaded it)

# Data preprocessing, renaming columns, and encoding

# Visualization with Seaborn
# Top 10 countries with life expectancy
# You've already provided code for this part

# Binary encoding
bin_enc = ce.BinaryEncoder()
df = bin_enc.fit_transform(df)

# Define your machine learning models
models = {
    'LR': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'KNN': KNeighborsRegressor(),
}

# Split data into train and test sets (xtrain, xtest, ytrain, ytest)

# Model fitting and evaluation for non-polynomial features
for name, model in models.items():
    print("=" * 6, name, "=" * 6)
    model.fit(xtrain, ytrain)
    y_pred = model.predict(xtest)
    print("R2", r2_score(ytest, y_pred))
    print("MSE", mean_squared_error(ytest, y_pred, squared=False))
    print("MAE", mean_absolute_error(ytest, y_pred))

# Create polynomial features
poly = PolynomialFeatures()
xtrain_poly = poly.fit_transform(xtrain)
xtest_poly = poly.transform(xtest)

# Model fitting and evaluation for polynomial features
for name, model in models.items():
    print("=" * 6, name, "=" * 6)
    model.fit(xtrain_poly, ytrain)
    y_pred = model.predict(xtest_poly)
    print("R2", r2_score(ytest, y_pred))
    print("MSE", mean_squared_error(ytest, y_pred, squared=False))
    print("MAE", mean_absolute_error(ytest, y_pred))
