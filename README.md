# Life_Expectancy
# Import necessary libraries
import pandas as pd
import seaborn as sns
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures



# Split the data into training and testing sets (xtrain, xtest, ytrain, ytest)


# Binary encoding for categorical data
bin_enc = ce.BinaryEncoder()
df = bin_enc.fit_transform(df)

# Define  machine learning models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'K-Nearest Neighbors': KNeighborsRegressor(),
}


# Model fitting and evaluation for non-polynomial features
for model_name, model in models.items():
    print("=" * 6, model_name, "=" * 6)
    model.fit(xtrain, ytrain)  # Fit the model
    y_pred = model.predict(xtest)  # Make predictions
    print("R-squared (R2) Score:", r2_score(ytest, y_pred))  # Evaluate using R2 score
    print("Root Mean Squared Error (RMSE):", mean_squared_error(ytest, y_pred, squared=False))  # Evaluate using RMSE
    print("Mean Absolute Error (MAE):", mean_absolute_error(ytest, y_pred))  # Evaluate using MAE

# Create polynomial features
poly = PolynomialFeatures()
xtrain_poly = poly.fit_transform(xtrain)
xtest_poly = poly.transform(xtest)

# Model fitting and evaluation for polynomial features
for model_name, model in models.items():
    print("=" * 6, model_name, "=" * 6)
    model.fit(xtrain_poly, ytrain)  # Fit the model with polynomial features
    y_pred = model.predict(xtest_poly)  # Make predictions
    print("R-squared (R2) Score (Polynomial Features):", r2_score(ytest, y_pred))  # Evaluate using R2 score
    print("Root Mean Squared Error (RMSE) (Polynomial Features):", mean_squared_error(ytest, y_pred, squared=False))  # Evaluate using RMSE
    print("Mean Absolute Error (MAE) (Polynomial Features):", mean_absolute_error(ytest, y_pred))  # Evaluate using MAE
