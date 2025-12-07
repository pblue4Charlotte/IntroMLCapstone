import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error

def run_poly_model():
    print("Running Classic Model: Polynomial Regression")
    try:
        df = pd.read_csv('train.csv')
    except FileNotFoundError:
        print("Error: 'train.csv' not found.")
        return
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    features = [c for c in numeric_cols if c not in ['SalePrice', 'Id']]

    X = df[features]
    y = np.log1p(df['SalePrice'])

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # important: PolynomialFeatures is applied after
    model = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2)),
        ('regressor', LinearRegression())
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)

    y_pred = np.expm1(model.predict(X_test))
    y_test_orig = np.expm1(y_test)

    print("\nPolynomial Regression Results\n")
    print(f"MSE: {mean_squared_error(y_test_orig, y_pred):,.2f}")
    print(f"MAE: {mean_absolute_error(y_test_orig, y_pred):,.2f}")
    
    plt.scatter(y_test_orig, y_pred, color='orange', alpha=0.5)
    plt.title('Polynomial Regression: Actual vs Predicted')
    plt.show()

if __name__ == "__main__":
    run_poly_model()