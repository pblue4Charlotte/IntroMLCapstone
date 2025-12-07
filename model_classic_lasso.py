import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def run_lasso_model():
    print("Running Classic Model: Lasso Regression")
    try:
        df = pd.read_csv('train.csv')
    except FileNotFoundError:
        print("Error: 'train.csv' not found.")
        return

    X = df.drop(columns=['SalePrice', 'Id'])
    y = np.log1p(df['SalePrice'])

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', Lasso(max_iter=2000))])

    # Tuning Alpha
    param_grid = {'regressor__alpha': [0.0001, 0.001, 0.01, 0.1]}
    grid = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = np.expm1(best_model.predict(X_test))
    y_test_orig = np.expm1(y_test)

    print(f"Best Alpha: {grid.best_params_}")
    print(f"MSE: {mean_squared_error(y_test_orig, y_pred):,.2f}")
    print(f"MAE: {mean_absolute_error(y_test_orig, y_pred):,.2f}")

    # check features zeroed by Lasso
    lasso_model = best_model.named_steps['regressor']
    print(f"Features kept: {np.sum(lasso_model.coef_ != 0)}")
    print(f"Features eliminated: {np.sum(lasso_model.coef_ == 0)}")

if __name__ == "__main__":
    run_lasso_model()