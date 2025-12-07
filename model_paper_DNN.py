import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def run_dnn():
    print("Running Adv Model 1: Deep Neural Network")
    try:
        df = pd.read_csv('train.csv')
    except:
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
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', MLPRegressor(random_state=42, max_iter=500))])

    param_grid = {
        'regressor__hidden_layer_sizes': [(100,), (100, 50)],
        'regressor__learning_rate_init': [0.001, 0.01]
    }

    grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
    grid.fit(X_train, y_train)

    y_pred = np.expm1(grid.best_estimator_.predict(X_test))
    y_test_orig = np.expm1(y_test)

    print(f"Best Params: {grid.best_params_}")
    print(f"MSE: {mean_squared_error(y_test_orig, y_pred):,.2f}")
    print(f"MAE: {mean_absolute_error(y_test_orig, y_pred):,.2f}")
    
    plt.plot(grid.best_estimator_['regressor'].loss_curve_)
    plt.title("Neural Network Loss Curve")
    plt.show()

if __name__ == "__main__":
    run_dnn()