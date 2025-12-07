import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def run_rf():
    print("Running Adv Model 2: Random Forest (Fu)")
    try:
        df = pd.read_csv('train.csv')
    except:
        return

    X = df.drop(columns=['SalePrice', 'Id'])
    y = np.log1p(df['SalePrice'])

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fu uses Random Forest
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

    model.fit(X_train, y_train)
    
    y_pred = np.expm1(model.predict(X_test))
    y_test_orig = np.expm1(y_test)

    print(f"MSE: {mean_squared_error(y_test_orig, y_pred):,.2f}")
    print(f"MAE: {mean_absolute_error(y_test_orig, y_pred):,.2f}")

    # Fu highlights feature importance
    rf = model.named_steps['regressor']
    plt.bar(range(10), rf.feature_importances_[:10])
    plt.title("Top 10 Feature Importances (Fu)")
    plt.show()

if __name__ == "__main__":
    run_rf()