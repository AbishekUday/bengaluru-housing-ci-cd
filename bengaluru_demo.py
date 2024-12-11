# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import pickle
import matplotlib.pyplot as plt

# Step 1: Load the dataset
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

def load_data(file_path):
    """
    Load the dataset from a specified file path.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"File not found at path: {file_path}")
        return None

def convert_sqft_to_num(x):
    """
    Convert the 'total_sqft' column to numeric.
    """
    try:
        if '-' in str(x):
            parts = str(x).split('-')
            return (float(parts[0]) + float(parts[1])) / 2
        return float(x)
    except (ValueError, TypeError):
        return None

import pandas as pd
import numpy as np

def preprocess_data(data):
    """
    Preprocesses the data by handling missing values, creating new features, and cleaning columns.
    """
    # Handle missing values
    data["location"].fillna("Unknown", inplace=True)  # Replace null locations with 'Unknown'
    data["size"].fillna("Unknown", inplace=True)      # Replace null sizes with 'Unknown'
    data["society"].fillna("Unknown", inplace=True)   # Replace null societies with 'Unknown'
    data["bath"].fillna(data["bath"].median(), inplace=True)  # Fill missing 'bath' with median
    data["balcony"].fillna(0, inplace=True)           # Replace null balconies with 0

    # Feature Engineering: Example for price per square foot
    data["total_sqft"] = pd.to_numeric(data["total_sqft"], errors="coerce")
    data["price_per_sqft"] = data["price"] / data["total_sqft"]

    # Drop any rows still containing nulls (if any remain)
    data.dropna(inplace=True)
    return data



    # Fill missing values
    data['location'] = data['location'].fillna('Unknown')
    data['size'] = data['size'].fillna('Missing')
    data['bath'] = data['bath'].fillna(data['bath'].mean())
    data['balcony'] = data['balcony'].fillna(data['balcony'].mean())

    # Drop unnecessary columns
    if 'availability' in data.columns:
        data = data.drop(['availability'], axis=1)

    # Extract numeric value from the 'size' column
    data['size'] = data['size'].str.extract(r'(\d+)').fillna(0).astype(int)

    # Convert 'total_sqft' to numeric
    data['total_sqft'] = data['total_sqft'].apply(convert_sqft_to_num)

    # Remove outliers
    upper_limit_sqft = data['total_sqft'].quantile(0.99)
    upper_limit_bath = data['bath'].quantile(0.99)
    upper_limit_price = data['price'].quantile(0.99)

    data['total_sqft'] = np.where(data['total_sqft'] > upper_limit_sqft, upper_limit_sqft, data['total_sqft'])
    data['bath'] = np.where(data['bath'] > upper_limit_bath, upper_limit_bath, data['bath'])
    data['price'] = np.where(data['price'] > upper_limit_price, upper_limit_price, data['price'])

    # Create new features
    data['price_per_sqft'] = data['price'] / data['total_sqft']

    # One-hot encode categorical features
    categorical_features = ['location', 'area_type', 'society', 'size']
    data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

    return data

def save_model_objects(feature_names, scaler, model, path_prefix):
    """
    Save feature names, scaler, and model to disk.
    """
    with open(f"{path_prefix}/feature_names.pkl", 'wb') as f:
        pickle.dump(feature_names, f)
    with open(f"{path_prefix}/scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    with open(f"{path_prefix}/best_xgb_model.pkl", 'wb') as f:
        pickle.dump(model, f)

def hyperparameter_tuning(X_train, y_train):
    """
    Perform hyperparameter tuning using RandomizedSearchCV for XGBoost.
    """
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 8, 10],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2]
    }

    xgb_model = XGBRegressor(random_state=42)
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=10,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)
    return random_search.best_estimator_, random_search.best_params_

def main():
    # File path configuration
    file_path = r"C:\Users\Indra\Desktop\Praxis\Term 2\MLOPS\bengaluru-housing-ci-cd\tests\Bengaluru_House_Data.csv"
    path_prefix = r"C:\Users\Indra\Desktop\Praxis\Term 2\MLOPS\bengaluru-housing-ci-cd"

    # Load data
    raw_data = load_data(file_path)
    if raw_data is None:
        return

    # Preprocess data
    processed_data = preprocess_data(raw_data)

    # Define features and target
    X = processed_data.drop(['price'], axis=1)
    y = processed_data['price']

    # Save feature names
    feature_names = X.columns

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Hyperparameter tuning
    best_model, best_params = hyperparameter_tuning(X_train_scaled, y_train)
    print(f"Best Parameters: {best_params}")

    # Evaluate the model
    y_pred = best_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared Score: {r2:.4f}")

    # Save objects
    save_model_objects(feature_names, scaler, best_model, path_prefix)

if __name__ == "__main__":
    main()








