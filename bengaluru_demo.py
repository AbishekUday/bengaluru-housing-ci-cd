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

def load_data(file_name="tests/Bengaluru_House_Data.csv"):
    """
    Load the dataset from the specified file path.

    Args:
        file_name (str): Relative path to the file.

    Returns:
        pd.DataFrame: Loaded and cleaned dataset.
    """
    try:
        data = pd.read_csv(file_name)
        return data
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise

def preprocess_data(data):
    """
    Preprocess the dataset by dropping unnecessary columns.

    Args:
        data (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    if 'availability' in data.columns:
        data = data.drop(['availability'], axis=1)
    return data


# Handle missing values
data['location'] = data['location'].fillna('Unknown')
data['size'] = data['size'].fillna(data['size'].mode()[0])
data['society'] = data['society'].fillna('Unknown')
data['bath'] = data['bath'].fillna(data['bath'].median())
data['balcony'] = data['balcony'].fillna(data['balcony'].median())

# Convert 'total_sqft' to numeric
def convert_sqft_to_num(sqft):
    try:
        if '-' in sqft:
            sqft_range = sqft.split('-')
            return (float(sqft_range[0]) + float(sqft_range[1])) / 2
        return float(sqft)
    except:
        return None

data['total_sqft'] = data['total_sqft'].apply(convert_sqft_to_num)
data = data.dropna(subset=['total_sqft'])

# Step 3: Handle Outliers
upper_limit_sqft = data['total_sqft'].quantile(0.99)
upper_limit_bath = data['bath'].quantile(0.99)
upper_limit_price = data['price'].quantile(0.99)

data['total_sqft'] = np.where(data['total_sqft'] > upper_limit_sqft, upper_limit_sqft, data['total_sqft'])
data['bath'] = np.where(data['bath'] > upper_limit_bath, upper_limit_bath, data['bath'])
data['price'] = np.where(data['price'] > upper_limit_price, upper_limit_price, data['price'])

# Create new features
data['price_per_sqft'] = data['price'] / data['total_sqft']

# Step 4: One-Hot Encoding for Categorical Features
categorical_features = ['location', 'area_type', 'society', 'size']
data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

# Step 5: Define Features and Target
X = data.drop(['price'], axis=1)
y = data['price']  # Raw price, no log transformation

# Save feature names
feature_names = X.columns
with open(r"C:\Users\Indra\Desktop\Praxis\Term 2\MLOPS\bengaluru-housing-ci-cd\feature_names.pkl", 'wb') as f:
    pickle.dump(feature_names, f)

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Preprocessing Numerical Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
with open(r"C:\Users\Indra\Desktop\Praxis\Term 2\MLOPS\bengaluru-housing-ci-cd\scaler.pkl", 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Step 8: Hyperparameter Tuning with RandomizedSearchCV
# Define the hyperparameter grid
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 8, 10],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2]
}

# Initialize XGBoost model
xgb_model = XGBRegressor(random_state=42)

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=10,  # Number of random combinations to try
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Fit the model with RandomizedSearchCV
random_search.fit(X_train_scaled, y_train)

# Best hyperparameters found by RandomizedSearchCV
print(f"Best Parameters: {random_search.best_params_}")

# Predict using the best model
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

# Step 9: Evaluate the Model
mse = mean_squared_error(y_test, y_pred)  # No log transformation needed
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.4f}")

# Step 10: Feature Importance Visualization
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12, 6))
plt.title("Feature Importance (XGBoost)")
plt.barh(range(X_train_scaled.shape[1]), importances[indices], align="center")
plt.yticks(range(X_train_scaled.shape[1]), X_train.columns[indices])
plt.xlabel("Relative Importance")
plt.show()

# Step 11: Save the Model
with open(r"C:\Users\Indra\Desktop\Praxis\Term 2\MLOPS\bengaluru-housing-ci-cd\best_xgb_model.pkl", 'wb') as model_file:
    pickle.dump(best_model, model_file)

print("Model, scaler, and feature names saved successfully!")







