import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the pre-trained model, scaler, and feature names
@st.cache_resource
def load_artifacts():
    try:
        # Load the model using a relative path
        with open("best_xgb_model.pkl", "rb") as model_file:
            model = pickle.load(model_file)

        # Load the scaler
        with open("scaler.pkl", "rb") as scaler_file:
            scaler = pickle.load(scaler_file)

        # Load the feature names
        with open("feature_names.pkl", "rb") as feature_file:
            feature_names = pickle.load(feature_file)

        return model, scaler, feature_names
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return None, None, None

# Align input features with training features
def align_features(input_df, feature_names):
    """
    Align the input data with the feature names used during training.
    """
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]
    return input_df

# Convert 'total_sqft' to numeric
def convert_sqft_to_num(sqft):
    try:
        if isinstance(sqft, str) and '-' in sqft:
            sqft_range = sqft.split('-')
            return (float(sqft_range[0]) + float(sqft_range[1])) / 2
        return float(sqft)
    except:
        return None

# Main function for the Streamlit app
def main():
    st.title('Bengaluru House Price Prediction 1')

    # Load the model, scaler, and feature names
    model, scaler, feature_names = load_artifacts()

    # Collect user input
    st.header("Enter Property Details")
    total_sqft = st.text_input('Total Square Feet', value='1000')
    bath = st.number_input('Number of Bathrooms', min_value=1, max_value=10, value=2)
    balcony = st.number_input('Number of Balconies', min_value=0, max_value=10, value=1)
    location = st.text_input('Location (e.g., Electronic City Phase II)', value='Electronic City Phase II')
    area_type = st.selectbox('Area Type', options=['Super built-up Area', 'Built-up  Area', 'Plot Area', 'Carpet Area'])
    size = st.selectbox('Size (e.g., 2 BHK, 3 BHK)', options=['1 BHK', '2 BHK', '3 BHK', '4 BHK', '5 BHK'])

    # Prediction button
    if st.button('Predict House Price'):
        if model is not None and scaler is not None and feature_names is not None:
            # Prepare the input data
            input_data = {
                'total_sqft': convert_sqft_to_num(total_sqft),
                'bath': bath,
                'balcony': balcony,
                'location': location,
                'area_type': area_type,
                'size': size
            }

            # Create a DataFrame for the input data
            input_df = pd.DataFrame([input_data])

            # One-hot encode categorical features
            categorical_features = ['location', 'area_type', 'size']
            encoded_df = pd.get_dummies(input_df, columns=categorical_features, drop_first=True)

            # Align features with the training model
            aligned_df = align_features(encoded_df, feature_names)

            # Scale the input features
            scaled_input = scaler.transform(aligned_df)

            # Predict the price using the model
            prediction = model.predict(scaled_input)[0]

            # Display the predicted house price
            st.success(f'Predicted House Price: ₹{prediction:,.2f}')
        else:
            st.error("Model, scaler, or feature names not loaded. Please check the required files.")

if __name__ == '__main__':
    main()
