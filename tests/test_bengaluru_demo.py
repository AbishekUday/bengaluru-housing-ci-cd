# tests/test_bengaluru_demo.pyimport pytest
import pytest
from bengaluru_demo import load_data, preprocess_data
import os

def test_load_data():
    """
    Test the load_data function to ensure it correctly loads data from a CSV file.
    """
    file_path = os.path.join(os.path.dirname(__file__), "Bengaluru_House_Data.csv")
    assert os.path.exists(file_path), f"Test file not found at path: {file_path}"
    data = load_data(file_path)
    assert data is not None, "Failed to load data."
    assert not data.empty, "Loaded data is empty."
    assert "location" in data.columns, "Expected column 'location' not found in the data."

def test_preprocess_data():
    """
    Test the preprocess_data function to ensure it preprocesses data correctly.
    """
    file_path = os.path.join(os.path.dirname(__file__), "Bengaluru_House_Data.csv")
    assert os.path.exists(file_path), f"Test file not found at path: {file_path}"
    data = load_data(file_path)
    assert data is not None, "Failed to load data for preprocessing."

    # Check for nulls before preprocessing
    null_count_before = data.isnull().sum().sum()

    processed_data = preprocess_data(data)

    # Check for nulls after preprocessing
    null_count_after = processed_data.isnull().sum().sum()

    # Assertions
    assert processed_data is not None, "Preprocessed data is None."
    assert not processed_data.empty, "Preprocessed data is empty."
    assert null_count_before > 0, "No null values to preprocess."
    assert null_count_after == 0, "Preprocessed data contains null values."
    assert "price_per_sqft" in processed_data.columns, "Feature engineering missing 'price_per_sqft'."

def test_preprocess_data():
    """
    Test the preprocess_data function to ensure it preprocesses data correctly.
    """
    # Use a relative path to the test data file
    file_path = os.path.join(os.path.dirname(__file__), "Bengaluru_House_Data.csv")

    # Check if the file exists before loading
    assert os.path.exists(file_path), f"Test file not found at path: {file_path}"

    # Load the data
    data = load_data(file_path)
    assert data is not None, "Failed to load data for preprocessing."

    # Preprocess the data
    processed_data = preprocess_data(data)

    # Assertions
    assert processed_data is not None, "Preprocessed data is None."
    assert not processed_data.empty, "Preprocessed data is empty."
    assert "price_per_sqft" in processed_data.columns, "Feature engineering missing 'price_per_sqft'."
    assert processed_data.isnull().sum().sum() == 0, "Preprocessed data contains null values."








