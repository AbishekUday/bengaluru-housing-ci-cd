# tests/test_bengaluru_demo.py
import pytest
from bengaluru_demo import load_data, preprocess_data

def test_load_data():
    """
    Test the `load_data` function to ensure it loads the data correctly.
    """
    # Use the relative path to the test data file
    file_path = "tests/Bengaluru_House_Data.csv"
    
    # Load the data using the function
    data = load_data(file_path)
    
    # Assertions to validate the data
    assert data is not None, "The dataset should not be None."
    assert not data.empty, "The dataset should not be empty."
    assert "price" in data.columns, "The dataset must have a 'price' column."

def test_preprocess_data():
    """
    Test the `preprocess_data` function to ensure it processes the data correctly.
    """
    file_path = "tests/Bengaluru_House_Data.csv"
    data = load_data(file_path)
    
    # Preprocess the data
    processed_data = preprocess_data(data)
    
    # Assertions to validate preprocessing
    assert 'availability' not in processed_data.columns, "The 'availability' column should be removed."



