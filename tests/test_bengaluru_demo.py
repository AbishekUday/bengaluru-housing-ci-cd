# tests/test_bengaluru_demo.py
import pytest
from bengaluru_demo import load_data, preprocess_data
import os

def test_load_data():
    # Use a relative path to the test data file
    file_path = os.path.join("tests", "Bengaluru_House_Data.csv")
    data = load_data(file_path)
    assert data is not None, "Failed to load data."
    assert not data.empty, "Loaded data is empty."

def test_preprocess_data():
    # Use a relative path to the test data file
    file_path = os.path.join("tests", "Bengaluru_House_Data.csv")
    data = load_data(file_path)
    assert data is not None, "Failed to load data for preprocessing."
    
    processed_data = preprocess_data(data)
    assert processed_data is not None, "Preprocessed data is None."
    assert not processed_data.empty, "Preprocessed data is empty."
    assert "price_per_sqft" in processed_data.columns, "Feature engineering missing 'price_per_sqft'."








