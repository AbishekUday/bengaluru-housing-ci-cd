# tests/test_bengaluru_demo.py
import pytest
from bengaluru_demo import load_data, preprocess_data, convert_sqft_to_num

def test_load_data():
    file_path = r"C:\Users\Indra\Desktop\Praxis\Term 2\MLOPS\bengaluru-housing-ci-cd\tests\Bengaluru_House_Data.csv"
    data = load_data(file_path)
    assert data is not None, "Failed to load data."
    assert not data.empty, "Loaded data is empty."

def test_convert_sqft_to_num():
    assert convert_sqft_to_num("1200") == 1200.0
    assert convert_sqft_to_num("1200-1500") == 1350.0
    assert convert_sqft_to_num("abc") is None

def test_preprocess_data():
    file_path = r"C:\Users\Indra\Desktop\Praxis\Term 2\MLOPS\bengaluru-housing-ci-cd\tests\Bengaluru_House_Data.csv"
    data = load_data(file_path)
    processed_data = preprocess_data(data)
    assert processed_data is not None, "Preprocessing failed."
    assert 'price_per_sqft' in processed_data.columns, "Feature 'price_per_sqft' not created."
    assert 'location' not in processed_data.columns, "'location' column not encoded."







