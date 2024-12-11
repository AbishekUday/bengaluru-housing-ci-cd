# tests/test_bengaluru_demo.py
import pytest
from bengaluru_demo import load_data, preprocess_data

def test_load_data():
    # Provide the correct file path for testing
    file_path = "C:/Users/Indra/Desktop/Praxis/Term 2/MLOPS/bengaluru-housing-ci-cd/tests/Bengaluru_House_Data.csv"
    data = load_data(file_path)
    assert data is not None, "Failed to load data."
    assert not data.empty, "Data is empty."

def test_preprocess_data():
    # Load sample data
    file_path = "C:/Users/Indra/Desktop/Praxis/Term 2/MLOPS/bengaluru-housing-ci-cd/tests/Bengaluru_House_Data.csv"
    data = load_data(file_path)
    processed_data = preprocess_data(data)
    assert processed_data is not None, "Failed to preprocess data."
    assert 'location' in processed_data.columns, "'location' column missing in preprocessed data."
    assert 'availability' not in processed_data.columns, "'availability' column not dropped in preprocessed data."




