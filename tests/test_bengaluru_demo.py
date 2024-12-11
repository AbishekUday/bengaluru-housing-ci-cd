# tests/test_bengaluru_demo.py
import pandas as pd
from bengaluru_demo import convert_sqft_to_num
import pytest
from unittest.mock import patch

def test_convert_sqft_to_num():
    # Test the conversion of square footage strings to numeric values
    assert convert_sqft_to_num('1000-2000') == 1500
    assert convert_sqft_to_num('1200') == 1200
    assert convert_sqft_to_num('invalid') is None

def test_data_loading():
    # Mock pandas.read_csv to simulate loading a CSV file
    with patch("pandas.read_csv") as mock_read_csv:
        # Return a fake DataFrame when pandas.read_csv is called
        mock_read_csv.return_value = pd.DataFrame({
            "total_sqft": [1200, 1400],
            "price": [75.0, 80.0],
            "bath": [2, 3],
        })

        # Reload the module to trigger the data loading
        with patch("bengaluru_demo.pd.read_csv", mock_read_csv):
            import bengaluru_demo
            assert not bengaluru_demo.data.empty  # Check that data is loaded
            assert "total_sqft" in bengaluru_demo.data.columns  # Check column presence


