# tests/test_bengaluru_demo.py
import pandas as pd
from bengaluru_demo import convert_sqft_to_num
import pytest
from unittest.mock import patch

def test_convert_sqft_to_num():
    assert convert_sqft_to_num('1000-2000') == 1500
    assert convert_sqft_to_num('1200') == 1200
    assert convert_sqft_to_num('invalid') is None

def test_data_loading():
    # Mock pandas.read_csv to return a fake DataFrame
    with patch("pandas.read_csv") as mock_read_csv:
        mock_read_csv.return_value = pd.DataFrame({"column1": [1, 2], "column2": ["A", "B"]})

        # Import data to trigger the read_csv mock
        from bengaluru_demo import data
        assert not data.empty  # Check that data is not empty

