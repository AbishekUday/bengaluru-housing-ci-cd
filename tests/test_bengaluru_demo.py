# tests/test_bengaluru_demo.py
import pytest
from bengaluru_demo import load_data

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


