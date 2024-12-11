# tests/test_bengaluru_demo.py
from bengaluru_demo import convert_sqft_to_num

def test_convert_sqft_to_num():
    assert convert_sqft_to_num('1000-2000') == 1500
    assert convert_sqft_to_num('1200') == 1200
    assert convert_sqft_to_num('invalid') is None
