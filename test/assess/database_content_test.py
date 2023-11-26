import pytest, numpy as np, pandas as pd

@pytest.mark.slow_for_db
def test_prices_coordinates_database_content_full_check(assess_module):
    assert assess_module.prices_coordinates_database_content_full_check()
    
def test_prices_coordinates_database_content_basic_check(assess_module):
    assert assess_module.prices_coordinates_database_content_basic_check()