import pytest, numpy as np, pandas as pd

def test_if_prices_coordinates_database_contains_null_or_nan(assess_module):
    assert assess_module.prices_coordinates_database_content_check()