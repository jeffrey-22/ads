import pytest
from datetime import date

@pytest.mark.slow_for_db
def test_simple_runnable(address_module):
    address_module.predict_price(52.206767, 0.119229, date(2023, 1, 1), 'S')

#TODO: Add more