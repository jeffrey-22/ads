import pytest

@pytest.mark.skip()
@pytest.mark.slow_for_db
def test_select_count_price(access_module):
    conn = access_module.DatabaseConnection.get_connection()
    res = access_module.select_count(conn, 'pp_data')
    while not (type(res) is int):
        res = res[0]
    assert(res == 28258161)

@pytest.mark.skip()
@pytest.mark.slow_for_db
def test_select_count_postcode(access_module):
    conn = access_module.DatabaseConnection.get_connection()
    res = access_module.select_count(conn, 'postcode_data')
    while not (type(res) is int):
        res = res[0]
    assert(res == 2631536)

@pytest.mark.skip()
@pytest.mark.slow_for_db
def test_select_count_prices_coordinates(access_module):
    conn = access_module.DatabaseConnection.get_connection()
    res = access_module.select_count(conn, 'prices_coordinates_data')
    while not (type(res) is int):
        res = res[0]
    assert(res == 28210620)