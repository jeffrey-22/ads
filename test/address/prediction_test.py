import pytest
import pandas as pd, shapely, datetime, math, numpy as np

@pytest.mark.slow_for_db
def test_simple_runnable(address_module):
    address_module.predict_price(52.206767, 0.119229, datetime.date(2023, 1, 1), 'S', validation_level=6)

def test_feature_gen_small_example(address_module):
    pois_list = {("amenity", 'school'): pd.DataFrame.from_dict({'geometry': [shapely.geometry.Point(1, 0)]}),
                 ("amenity", 'hospital'): pd.DataFrame.from_dict({'geometry': [shapely.geometry.Point(1, 0)]}),
                 ("amenity", 'restaurant'): pd.DataFrame.from_dict({'geometry': [shapely.geometry.Point(1, 0)]}),
                 ("public_transport", True): pd.DataFrame.from_dict({'geometry': [shapely.geometry.Point(1, 0)]}),
                 ("shop", True): pd.DataFrame.from_dict({'geometry': [shapely.geometry.Point(1, 0), shapely.geometry.Point(2, 2)]}),
                 ("leisure", True): pd.DataFrame.from_dict({'geometry': [shapely.geometry.Point(1, 0)]})}
    latitude = 3
    longitude = 0
    date = datetime.date(2021, 1, 1)
    property_type = 'O'
    feature_array = address_module.prepare_feature_array_for_unseen_data(pois_list, latitude, longitude, date, property_type)
    feature_array = pd.DataFrame([feature_array])
    expected_feature_array = pd.DataFrame.from_dict({
        'latitude': [3],
        'longitude': [0],
        'date_of_transfer': [date],
        'property_type': ['O'],
        'amenity_school': math.sqrt(10),
        'amenity_hospital': math.sqrt(10),
        'amenity_library': -1,
        'amenity_restaurant': math.sqrt(10),
        'public_transport': math.sqrt(10),
        'shop': math.sqrt(5),
        'leisure': math.sqrt(10),
    })
    def compare_values(x, y):
        if isinstance(x, (float, np.floating)):
            return np.isclose(x, y, rtol=1e-6, atol=1e-6)
        else:
            return x == y
    assert (set(expected_feature_array.columns) == set(feature_array.columns)),\
        f"expected {expected_feature_array.columns} get {feature_array.columns}"
    for col in feature_array.columns:
        assert(len(feature_array) == len(expected_feature_array))
        for i in range(len(feature_array)):
            assert compare_values(expected_feature_array.iloc[i][col], feature_array.iloc[i][col]),\
                f"expected {expected_feature_array.iloc[i][col]} get {feature_array.iloc[i][col]}"
    design_matrix = address_module.process_feature_array_into_design_matrix(feature_array)
    assert(design_matrix.shape == (1, 14))
    assert not np.isnan(design_matrix).any()