import pytest, numpy as np, pandas as pd

def test_all_tags_exist_for_20_sampled_locations(assess_module):
    locations_df = assess_module.extract_locations_from_prices_coordinates_database()
    locations_df = locations_df.sample(n = 20)
    for i in range(len(locations_df)):
        lat = locations_df.iloc[i]['latitude']
        lon = locations_df.iloc[i]['longitude']
        dis_list = np.array(assess_module.distance_extraction_from_closest(lat, lon))
        assert(len(dis_list) == len(assess_module.default_tag_list)), f"len neq for lat={lat}, lon={lon}"
        assert(dis_list.min() >= 0), f"dis min < 0 for lat={lat}, lon={lon}"
        assert(dis_list.max() <= 5), f"dis max > 5 for lat={lat}, lon={lon}"