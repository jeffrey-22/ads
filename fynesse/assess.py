from .config import *

from . import access

import osmnx as ox, pandas as pd, numpy as np
from datetime import datetime
import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

default_tag_list = [{"amenity": 'school'},
                    {"amenity": 'hospital'},
                    {"amenity": 'library'},
                    {"amenity": 'restaurant'},
                    {"public_transport": True},
                    {"shop": True},
                    {"leisure": True}]

class PricesCoordinatesData:
    _data = None
    prices_coordinates_data_sample_limit = None

    @staticmethod
    def fetch_sample_limit():
        return config['prices_coordinates_data_sample_limit']

    @staticmethod
    def fetch_data():
        conn = access.DatabaseConnection.get_connection()
        query = f'SELECT price, date_of_transfer, property_type,\
                    latitude, longitude FROM prices_coordinates_data'
        if PricesCoordinatesData.prices_coordinates_data_sample_limit is None:
            PricesCoordinatesData.prices_coordinates_data_sample_limit = PricesCoordinatesData.fetch_sample_limit()
            assert(PricesCoordinatesData.prices_coordinates_data_sample_limit >= 0)
            assert(PricesCoordinatesData.prices_coordinates_data_sample_limit <= 28210620)
            query += f" LIMIT {PricesCoordinatesData.prices_coordinates_data_sample_limit};"
        return pd.read_sql_query(query, conn)

    @staticmethod
    def reset_data_with_new_sample_limit(new_sample_limit):
        PricesCoordinatesData.prices_coordinates_data_sample_limit = new_sample_limit
        PricesCoordinatesData._data = PricesCoordinatesData.fetch_data()

    def __init__(self):
        if PricesCoordinatesData._data is None:
            PricesCoordinatesData._data = PricesCoordinatesData.fetch_data()

    @staticmethod
    def get_data():
        if PricesCoordinatesData._data is None:
            PricesCoordinatesData._data = PricesCoordinatesData.fetch_data()
        return PricesCoordinatesData._data.copy()

def prices_coordinates_database_content_check():
    df = PricesCoordinatesData.get_data()
    ok = True
    ok &= len(df) > 1000
    ok &= not df.isnull().values.any()
    ok &= df['price'].min() > 0
    ok &= df['price'].max() < 10000000000
    ok &= type(df['date_of_transfer'].iloc[0]) is datetime.date
    ok &= df['date_of_transfer'].min() == datetime.date(1995, 1, 1)
    ok &= df['date_of_transfer'].max() == datetime.date(2022, 12, 31)
    ok &= sorted(df['property_type'].unique()) == sorted(np.array(['D', 'S', 'T', 'F', 'O']))
    ok &= df['latitude'].min() >= 48
    ok &= df['latitude'].max() <= 65
    ok &= df['longitude'].min() >= -10
    ok &= df['longitude'].max() <= 20
    return ok

def extract_locations_from_prices_coordinates_database(df = PricesCoordinatesData.get_data()):
    df = df[['latitude', 'longitude']]
    df = df[~df.duplicated(keep='last')]
    return df

def get_pois_from_bbox(tag_list, bounding_box):
    try:
        pois = ox.features_from_bbox(north=bounding_box['north'], 
                                        south=bounding_box['south'], 
                                        west=bounding_box['west'], 
                                        east=bounding_box['east'],
                                        tags=tag_list)
        return (len(pois), pois)
    except:
        return (0, ())
    
def generate_bbox(latitude, longitude, box_height = 0.04, box_width = 0.04):
    return {
        'north': latitude + box_height / 2,
        'south': latitude - box_height / 2,
        'west': longitude - box_width / 2,
        'east': longitude + box_width / 2
    }

def get_closest_pois_list(tag_list, feature_list, latitude, longitude, start_deg = 0.002, end_deg = 3, increase_factor = 4):
    pois_list = {}
    for tag in tag_list:
        [(k, v)] = tag.items()
        box_size = start_deg
        sz = 0
        while (sz == 0):
            (sz, pois) = get_pois_from_bbox(tag, generate_bbox(latitude, longitude, box_size, box_size))
            if (box_size > end_deg):
                break
            box_size *= increase_factor
        if (box_size <= end_deg):
            pois = pois[[k] + feature_list]
            pois_list[(k, v)] = pois
    return pois_list

def extract_closest_euclidean_dist_from_pois(pois_list, tag, latitude, longitude, fail_filler = -1):
    def euclidean_distance(a_loc, b_loc):
        (a_lat, a_lon) = (a_loc.y, a_loc.x)
        (b_lat, b_lon) = b_loc
        return ((a_lat - b_lat) ** 2) + ((a_lon - b_lon) ** 2)
    current_location = (latitude, longitude)
    (k, v) = tag
    if (k, v) in pois_list:
        pois = pois_list[(k, v)]
        distance = pois['geometry'].apply(lambda obj: euclidean_distance(obj.centroid, current_location)).min()
    else:
        distance = fail_filler
    return distance

def distance_extraction_from_closest(latitude, longitude, tag_list = default_tag_list):
    feature_list = ["geometry"]
    pois_list = get_closest_pois_list(tag_list, feature_list, latitude, longitude)
    distance_list = []
    for tag in tag_list:
        assert len(list(tag.items())) == 1
        distance_list.append(extract_closest_euclidean_dist_from_pois(pois_list, list(tag.items())[0], latitude, longitude))
    return distance_list

def get_bounded_pois_list(tag_list, bounding_box):
    pois_list = {}
    for tag in tag_list:
        [(k, v)] = tag.items()
        (sz, pois) = get_pois_from_bbox(tag, bounding_box)
        if (sz > 0):
            pois = pois[[k, "geometry"]]
            pois_list[(k, v)] = pois
    return pois_list

def select_all_price_data_within_bbox_and_date_range(bounding_box, date_range, conn):
    lat_min = bounding_box["south"]
    lat_max = bounding_box["north"]
    lon_min = bounding_box["west"]
    lon_max = bounding_box["east"]
    date_min = date_range["start"]
    date_max = date_range["end"]
    query = f'SELECT price, date_of_transfer, property_type,\
                latitude, longitude FROM prices_coordinates_data\
                WHERE latitude >= {lat_min} AND latitude <= {lat_max} AND longitude >= {lon_min} AND longitude <= {lon_max}\
                AND date_of_transfer >= \'{str(date_min)}\' AND date_of_transfer <= \'{str(date_max)}\''
    return pd.read_sql_query(query, conn)

def column_name_of_tag(tag):
    (k, v) = tag
    if (type(v) is str):
        return k + "_" + v
    else:
        return k

def find_bounding_box_for_price_data(price_data):
    assert set(price_data.columns).issubset(['price', 'date_of_transfer', 'property_type', 'latitude', 'longitude']), \
        'cannot find all price data columns'
    bounding_box = {}
    bounding_box["south"] = price_data['latitude'].min()
    bounding_box["north"] = price_data['latitude'].max()
    bounding_box["west"] = price_data['longitude'].min()
    bounding_box["east"] = price_data['longitude'].max()
    return bounding_box

def pad_bounding_box(bounding_box, padding_deg = 0.1):
    padded_bounding_box = {'south': bounding_box['south'] - padding_deg,
                           'north': bounding_box['north'] + padding_deg,
                           'west': bounding_box['west'] - padding_deg,
                           'east': bounding_box['east'] + padding_deg}
    return padded_bounding_box

def prepare_full_price_data_from_price_data(price_data, padding_deg = 0.1, tag_list = default_tag_list):
    bounding_box = find_bounding_box_for_price_data(price_data)
    pois_bounding_box = pad_bounding_box(bounding_box, padding_deg)
    pois_list = get_bounded_pois_list(tag_list, pois_bounding_box)
    for (k, v) in pois_list:
        col_name = column_name_of_tag((k, v))
        price_data[col_name] = price_data.apply(lambda row: \
                                                extract_closest_euclidean_dist_from_pois(pois_list,
                                                                                         (k, v),
                                                                                         row['latitude'], 
                                                                                         row['longitude']), 
                                                                                         axis=1)
    return price_data, pois_list
    
def prepare_full_price_data_within_bbox_and_date_range(bounding_box, date_range, conn, \
                                                  padding_deg = 0.1, tag_list = default_tag_list, \
                                                  ):
    pois_bounding_box = pad_bounding_box(bounding_box, padding_deg)
    pois_list = get_bounded_pois_list(tag_list, pois_bounding_box)
    price_data = select_all_price_data_within_bbox_and_date_range(bounding_box, date_range, conn)
    for (k, v) in pois_list:
        col_name = column_name_of_tag((k, v))
        price_data[col_name] = price_data.apply(lambda row: \
                                                extract_closest_euclidean_dist_from_pois(pois_list,
                                                                                         (k, v),
                                                                                         row['latitude'], 
                                                                                         row['longitude']), 
                                                                                         axis=1)
    return price_data, pois_list

def one_hot_encode_column(df, column_name):
    one_hot_encoded_df = pd.get_dummies(df[column_name])
    one_hot_encoded_df.columns = [column_name + "_" + val for val in one_hot_encoded_df.columns]
    df = df.drop(column_name, axis=1)
    df = pd.concat([df, one_hot_encoded_df], axis=1)
    return df

def date_to_days_encode_column(df, column_name):
    reference_date = datetime.datetime(1990, 1, 1)
    example_obj = df.iloc[0][column_name]
    data = df[column_name]
    if isinstance(example_obj, datetime.datetime):
        data = data.apply(lambda dt: (dt - reference_date).days)
    elif isinstance(example_obj, datetime.date):
        data = data.apply(lambda dt: (dt - reference_date.date()).days)
    else:
        raise ValueError("Unsupported type in df. Supported types: datetime, date")
    df[column_name] = data
    return df

def general_PCA_plot(df):
    df_standardized = (df - df.mean()) / df.std()
    assert df.notnull().all().all()
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_standardized)
    pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    plt.scatter(pc_df['PC1'], pc_df['PC2'])
    plt.title('PCA Plot')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

def general_PCA_plot_with_one_column_colorcoded(df, colorcoded_column_name = 'price'):
    assert df.notnull().all().all()
    assert colorcoded_column_name in df.columns
    columns_for_PCA = df.columns.difference([colorcoded_column_name])
    df_standardized = (df[columns_for_PCA] - df[columns_for_PCA].mean()) / df[columns_for_PCA].std()
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_standardized)
    pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    plt.scatter(pc_df['PC1'], pc_df['PC2'], c=df[colorcoded_column_name], cmap='viridis')
    plt.title(f'PCA Plot with {colorcoded_column_name} color coded')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label=colorcoded_column_name)
    plt.show()

def plot_general_house_distribution(price_data):
    pass

def encode_pure_price_data(price_data):
    assert set(price_data.columns).issubset(['price', 'date_of_transfer', 'property_type', 'latitude', 'longitude'])
    price_data = price_data.copy()
    price_data = one_hot_encode_column(price_data, 'property_type')
    price_data = date_to_days_encode_column(price_data, 'date_of_transfer')
    return price_data