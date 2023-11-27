from .config import *

from . import access

import osmnx as ox, pandas as pd, numpy as np
from datetime import datetime
import datetime, geopandas, shapely
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

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
        # TODO: Make this sample RANDOM
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
    
    @staticmethod
    def get_sample_limit():
        if PricesCoordinatesData.prices_coordinates_data_sample_limit is None:
            PricesCoordinatesData.prices_coordinates_data_sample_limit = PricesCoordinatesData.fetch_sample_limit()
        return PricesCoordinatesData.prices_coordinates_data_sample_limit

def prices_coordinates_database_content_full_check():
    full_size = 28210620
    current_size = PricesCoordinatesData.get_sample_limit()
    changed = False
    if (current_size < full_size):
        PricesCoordinatesData.reset_data_with_new_sample_limit(full_size)
        changed = True
    df = PricesCoordinatesData.get_data()
    if changed:
        PricesCoordinatesData.reset_data_with_new_sample_limit(current_size)
    ok = True
    ok &= len(df) >= full_size
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

def prices_coordinates_database_content_basic_check():
    full_size = 10000
    current_size = PricesCoordinatesData.get_sample_limit()
    changed = False
    if (current_size < full_size):
        PricesCoordinatesData.reset_data_with_new_sample_limit(full_size)
        changed = True
    df = PricesCoordinatesData.get_data()
    if changed:
        PricesCoordinatesData.reset_data_with_new_sample_limit(current_size)
    ok = True
    ok &= len(df) >= full_size
    ok &= not df.isnull().values.any()
    ok &= df['price'].min() > 0
    ok &= df['price'].max() < 10000000000
    ok &= type(df['date_of_transfer'].iloc[0]) is datetime.date
    ok &= df['date_of_transfer'].min() >= datetime.date(1995, 1, 1)
    ok &= df['date_of_transfer'].max() <= datetime.date(2022, 12, 31)
    ok &= set(df['property_type'].unique()).issubset(['D', 'S', 'T', 'F', 'O'])
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
    assert set((['latitude', 'longitude'])).issubset(set(price_data.columns)), \
        'cannot find location columns'
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

def one_hot_encode_column(df, column_name, possible_values = None):
    one_hot_encoded_df = pd.get_dummies(df[column_name])
    if (not (possible_values is None) and len(one_hot_encoded_df.columns) < len(possible_values)):
        for value in possible_values:
            if value not in one_hot_encoded_df.columns:
                one_hot_encoded_df[value] = False
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

def encode_pure_price_data(price_data = PricesCoordinatesData.get_data()):
    assert set(['price', 'date_of_transfer', 'property_type', 'latitude', 'longitude']).issubset(set(price_data.columns))
    price_data = price_data.copy()
    price_data = one_hot_encode_column(price_data, 'property_type', ['D', 'S', 'T', 'F', 'O'])
    price_data = date_to_days_encode_column(price_data, 'date_of_transfer')
    return price_data

def general_PCA_plot(df = encode_pure_price_data()):
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

def general_PCA_plot_with_one_column_colorcoded(df = encode_pure_price_data(), colorcoded_column_name = 'price'):
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

def get_entire_gdf():
    return ox.geocoder.geocode_to_gdf("United Kingdom")

def get_basemap_from_bbox(bounding_box):
    try:
        return ox.graph_from_bbox(north=bounding_box['north'], 
                                  south=bounding_box['south'], 
                                  west=bounding_box['west'], 
                                  east=bounding_box['east'])
    except:
        print(f"Error from getting graph from bbox {bounding_box}!")

def bounding_box_to_geometry(bounding_box):
    minx = bounding_box["west"]
    maxx = bounding_box["east"]
    miny = bounding_box["south"]
    maxy = bounding_box["north"]
    return geopandas.GeoDataFrame(geometry=[shapely.geometry.Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)])])

def plot_general_house_distribution(price_data = PricesCoordinatesData.get_data()):
    gdf = get_entire_gdf()
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, color=(0.95, 0.96, 0.95), edgecolor='black', alpha=0.7)
    housing_gdf = geopandas.GeoDataFrame(
        geometry=geopandas.points_from_xy(price_data['longitude'], price_data['latitude'])
    )
    housing_gdf.plot(ax=ax, color='red', marker='o', alpha=0.7, markersize=0.3)
    ax.set_aspect('equal', adjustable='datalim')
    bbox = find_bounding_box_for_price_data(price_data)
    bbox = pad_bounding_box(bbox, 0.01)
    plt.xlim((bbox["west"], bbox["east"]))
    plt.ylim((bbox["south"], bbox["north"]))
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('House Distribution From Sampled Data')
    plt.show()

def plot_general_house_density_heatmap(price_data = PricesCoordinatesData.get_data()):
    bbox = find_bounding_box_for_price_data(price_data)
    bbox = pad_bounding_box(bbox, 0.01)
    bbox_gdf = bounding_box_to_geometry(bbox)
    fig_width = 10
    fig_height = fig_width / ((bbox["east"] - bbox["west"]) / (bbox["north"] - bbox["south"]))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    gdf = get_entire_gdf()
    ax = sns.kdeplot(data=price_data, x='longitude', y='latitude', fill=True, cmap='Reds', thresh=0, levels=100)
    cbar = ax.figure.colorbar(ax.collections[0])
    cbar.set_label('Density level')
    inverse_mask = geopandas.overlay(bbox_gdf, gdf, how='difference')
    inverse_mask.plot(ax=ax, facecolor='blue')
    ax.set_aspect('equal', adjustable='datalim')
    plt.title('Density Heatmap of Sampled Data with Density Colorbar')
    print(bbox)
    plt.xlim((bbox["west"], bbox["east"]))
    plt.ylim((bbox["south"], bbox["north"]))
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

def plot_average_house_price_against_year(price_data = PricesCoordinatesData.get_data()):
    price_data['year'] = price_data['date_of_transfer'].apply(lambda x: x.year)
    average_prices = price_data.groupby('year')['price'].mean()
    plt.figure(figsize=(10, 6))
    average_prices.plot(marker='o', linestyle='-', color='b')
    plt.title('Average Price of Sampled Data per Year')
    plt.xlabel('Year')
    plt.ylabel('Average Price')
    plt.xlim([1995, 2021])
    plt.grid(True)
    plt.show()