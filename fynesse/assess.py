from .config import *

from . import access

import osmnx as ox, pandas as pd
from datetime import datetime
default_tag_list = [{"amenity": 'school'},
                    {"amenity": 'hospital'},
                    {"amenity": 'library'},
                    {"amenity": 'restaurant'},
                    {"public_transport": True},
                    {"shop": True},
                    {"leisure": True}]

def prices_coordinates_database_content_check(conn):
    query = f'SELECT price, date_of_transfer, property_type,\
              latitude, longitude FROM prices_coordinates_data'
    df = pd.read_sql_query(query, conn)

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

# currently unused, might be useful for testing
def get_closest_pois_list(tag_list, feature_list, latitude, longitude, start_deg = 0.002, end_deg = 0.5, increase_factor = 5):
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

# currently unused, might be useful for testing
def distance_extraction_from_closest(latitude, longitude, tag_list = default_tag_list):
    feature_list = ["geometry"]
    pois_list = get_closest_pois_list(tag_list, feature_list, latitude, longitude)
    distance_list = extract_closest_euclidean_dist_from_pois(pois_list, tag_list, latitude, longitude)
    return list(distance_list.values())

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
    
def prepare_price_data_within_bbox_and_date_range(bounding_box, date_range, conn, \
                                                  padding_deg = 0.1, tag_list = default_tag_list, \
                                                  ):
    pois_bounding_box = {'south': bounding_box['south'] - padding_deg,
                         'north': bounding_box['north'] + padding_deg,
                         'west': bounding_box['west'] - padding_deg,
                         'east': bounding_box['east'] + padding_deg}
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