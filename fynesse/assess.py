from .config import *

from . import access, address

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""


def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access.data()
    raise NotImplementedError

def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError

def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError

def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError

"""
predict:
    (warning, target) = calculate(embeddings, features)
    return (warning, target)

cross_validation_calc_Rsqr()

ideal_embeddings = calc_ideal_emb()

example prediction plot of cambridge loc vs heat map of price
"""
import numpy as np, pandas as pd
import statsmodels.api as sm
from datetime import datetime
from datetime import date as date_class
from datetime import datetime as datetime_class

def predict_price(latitude, longitude, date, property_type, pp_database_conn = access.create_connection()):
    if isinstance(date, datetime_class):
        date = date.date()

    date_range = {'start': datetime(2022, 9, 1, 0, 0, 0),
                'end': datetime.now()}
    example_price_data, example_pois_list = \
        address.prepare_price_data_within_bbox_and_date_range(pp_database_conn, \
                                                              address.generate_bbox(52.206767, 0.119229), date_range)
    # TODO: get data in
    price_data = example_price_data # replace this!
    pois_list = example_pois_list

    target_array = price_data['price'].values
    feature_array = price_data.drop('price', axis=1)
    current_features = {
        'date_of_transfer': date, 
        'property_type': property_type, 
        'latitude': latitude, 
        'longitude':longitude,
    }
    for (k, v) in pois_list:
        col_name = address.column_name_of_tag((k, v))
        current_features[col_name] = address.extract_closest_euclidean_dist_from_pois(pois_list, (k, v), latitude, longitude)
    feature_array = pd.concat([feature_array, pd.DataFrame([current_features])], ignore_index=True)
    feature_array = feature_array.values

    design_matrix = np.ones((feature_array.shape[0], 1))
    for column_index in range(feature_array.shape[1]):
        column_array = feature_array[:, column_index]
        if (type(column_array[0]) is str):
            unique_values = np.unique(column_array)
            for value in unique_values:
                new_column = np.where(column_array == value, 1, 0)
                design_matrix = np.column_stack((design_matrix, new_column))
        elif isinstance(column_array[0], date_class):
            design_matrix = np.column_stack((design_matrix, np.array([d.days for d in column_array - date_class(2000, 1, 1)])))
        elif (type(column_array[0]) is float):
            design_matrix = np.column_stack((design_matrix, column_array))
        else:
            raise TypeError
    design_matrix = np.asarray(design_matrix, dtype=np.float64)

    training_design_matrix = design_matrix[:-1]
    predicting_design_matrix = np.array(design_matrix[-1])

    predict_model = sm.OLS(target_array, training_design_matrix)
    result = predict_model.fit()
    optimal_params = result.params

    print(result.summary())

    return predict_model.predict(optimal_params, predicting_design_matrix)