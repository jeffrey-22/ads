from .config import *

from . import access, address

import numpy as np, pandas as pd
import statsmodels.api as sm
from datetime import datetime, timedelta
from datetime import date as date_class
from datetime import datetime as datetime_class

def generate_suitable_bbox(latitude, longitude, default_bbox_size = 0.05):
    status_code = 0
    bbox = address.generate_bbox(latitude, longitude, default_bbox_size)
    return status_code, bbox

def generate_suitable_date_range(pred_date, tolerable_days_exceeding_the_bounds = 500, default_range_size = 400):
    latest_date_in_dateset = date_class(2022, 12, 31)
    earliest_date_in_dataset = date_class(1995, 1, 1)

    status_code = 0

    target_date = pred_date
    if (pred_date > latest_date_in_dateset + timedelta(days=tolerable_days_exceeding_the_bounds)):
        status_code = 1
        target_date = latest_date_in_dateset
    
    if (pred_date < earliest_date_in_dataset - timedelta(days=tolerable_days_exceeding_the_bounds)):
        status_code = 1
        target_date = earliest_date_in_dataset
    
    left_bound = target_date - timedelta(days=default_range_size / 2)
    right_bound = target_date + timedelta(default_range_size / 2)

    if (left_bound < earliest_date_in_dataset):
        delta = earliest_date_in_dataset - left_bound
        left_bound += delta
        right_bound += delta
    if (right_bound > latest_date_in_dateset):
        delta = right_bound - latest_date_in_dateset
        left_bound -= delta
        right_bound -= delta

    date_range = {'start': left_bound,
                  'end': right_bound}
    
    return status_code, date_range

def process_feature_array_into_design_matrix(feature_array):
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
            design_matrix = np.column_stack((design_matrix, np.array([d.days for d in column_array - date_class(1995, 1, 1)])))
        elif (type(column_array[0]) is float):
            design_matrix = np.column_stack((design_matrix, column_array))
        else:
            raise TypeError
    design_matrix = np.asarray(design_matrix, dtype=np.float64)

    return design_matrix

def prepare_feature_array_and_target_array(price_data, pois_list, latitude, longitude, date, property_type):
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
    return feature_array, target_array

def validate_model(validation_level, result, model, feature_array, design_matrix, target_array, warning):
    if validation_level >= 0:
        print(f"==== Validation of current model, level {validation_level} ====")
    if validation_level >= 1:
        if warning == 1:
            print(f"A warning is issued - this signals the results are poor (and this is expected)")
        else:
            print("No warning issued")
    if validation_level >= 3:
        print("""
Details of each validation level:
Level | Message
  0   | No validation
  1   | Warnings
  2   | Summary
  3   | Help message
  4   | PCA of feature array
  5   | Stratified Cross-Validation on training dataset
              """)
    if validation_level >= 2:
        print(result.summary())
    if validation_level >= 4:
        # PCA
        pass
    if validation_level >= 5:
        # CV
        pass
    if validation_level >= 0:
        print(f"==== End of Validation ====")

def predict_price(latitude, longitude, date, property_type, pp_database_conn,\
                  validation_level = 2, default_bbox_size = 0.04, tolerable_days_exceeding_the_bounds = 500,\
                  default_range_size = 400,\
                  ):
    """
    Usage: TODO
    """
    if isinstance(date, datetime_class):
        date = date.date()
    status_code, training_bbox = generate_suitable_bbox(latitude, longitude, default_bbox_size)
    warning = status_code
    status_code, training_date_range = generate_suitable_date_range(date, tolerable_days_exceeding_the_bounds, default_range_size)
    warning |= status_code
    price_data, pois_list = \
        address.prepare_price_data_within_bbox_and_date_range(training_bbox, \
                                                              training_date_range, \
                                                              conn=pp_database_conn)
    feature_array, target_array = prepare_feature_array_and_target_array(price_data, pois_list, \
                                                                         latitude, longitude, date, property_type)
    design_matrix = process_feature_array_into_design_matrix(feature_array)

    training_design_matrix = design_matrix[:-1]
    predicting_design_matrix = np.array(design_matrix[-1])
    predict_model = sm.GLM(target_array, training_design_matrix, \
                           family=sm.families.Gaussian(sm.genmod.families.links.Identity()))
    result = predict_model.fit()
    optimal_params = result.params
    predict_result = predict_model.predict(optimal_params, predicting_design_matrix)

    validate_model(validation_level, result, predict_model, feature_array, training_design_matrix, target_array, warning)

    return predict_result

# plan: plot a heat map corresponding to the price in cambridge
def prediction_examples():
    pass