from .config import *

from . import assess, access

import numpy as np, pandas as pd
import statsmodels.api as sm
from datetime import datetime, timedelta
from datetime import date as date_class
from datetime import datetime as datetime_class
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import mlai.plot as plot
import osmnx as ox

model_link_function = sm.families.Poisson(sm.genmod.families.links.Log())

def generate_suitable_bbox(latitude, longitude, default_bbox_size = 0.05):
    status_code = 0
    bbox = assess.generate_bbox(latitude, longitude, default_bbox_size)
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
    feature_array = feature_array.copy()
    feature_array = assess.one_hot_encode_column(feature_array, 'property_type', ['D', 'S', 'T', 'F', 'O'])
    feature_array = assess.date_to_days_encode_column(feature_array, 'date_of_transfer')
    for col_name in assess.default_tag_col_list:
        feature_array = assess.square_root_column(feature_array, assess.column_name_of_tag(col_name))
    feature_array = feature_array.drop('latitude', axis=1)
    feature_array = feature_array.drop('longitude', axis=1)

    design_matrix = np.asarray(feature_array.values, dtype=np.float64)
    design_matrix = sm.add_constant(design_matrix, has_constant = 'add')

    return design_matrix

def prepare_feature_array_for_unseen_data(pois_list, latitude, longitude, date, property_type,\
                                          tag_list = assess.default_tag_col_list):
    current_features = {
        'date_of_transfer': date, 
        'property_type': property_type, 
        'latitude': latitude, 
        'longitude':longitude,
    }
    for (k, v) in tag_list:
        col_name = assess.column_name_of_tag((k, v))
        distance_array = assess.extract_closest_euclidean_dist_from_pois(pois_list, (k, v), latitude, longitude)
        current_features[col_name] = np.where(distance_array >= 0, np.sqrt(distance_array), distance_array)
    return current_features

def prepare_feature_array_and_target_array(price_data, pois_list, latitude, longitude, date, property_type):
    target_array = price_data['price'].values
    feature_array = price_data.drop('price', axis=1)
    current_features = prepare_feature_array_for_unseen_data(pois_list, latitude, longitude, date, property_type)
    feature_array = pd.concat([feature_array, pd.DataFrame([current_features])], ignore_index=True)
    return feature_array, target_array

def prepare_feature_array_for_changing_locations(pois_list, latitude_list, longitude_list, date, property_type,\
                                                 tag_list = assess.default_tag_col_list):
    size = len(latitude_list)
    assert size == len(longitude_list)
    current_features = {
        'date_of_transfer': np.full(size, date), 
        'property_type': np.full(size, property_type), 
        'latitude': latitude_list, 
        'longitude':longitude_list,
    }
    feature_array = pd.DataFrame.from_dict(current_features)
    for i in range(size):
        for (k, v) in tag_list:
            col_name = assess.column_name_of_tag((k, v))
            distance_array =\
                assess.extract_closest_euclidean_dist_from_pois(pois_list, (k, v), latitude_list[i], longitude_list[i])
            feature_array.at[i, col_name] = np.where(distance_array >= 0, np.sqrt(distance_array), distance_array)
    return feature_array

def validate_model(validation_level, result, model, feature_array, design_matrix, target_array, warning, pois_list,\
                   example_plot_bbox_size = 0.04, default_bbox_size = 0.05):
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
  6   | Area Prediction Plot
              """)
    if validation_level >= 2:
        print(result.summary())
    if validation_level >= 4:
        encoded_feature_array = design_matrix.copy()
        encoded_feature_array = pd.DataFrame(encoded_feature_array)
        encoded_feature_array['price'] = target_array
        assess.general_PCA_plot_with_one_column_colorcoded(encoded_feature_array)
    if validation_level >= 5:
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True)
        X = design_matrix
        y = target_array
        test_result = []
        pred_result = []
        for train_index, test_index in stratified_kfold.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model = sm.GLM(y_train, X_train, \
                           family=model_link_function)
            result = model.fit()
            optimal_params = result.params
            y_pred = model.predict(optimal_params, X_test)
            test_result = np.concatenate((test_result, y_test))
            pred_result = np.concatenate((pred_result, y_pred))
        r2 = r2_score(test_result, pred_result)
        print(f'Stratified Cross-Validation on 5 folds result R^2: {r2}')
    if validation_level >= 6:
        assert example_plot_bbox_size <= default_bbox_size
        date = feature_array.iloc[-1]['date_of_transfer']
        property_type = feature_array.iloc[-1]['property_type']
        latitude = feature_array.iloc[-1]['latitude']
        longitude = feature_array.iloc[-1]['longitude']
        bbox = assess.generate_bbox(latitude, longitude, example_plot_bbox_size, example_plot_bbox_size)
        sample_frequency = 20
        x_vals = np.linspace(bbox["west"], bbox["east"], sample_frequency)
        y_vals = np.linspace(bbox["south"], bbox["north"], sample_frequency)
        x_grid, y_grid = np.meshgrid(x_vals, y_vals)
        example_feature_array = prepare_feature_array_for_changing_locations(pois_list, y_grid.flatten(), x_grid.flatten(),\
                                                                             date, property_type)
        example_design_matrix = process_feature_array_into_design_matrix(example_feature_array)
        optimal_params = result.params
        example_target_array = model.predict(optimal_params, example_design_matrix)
        graph = ox.graph_from_bbox(bbox["north"], bbox["south"], bbox["east"], bbox["west"])
        nodes, edges = ox.graph_to_gdfs(graph)
        fig, ax = plt.subplots(figsize=plot.big_figsize)
        edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")
        ax.set_xlim([bbox["west"], bbox["east"]])
        ax.set_ylim([bbox["south"], bbox["north"]])
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.tight_layout()


        plt.contourf(x_grid, y_grid, np.reshape(example_target_array, x_grid.shape), cmap='viridis')
        plt.colorbar(label=f'Price')
        plt.title(f'Price prediction around the given area for {date} {property_type}')
    if validation_level >= 0:
        print(f"==== End of Validation ====")

def predict_price(latitude, longitude, date, property_type, pp_database_conn = access.DatabaseConnection.get_connection(),\
                  validation_level = 2, default_bbox_size = 0.05, tolerable_days_exceeding_the_bounds = 300,\
                  default_range_size = 800,\
                  ):
    """
    Usage: TODO
    Takes ~5 minutes to complete prediction
    """
    if isinstance(date, datetime_class):
        date = date.date()
    status_code, training_bbox = generate_suitable_bbox(latitude, longitude, default_bbox_size)
    warning = status_code
    status_code, training_date_range = generate_suitable_date_range(date, tolerable_days_exceeding_the_bounds, default_range_size)
    warning |= status_code
    price_data, pois_list = \
        assess.prepare_full_price_data_within_bbox_and_date_range(training_bbox, \
                                                                  training_date_range, \
                                                                  conn=pp_database_conn)
    assert len(price_data) > 0, f"Found no available data within default_bbox_size = {default_bbox_size}, try increase the size"
    warning |= len(price_data) < 100
    feature_array, target_array = prepare_feature_array_and_target_array(price_data, pois_list, \
                                                                         latitude, longitude, date, property_type)
    design_matrix = process_feature_array_into_design_matrix(feature_array)

    training_design_matrix = design_matrix[:-1]
    predicting_design_matrix = np.array(design_matrix[-1])
    predict_model = sm.GLM(target_array, training_design_matrix, \
                           family=model_link_function)
    result = predict_model.fit()
    optimal_params = result.params
    predict_result = predict_model.predict(optimal_params, predicting_design_matrix)

    validate_model(validation_level, result, predict_model, feature_array, training_design_matrix, target_array, warning,\
                   pois_list, default_bbox_size)

    return predict_result