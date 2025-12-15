#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 21:55:00 2025

@description:
    This module contains all needed dataframes and helper functions that would be needed for the modelling operations.

@author: xenon
"""

import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score

"""
ENCODING:
    # Training data
    A --> intersect_train
    B --> union_train
    
    # Testing data
    C --> intersect_test
    D --> union_test
"""

# getting file path

file_path_a = "../../../data/house-prices/transformed/intersect_train_df.csv"

file_path_b = "../../../data/house-prices/transformed/union_train_df.csv"

file_path_c = "../../../data/house-prices/transformed/intersect_test_df.csv"

file_path_d = "../../../data/house-prices/transformed/union_test_df.csv"

# getting the y_true from submission file
submission_file_path = "../../../data/house-prices/sample_submission.csv"

submission_df = pd.read_csv(submission_file_path)

# subsetting SalePrice
y_true = submission_df["SalePrice"]

# importing train data
df_a = pd.read_csv(file_path_a)
df_b = pd.read_csv(file_path_b)

# importing test data
df_c = pd.read_csv(file_path_c)
df_d = pd.read_csv(file_path_d)


# Creating the export class
class Helper:
    """
    @description:
        This class is used for exporting all helpers in this module
    """
    df_a = df_a
    df_b = df_b
    df_c = df_c
    df_d = df_d

    def __init__(self, model_type):
        self.model_type = model_type

    def calc_metrics(self, y_true, y_pred):
        """
        @params:
            y_true: True values
            y_pred: Predicted values

        @description:
            This function calculate r-squared and RMSE

        @returns:
            {r_squared:float, rmse:float}
        """

        r_squared = r2_score(y_true, y_pred)

        rmse = root_mean_squared_error(y_true, y_pred)

        result = {

            "r_squared": r_squared,
            "rmse": rmse

        }

        return result

    def split(self, df):
        """
        @params:
            df: The dataframe to subset from

        @description:
            This function is only useful for train data to separate X and y

        @returns:
            X : Independent variables
            y: Dependent variables
        """
        target = "SalePrice"

        X = df.drop(target, axis=1)
        y = df[target]

        return [X, y]

    def make_predictions(self, model, train_df, test_df):
        """
        @params:
            model: The regression model to be used for predictions
            train_df
            test_df
        @description:
            This method accepts instantiated model class, fit and predict target features.

        @returns:
            [y_pred[], metrics{}]
        """

        # splitting train df
        X, y = self.split(train_df)

        # Training model
        model.fit(X, y)

        met = self.calc_metrics(y, model.predict(X))
        print("METRICS FOR TRAINING DATA")
        print(met)
        # predicting target
        y_pred = model.predict(test_df)

        metrics = self.calc_metrics(y_true, y_pred)

        return [y_pred, metrics]
