#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 19:30:29 2025

@author: xenon

@description:
    This module serves as the store for dataframes and utility functions

"""

import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error as mse
# dataset file path
train_file_path = "../../../data/house-prices/cleaned/train.csv"
test_file_path = "../../../data/house-prices/cleaned/test.csv"
submission_file_path = "../../../data/house-prices/sample_submission.csv"

# importing dataset
train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)
submission_df = pd.read_csv(submission_file_path)


# getting y_true
y_true = submission_df["SalePrice"]

Id = submission_df["Id"]


# getting y_train and dropping it from the train df
y_train = train_df["SalePrice"]

# dropping y
train_df = train_df.drop("SalePrice", axis=1)

# getting columns selected from preprocessing


union_cols = ['BsmtFinSF1', 'FullBath', 'GarageType', 'MSSubClass', 'MasVnrArea', 'BldgType', 'GarageFinish', 'GarageYrBlt', 'LotShape', 'MSZoning', 'SaleType', 'OverallQual', 'WoodDeckSF', 'HeatingQC', 'YearRemodAdd', '1stFlrSF', 'OpenPorchSF', 'BsmtExposure', 'RoofMatl', 'KitchenQual', 'TotRmsAbvGrd', 'BsmtQual', 'Foundation', 'LotArea', 'BedroomAbvGr', 'GrLivArea',
              'Neighborhood', 'HalfBath', 'ExterQual', 'Condition2', 'Electrical', 'Exterior2nd', 'CentralAir', 'Fireplaces', 'BsmtFinType1', 'RoofStyle', 'HouseStyle', 'FireplaceQu', 'OverallCond', 'LotConfig', 'YearBuilt', 'SaleCondition', 'BsmtUnfSF', 'GarageArea', '2ndFlrSF', 'TotalBsmtSF', 'GarageCars', 'Exterior1st', 'BsmtFinType2', 'LotFrontage', 'Condition1', 'Heating']

intersect_cols = ['SaleCondition', 'RoofStyle', 'HouseStyle',
                  'LotConfig', 'Exterior1st', 'Foundation', 'Neighborhood']


def build_df(df, original_cols_list):

    transformed_cols = df.columns.to_list()

    # saving id col for later (if necessary)
    # id_col = df["Id"]

    # concatenation list storing all selected df
    res_list = []

    # selecting the features
    for original_col in original_cols_list:

        for col in transformed_cols:
            if original_col in col:
                res_list.append(df[col])

    # creating the selected features dataset
    result_df = pd.concat(res_list, axis=1)

    # returning the result
    return result_df


def make_prediction(model, X_train, X_test):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # y_pred = model.predict(X_train)

    return [model, y_pred]


def get_metrics(y_pred):
    # y_true = y_train
    mse_score = mse(y_true, y_pred)
    r_squared_score = r2_score(y_true, y_pred)

    res = {
        "mse_score": mse_score,
        "rmse": mse_score**0.5,
        "r_squared_score": r_squared_score
    }
    return res


def create_submission(y_pred, file_name):
    submission = pd.DataFrame()
    submission["Id"] = Id
    submission["SalePrice"] = y_pred

    submission.to_csv(
        f"../../../data/house-prices/submission/{file_name}.csv", index=False)

# final product


# Train data
X_train_u = build_df(train_df, union_cols)
X_train_i = build_df(train_df, intersect_cols)


# Test data
X_test_u = build_df(test_df, union_cols)
X_test_i = build_df(test_df, intersect_cols)


intersect_df = [X_train_i, X_test_i]
union_df = [X_train_u, X_test_u]
