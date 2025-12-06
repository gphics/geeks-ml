#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 23:34:46 2025

@description:
    This module build the final dataframe for both train and test data on intersect and union

@author: xenon
"""

import numpy as np

# importing  transformed training data
from feature_selection_one import product_df as df

# importing  transformed testing data
from feature_selection_one import transformed_test_df
import pandas as pd

# importing original train data to get the SalePrice
file_path = "../../data/house-prices/train.csv"

original_train_df = pd.read_csv(file_path)

X = df

y = original_train_df["SalePrice"]

# Feature selected using Lasso
lasso_cols = set(['GrLivArea', 'Neighborhood', 'OverallQual', 'BsmtQual', 'BsmtExposure',
                  'RoofMatl', 'GarageCars', 'KitchenQual', 'ExterQual', 'GarageYrBlt',
                  'Exterior1st', 'SaleType', 'OverallCond', 'BsmtFinSF1', 'MasVnrArea',
                  'LotArea', 'Fireplaces', 'BsmtFinType1', 'LotConfig'])

# feature selected using Mutual info
mutual_cols = set(['OverallQual', 'Neighborhood', 'GarageYrBlt', 'GrLivArea',
                   'TotalBsmtSF', 'GarageArea', 'YearBuilt', 'GarageCars', 'KitchenQual',
                   'BsmtQual', 'ExterQual', '1stFlrSF', 'Foundation', 'GarageFinish',
                   'MSSubClass', 'FullBath', 'YearRemodAdd', 'Exterior2nd', 'LotFrontage',
                   'TotRmsAbvGrd'])

# performing set ops
cols_intersect = lasso_cols & mutual_cols
cols_union = lasso_cols | mutual_cols

# Adding the Id column as the selector did not select it
cols_intersect.add("Id")
cols_union.add("Id")


def build_df(cols_list, working_df=X):
    """
    @params:
        cols_list --> list of column names
    @return:
        dataframe
    """
    base_df = pd.DataFrame()
    encoded_cols = working_df.columns

    # looping through the list with original column names
    for original_col in cols_list:

        # looping through the list with encoded column names
        for col in encoded_cols:

            # if original column name in encoded
            # column name, store the column data
            # from X into base_df
            if original_col in col:
                base_df[col] = working_df[col]
    return base_df

# creating the dataframe

# FOR TRAINING DATA
# # #
# # #


# for intersecting columns
intersect_df = build_df(cols_intersect)
intersect_df["SalePrice"] = y

# for union columns
union_df = build_df(cols_union)
union_df["SalePrice"] = y

# saving to csv
union_df.to_csv(
    "../../data/house-prices/transformed/union_train_df.csv", index=False)

intersect_df.to_csv(
    "../../data/house-prices/transformed/intersect_train_df.csv", index=False)


# FOR TESTING DATA
# # #
# # #

# for intersecting columns
test_intersect_df = build_df(cols_intersect, transformed_test_df)

# for union columns
test_union_df = build_df(cols_union, transformed_test_df)

# saving to csv
test_union_df.to_csv(
    "../../data/house-prices/transformed/union_test_df.csv", index=False)

test_intersect_df.to_csv(
    "../../data/house-prices/transformed/intersect_test_df.csv", index=False)
