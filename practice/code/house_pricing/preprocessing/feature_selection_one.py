#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 18:02:05 2025

@author: xenon

@step: 5a

@description:
    This module is responsible for feature selection using mutual_info_regression from sklearn.
    
    

"""

from data_encoding import X_train, X_test
from base import y_train
from sklearn.feature_selection import mutual_info_regression
import pandas as pd
import numpy as np

# setting seed for reproducibility
np.random.seed(39)

# calculating the X vs y strength
strength = mutual_info_regression(X_train, y_train)

# computing the result dataframe
res = pd.DataFrame()
res["col"] = X_train.columns
res["strength"] = strength

# Computing column name


def get_orig_name(col_name):
    if "_" in col_name:
        return col_name.split("_")[0]
    return col_name


# getting final result
grouped = res.groupby(res["col"].apply(get_orig_name))["strength"].sum()

# setting threshold base on grouped stat (describe())
threshold = 0.05

selected_cols = grouped[grouped > threshold].index.to_list()
