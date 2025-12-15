#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 18:10:26 2025

@author: xenon

@step: 5b

@description:
    This module is responsible for feature selection using Recursive feature elimination (RFE) from sklearn
    

"""


from data_encoding import X_train, X_test
from base import y_train
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

# setting seed for reproducibility
np.random.seed(39)

# setting number of features to choose
n_features = 165

# initiating the estimator model
model = RandomForestRegressor(n_estimators=50, max_depth=5)

# initiating the feature selector
rfe = RFE(estimator=model, n_features_to_select=n_features)

# traininf the selector
rfe.fit(X_train, y_train)

# building result dataframe
res = pd.DataFrame()
res["col"] = X_train.columns
res["rfe"] = rfe.support_
res["rfe"] = res["rfe"].astype(int)

# Computing column name


def get_orig_name(col_name):

    if "_" in col_name:
        return col_name.split("_")[0]
    return col_name


# getting final result
grouped = res.groupby(res["col"].apply(get_orig_name))["rfe"].sum()


# setting threshold base on grouped stat (describe())
threshold = 3.5

selected_cols = grouped[grouped >= threshold].index.to_list()
