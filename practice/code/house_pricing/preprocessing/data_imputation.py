#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 15:20:58 2025

@author: xenon

@step: 3

@description: 
    This module is for imputing missing values for both train and test dataset.

"""

from drop_missing_cols import X_train, X_test
from base import get_missing
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


"""
NOTE: The columns with missing data in test data is more than the one in training data so I used the union of  cols name of missing data from both test and train data 

"""


def get_missing_cols_union():
    """
    This function get the union of missing cols from test and train data
    """
    train_missing_cols = set(get_missing(X_train).index.to_list())
    test_missing_cols = set(get_missing(X_test).index.to_list())

    union_missing_cols = train_missing_cols | test_missing_cols

    return list(union_missing_cols)


missing_cols = get_missing_cols_union()

missing_num_cols = X_train[missing_cols].select_dtypes(
    exclude="O").columns.to_list()

missing_cat_cols = X_train[missing_cols].select_dtypes(
    include="O").columns.to_list()


transformers = [
    ("num", SimpleImputer(strategy="median"), missing_num_cols),
    ("cat", SimpleImputer(strategy="most_frequent"), missing_cat_cols)

]

ct = ColumnTransformer(transformers, remainder="passthrough",
                       verbose_feature_names_out=False)

ct.set_output(transform="pandas")

X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)
