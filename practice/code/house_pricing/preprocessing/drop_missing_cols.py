#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 14:55:04 2025

@author: xenon

@step: 2

@description:
    This module is responsible for dropping data. I am dropping columns with missing values greater than 50%.

"""

from base import X_train, X_test, get_missing

# the percentage of missing values allowed within a col
threshold = 0.50

# getting the percentage of missing values in the train df
missing_x_train = get_missing(X_train)

# setting condition
missing_condition = missing_x_train > threshold

# getting missing cols ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType')

missing_cols = missing_x_train[missing_condition].index.to_list()

# Transforming the train and test data
X_train = X_train.drop(missing_cols, axis=1)
X_test = X_test.drop(missing_cols, axis=1)
