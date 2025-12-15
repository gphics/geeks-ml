#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 14:43:37 2025

@author: xenon

@step: 1

@description:
    This is a file holding all the original dataset. It also holds basic util functions

"""

import pandas as pd

# GETTING FILE PATH

train_file_path = "../../../data/house-prices/train.csv"

test_file_path = "../../../data/house-prices/test.csv"

submission_file_path = "../../../data/house-prices/sample_submission.csv"

# IMPORTING DATASET
train_df = pd.read_csv(train_file_path)

test_df = pd.read_csv(test_file_path)

submission_df = pd.read_csv(submission_file_path)

target = "SalePrice"


# Utility functions
def split_df(df):
    """
    This function helps to split train data into X and y
    """
    X = df.drop(target, axis=1)
    y = df[target]

    return [X, y]


def get_missing(df, method="rel"):
    """
    This function helps to calculate relative or absolute missing values in a dataset.

    @params:
        df: pandas dataframe
        method: rel | abs
    """
    res = None
    if method == "rel":
        res = df.isnull().mean().sort_values(ascending=False)
    else:
        res = df.isnull().sum().sort_values(ascending=False)
    condition = res > 0
    return res[condition]


# EXPORT PPTIES
X_train, y_train = split_df(train_df)

X_test, y_test = test_df, submission_df[target]
