#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 22:06:15 2025

@description:
    Performing feature selection using mutual_info_regression

@author: xenon
"""

from feature_selection_one import product_df as df
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import seaborn as sns
import matplotlib.pyplot as plt

# importing original train data to get the SalePrice
file_path = "../../data/house-prices/train.csv"

original_train_df = pd.read_csv(file_path)


X = df
y = original_train_df["SalePrice"]

info = mutual_info_regression(X, y, random_state=43)

score_df = pd.DataFrame()
score_df["col"] = X.columns
score_df["rel"] = info


def get_orig_name(feature_name):

    if "_" in feature_name:
        return feature_name.split("_")[0]
    return feature_name


score_group = score_df.groupby(
    score_df["col"].apply(get_orig_name))["rel"].sum()


score_group_sorted = score_group.sort_values(ascending=False)

# graded using 0.2 as threshold
graded_score = score_group_sorted[score_group_sorted > 0.2]

# print(graded_score.index)

# plotting the score
plt.xlim(xmin=None, xmax=0.6)
plt.barh(width=graded_score, y=graded_score.index)
plt.title("Feature with score > 0.2")
plt.ylabel("Feature")
plt.xlabel("Score")
