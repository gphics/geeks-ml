#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 19:38:03 2025

@author: xenon
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector, RFE
from sklearn.tree import DecisionTreeClassifier

np.random.seed(88)
file_path = "../data/wine/winequality-red.csv"

df = pd.read_csv(file_path, sep=";")

# data segregation
X = df.drop("quality", axis=1)
y = df["quality"]

# Model creation
model = DecisionTreeClassifier(criterion="entropy", max_depth=5)

# Feature selection
sfs_for = SequentialFeatureSelector(model, n_features_to_select=3)
sfs_back = SequentialFeatureSelector(
    model, n_features_to_select=3, direction="backward")
rfe = RFE(model, n_features_to_select=3)

sfs_for.fit(X, y)
sfs_back.fit(X, y)
rfe.fit(X, y)


res_df = pd.DataFrame()
res_df["col"] = X.columns
res_df["for_support"] = sfs_for.support_
res_df["back_support"] = sfs_back.support_
res_df["rfe_support"] = rfe.support_
print(res_df)
