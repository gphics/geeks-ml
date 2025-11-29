#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 22:38:45 2025

@author: xenon
"""


import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

np.random.seed(88)
file_path = "../data/wine/winequality-red.csv"

df = pd.read_csv(file_path, sep=";")

# data segregation
X = df.drop("quality", axis=1)
y = df["quality"]


# Feature selection
info_gain = mutual_info_classif(X, y)
info_rel = mutual_info_regression(X, y)

# result datafram creation
res_df = pd.DataFrame()
res_df["cols"] = X.columns
res_df["gain"] = info_gain
res_df["rel"] = info_rel
res_df.reset_index(drop=True)

print(res_df.sort_values(by="gain", ascending=False))
