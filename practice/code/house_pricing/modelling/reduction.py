#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 21:15:38 2025

@author: xenon
"""

from sklearn.decomposition import PCA
from base import union_df, intersect_df, y_train, y_true, create_submission
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


sep = "<==---==>"


def decompose(X):
    scaler = StandardScaler()

    scaled_X = scaler.fit_transform(X)

    reducer = PCA(n_components=4)

    return reducer.fit_transform(X)


X_train = decompose(intersect_df[0])
X_test = decompose(intersect_df[1])


# model = LinearRegression()

model = RandomForestRegressor(max_depth=5)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


r_squared = r2_score(y_true, y_pred)

print("TEST R-SQUARED SCORE")
print(r_squared)


# create_submission(y_pred, "reduction_intersect")
