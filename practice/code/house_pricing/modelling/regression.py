#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 20:46:52 2025

@author: xenon

@description:
    Making prediction using LinearRegression

"""

from base import union_df, intersect_df, y_train, make_prediction, get_metrics, create_submission
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


sep = "<==---==>"
# model = LinearRegression()

model = RandomForestRegressor(max_depth=5)

_, y_pred_u = make_prediction(model, *union_df)

_, y_pred_i = make_prediction(model, *intersect_df)

print("USING RANDOM FOREST REGRESSOR")
print("UNION DATASET METRICS FOR THE TEST DATA")
union_metrics = get_metrics(y_pred_u)
print(union_metrics)

print(sep)
print(sep)

print("INTERSECT DATASET METRICS FOR THE TEST DATA")
intersect_metrics = get_metrics(y_pred_i)
print(intersect_metrics)


# create_submission(y_pred_u, "union")
# create_submission(y_pred_u, "intersect")
