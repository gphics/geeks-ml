#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 22:19:25 2025

@author: xenon
"""

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from helper import Helper

model_helper = Helper("linear regression")


# model = LinearRegression()
model = DecisionTreeRegressor()

# making predictions
_, intersect_metrics = model_helper.make_predictions(
    model, model_helper.df_a, model_helper.df_c)

# _, union_metrics = model_helper.make_predictions(
#     model, model_helper.df_b, model_helper.df_d)

print("METRICS FOR TESTING DATA")
print(intersect_metrics)

# print(union_metrics)
