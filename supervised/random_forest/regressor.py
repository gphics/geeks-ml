#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 22:25:45 2025

@author: xenon
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, f1_score
from sklearn.model_selection import GridSearchCV


file_path = "../../data/Position_Salaries.csv"


df = pd.read_csv(file_path)

# data transformation
enc = LabelEncoder()
cat_feature = df.select_dtypes(include=["object"]).apply(enc.fit_transform)
cat_feature.columns = ["enc_position"]
df = df.join(cat_feature)


# data segregation
X = df.drop(["Position", "Salary"], axis=1).values
y = df["Salary"].values

# print(y)
# print(X)

# model creation
# model = RandomForestRegressor(oob_score=True,max_depth=41,min_samples_split=3, n_estimators=440)

model = RandomForestRegressor(
    oob_score=True, max_depth=100, min_samples_split=3, n_estimators=20, random_state=20)

# model training
model.fit(X, y)

# model prediction
y_pred = model.predict(X)
# print(y_pred)
# # # performance

r2 = r2_score(y, y_pred)
oob = model.oob_score_
print(r2, oob)

# Because the model isnt performing well, I want to tune the hyperparameter using GridSearchCV

param_grid = {
    "min_samples_split": [3, 4, 5],
    "n_estimators": [20, 50, 70],
    "max_depth": [100, 300, 500],
    "random_state": [20, 70, 96],
    "oob_score": [True, False]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X, y)
print(grid.best_params_)
print(grid.best_score_)
