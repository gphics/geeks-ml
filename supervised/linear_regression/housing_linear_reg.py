#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 21:23:08 2025
This is a univariate linear regression exercise

@author: xenon
"""
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

file_path = "../../data/housing.csv"

df = pd.read_csv(file_path)

X = df["total_rooms"].values.reshape(-1, 1)
y = df["population"].values

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

sns.scatterplot(df, x="total_rooms", y="population", label="Data Points")
sns.regplot(x=df["total_rooms"], y=y_pred, color="red")


mse = mean_squared_error(y, y_pred)
r_squared = r2_score(y, y_pred)*100
