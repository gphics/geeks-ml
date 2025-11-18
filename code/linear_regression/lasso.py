#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 22:16:26 2025

@author: xenon
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
import seaborn as sns
from sklearn.metrics import r2_score
import random

file_path = "../../data/housing.csv"

df = pd.read_csv(file_path)

df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].mean())

scaler = StandardScaler()


features = df.drop(["ocean_proximity", "longitude", "latitude"], axis=1)
corr = features.corr()

colors = random.choice(["inferno", "magma", "Blues"])

# sns.heatmap( corr, annot=True, cmap=colors)

X = features.drop("population", axis=1).values
y = features["population"].values

X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.4)

lasso = Lasso(alpha=5)

lasso.fit(X_train, y_train)

y_pred = lasso.predict(X_test)

r_squared = r2_score(y_test, y_pred)

print(r_squared)
