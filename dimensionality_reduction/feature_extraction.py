#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 21:08:24 2025

@author: xenon
"""
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = {
    'Height': [170, 165, 180, 175, 160, 172, 168, 177, 162, 158],
    'Weight': [65, 59, 75, 68, 55, 70, 62, 74, 58, 54],
    'Age': [30, 25, 35, 28, 22, 32, 27, 33, 24, 21],
    'Gender': [1, 0, 1, 1, 0, 1, 0, 1, 0, 0]  # 1 = Male, 0 = Female
}
df = pd.DataFrame(data)

X = df.drop("Gender", axis=1)
y = df["Gender"]

# data standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# dimensionality reduction
reducer = PCA(n_components=2)
reduced_X = reducer.fit_transform(X_scaled)

# modelling
model = LogisticRegression()
model.fit(reduced_X, y)

y_pred = model.predict(reduced_X)


acc = accuracy_score(y, y_pred)
print(acc)
