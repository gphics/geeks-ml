#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 21:16:56 2025

@author: xenon
"""

import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

file_path = "../../data/breast_cancer.csv"


df = pd.read_csv(file_path)

# binarizing diagnosis
df["dx"] = df["diagnosis"].map({"M": 1, "B": 0})
# print(df.info())

cols = ["fractal_dimension_worst", "symmetry_worst", "concave_points_worst",
        "texture_se", "radius_mean", "texture_mean", "texture_se", "fractal_dimension_mean"]

subset = df[cols]

# corr_df = subset.corr()
# second = ["radius_mean","concave_points_worst"]


first = ["fractal_dimension_mean", "fractal_dimension_worst", "dx"]

first_df = df[first]
X = first_df.drop("dx", axis=1)
y_true = df["dx"]

# sns.scatterplot(x =first_df.iloc[:, 0], y=first_df.iloc[:, 1])

# Getting the best k
# K = np.arange(1,5,1)
# inertias = []

# for i in K:
#     model = KMeans(n_clusters=i)
#     model.fit(X)
#     inertias.append(model.inertia_)

# plt.plot(K, inertias, "bx-")
# plt.xlabel("N-clusters")
# plt.ylabel("Inertia")
# plt.title("k vs Inertia")

# The best k = 2


# Model creation

model = KMeans(n_clusters=2)
model.fit(X)

y_pred = model.predict(X)


acc = accuracy_score(y_true, y_pred)
# print(acc)
