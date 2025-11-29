#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 22:22:10 2025

@author: xenon
"""

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(23)

X, y_true = make_blobs(n_samples=300, n_features=2, centers=3)


k_list = np.arange(1, 10)
inertias = []

for k in k_list:
    model = KMeans(n_clusters=k)
    model.fit(X)
    inertias.append(model.inertia_)

# sns.set_style("ticks")
# sns.lineplot(x=k_list, y=inertias, marker="*")

plt.plot(k_list, inertias, 'bx-')

# model = KMeans(n_clusters=3)

# model.fit(X)


# y_pred = model.predict(X)

# # getting variables
# centroids =model.cluster_centers_
# labels = model.labels_
# inertia = model.inertia_

# # plotting the data distribution
# plt.grid(True)
# sns.scatterplot(x=X[:, 0], y=X[:, 1], c=y_pred, cmap="viridis")

# # Plotting the centroids
# for i in centroids:
#     sns.scatterplot(x = [i[0]], y = [i[1]], markers="*", c="red")
