#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 16:24:02 2025

@author: xenon
"""

from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=30)

model = AgglomerativeClustering(n_clusters=3, linkage="average")

y_pred = model.fit_predict(X)


sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="deep")
