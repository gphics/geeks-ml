#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 16:45:39 2025

@author: xenon
"""

import pandas as pd
import numpy as np
from skfuzzy.cluster import cmeans
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

file_path = "../../data/wine/winequality-red.csv"

df = pd.read_csv(file_path, sep=";")

# print(df.info())

# feature selection

# num = df.drop(["quality"], axis=1).corr()

# sns.heatmap(num, annot=True)

# Base on research alcohol and volatile acidity was selected

features = ["volatile acidity", "alcohol", "quality"]

subset_df = df[features]


def plot_df(y, title="Original Data"):
    sns.scatterplot(data=subset_df, x="alcohol",
                    y="volatile acidity", hue=y, palette="deep")
    plt.title(title)


def get_performance(y_pred, y_true):
    sil_score = silhouette_score(X, y_pred)
    adj = adjusted_rand_score(y_true, y_pred)

    return [sil_score, adj]


X = subset_df.drop("quality", axis=1)
y_true = subset_df["quality"]

# original plot
# plot_df(y_true)

# Model creation
#
#

# KMeans
k_model = KMeans(n_clusters=3)
k_pred = k_model.fit_predict(X)
k_performance = get_performance(k_pred, y_true)

# DBSCANS
db_model = DBSCAN(min_samples=50, eps=0.2)
db_pred = db_model.fit_predict(X)
db_performance = get_performance(db_pred, y_true)

# Agg clustering
agg_model = AgglomerativeClustering(n_clusters=3, linkage="ward")
agg_pred = agg_model.fit_predict(X)
agg_performance = get_performance(agg_pred, y_true)


# print(k_performance)
# print(db_performance)
# print(agg_performance)

# plotting data
# plot_df(k_pred, "KMeans Data")
# plot_df(db_pred, "Density Based Data")
# plot_df(agg_pred, "Agglomerative Data")


# DENDROGRAM
# fig, ax = plt.subplots()
plt.figure(figsize=(15, 10))
linked = linkage(X, method="ward")
# fig = dendrogram(linked, show_leaf_counts=True, distance_sort='descending',
#                  orientation="top", leaf_rotation=90, leaf_font_size=8, truncate_mode='lastp')

# fig.
