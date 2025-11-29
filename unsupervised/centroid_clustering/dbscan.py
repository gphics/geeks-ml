#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 22:11:16 2025

@author: xenon
"""

from breast_cancer import X, y_true, accuracy_score
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score

db_model = DBSCAN(min_samples=20, eps=0.6)
db_model.fit(X)

y_pred = db_model.labels_


db_acc = accuracy_score(y_true, y_pred)

print(db_acc)
