#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 23:52:39 2025

Classifying wine quality using DecisionTreeClassifier and LogisticRegression
@author: xenon
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# importing data
file_path = "../../data/wine/winequality-red.csv"

df = pd.read_csv(file_path, sep=";")
df["quality"] = df["quality"].astype("category")

# data segregation
X = df.drop("quality", axis=1)
y = df["quality"]

# data splitting
X_train, X_test, y_train, y_true = train_test_split(X, y, test_size=0.3)

# Creating model
tree_model = DecisionTreeClassifier(
    criterion="log_loss", max_depth=4, min_samples_leaf=15)

logit_model = LogisticRegression(max_iter=10000)
# fitting model
tree_model.fit(X_train, y_train)
logit_model.fit(X_train, y_train)
# predicting quality
tree_y_pred = tree_model.predict(X_test)
logit_y_pred = logit_model.predict(X_test)


# getting performance

tree_acc = accuracy_score(y_true, tree_y_pred)
tree_cr = classification_report(y_true, tree_y_pred, zero_division=0)

logit_acc = accuracy_score(y_true, logit_y_pred)
logit_cr = classification_report(y_true, logit_y_pred, zero_division=0)


print("Tree")
print(tree_acc)
print(tree_cr)

print("logit")
print(logit_acc)
print(logit_cr)



