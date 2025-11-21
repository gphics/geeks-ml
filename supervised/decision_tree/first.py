#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 23:17:55 2025

@author: xenon
"""

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns


file_path = "../../data/scale/balance-scale.data"

df = pd.read_csv(file_path)
df.columns = ["class", "left-weight",
              "left-distance", "right-weight", "right-distance"]

# data segregation
X = df.iloc[:, 1:5].values
y = df.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)


# creating helper functions
def train_model(criterion="entropy"):
    model = DecisionTreeClassifier(
        criterion=criterion, min_samples_leaf=5, max_depth=3)

    model.fit(X_train, y_train)
    return model


def predict(model):
    y_pred = model.predict(X_test)
    return y_pred


def calc_performance(model, y_pred, do_print=True):
    cm = confusion_matrix(y_test, y_pred)

    cr = classification_report(y_test, y_pred, zero_division=0)

    acc = accuracy_score(y_test, y_pred)

    result = {

        "cm": cm,
        "cr": cr,
        "acc": acc
    }

    if do_print:
        divider = "<<--->>"
        print(divider)
        print(cm)

        print(divider)
        print(cr)

        print(divider)
        print(acc)
    return result


# criteria
gini = "gini"
entropy = "entropy"

# creating model
gini_model = train_model(gini)
entropy_model = train_model()

# predictions
gini_pred = predict(gini_model)
entropy_pred = predict(entropy_model)

# calculating performance

gini_performance = calc_performance(gini_model, gini_pred)

entropy_performance = calc_performance(entropy_model, entropy_pred)


# print(gini_performance)
# print(y_test)
