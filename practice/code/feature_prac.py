#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 23:05:34 2025

@author: xenon
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


data = {
    'Age': [30, 45, 22, 50, 35],
    'Profession': ['Dancer', 'Lawyer', 'Dancer', 'Lawyer', 'Artist'],
    'Region': ['North', 'South', 'North', 'West', 'South'],
    # This column should be deemed irrelevant
    'Irrelevant_Feature': ['A', 'B', 'A', 'C', 'B'],
    'Is_High_Value': [1, 0, 1, 0, 0]  # Target variable (0 or 1)
}

df = pd.DataFrame(data)

# Data segregation
X = df.drop("Is_High_Value", axis=1)
y = df["Is_High_Value"]

# OHE
X_encoded = pd.get_dummies(X, dtype="int")
# print(X_encoded)

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.3)


# Modelling
model = RandomForestClassifier()
model.fit(X_train, y_train)

res = pd.DataFrame()
res["column"] = X_train.columns
res["score"] = model.feature_importances_

# print(res)


def get_orig_name(feature_name):

    if "_" in feature_name:
        return feature_name.split("_")[0]

    return feature_name


sec_res = res.groupby(res["column"].apply(get_orig_name))["score"].sum()
# print(sec_res)

test_ohe = pd.get_dummies(df["Profession"], drop_first=True, dtype="int")

print(test_ohe)
