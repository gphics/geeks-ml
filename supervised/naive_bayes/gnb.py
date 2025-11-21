#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 20:23:24 2025

@author: xenon
"""

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns

file_path = "../../data/wine/winequality-white.csv"

df = pd.read_csv(file_path, sep=";")

# data segregation
X = df.drop("quality", axis=1).values
y = df["quality"].values

# data splitting
X_train, X_test, y_train, y_true = train_test_split(X, y, test_size=0.3)

# initiating model
model = GaussianNB()

# model training
model.fit(X_train, y_train)

# making predictions
y_pred = model.predict(X_test)


# performance
c_rep = classification_report(y_true, y_pred, zero_division=1)

print(c_rep)
