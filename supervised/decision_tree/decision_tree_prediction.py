#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 16:54:50 2025

@author: xenon
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt

file_path = "../../data/Experience-Salary.csv"


df = pd.read_csv(file_path)

# removing rows with salary less than 1
condition = df["Salary"] < 1
df = df[~condition]

print(df.describe())


# data segregation
X = df["Years of Experience"].values.reshape(-1, 1)
y = df["Salary"].values

# data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


def create_model(alg="decision"):
    model = {}
    if alg == "decision":
        model = DecisionTreeRegressor(max_depth=4, min_samples_split=100)
    else:
        model = LinearRegression()

    model.fit(X_train, y_train)

    return model


def predict(model):
    y_pred = model.predict(X_test)
    return y_pred


def get_performance(y_pred):
    r_squared = r2_score(y_test, y_pred)
    return r_squared


# Instantiating model
tree_reg = create_model()
linear_reg = create_model("linear")

# predicting values ....
tree_pred = predict(tree_reg)
linear_pred = predict(linear_reg)

# performance
tree_performance = get_performance(tree_pred)
linear_performance = get_performance(linear_pred)

print(tree_performance)
print(linear_performance)


# plotting the data


sns.scatterplot(x=X_train.flatten(), y=y_train.flatten(), alpha=0.5)

sns.regplot(x=X_test.ravel(), y=tree_pred.ravel(),
            color="orange", scatter_kws={"alpha": 0.5})
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
