#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 22:59:25 2025

@description:
    Performing feature selection using Lasso

@author: xenon
"""
import numpy as np
from feature_selection_one import product_df as df
import pandas as pd
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# importing original train data to get the SalePrice
file_path = "../../data/house-prices/train.csv"

original_train_df = pd.read_csv(file_path)


X = df
y = original_train_df["SalePrice"]


def get_orig_name(feature_name):

    if "_" in feature_name:
        return feature_name.split("_")[0]
    return feature_name


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# cross validation to select best alpha
lass_cv = LassoCV(random_state=43, max_iter=10000, cv=5)
lass_cv .fit(X_scaled, y)

best_alpha = lass_cv .alpha_
best_coef_cv_sum = np.sum(lass_cv .coef_ != 0)

# Lasso modelling
lass_model = Lasso(alpha=best_alpha, random_state=43, max_iter=10000)

lass_model.fit(X_scaled, y)

best_coef_sum = np.sum(lass_model.coef_ != 0)

# creating coef df
coef_df = pd.DataFrame()
coef_df["col"] = X.columns
coef_df["score"] = lass_model.coef_

coef_group = coef_df.groupby(coef_df["col"].apply(get_orig_name))[
    "score"].sum()

sorted_coef = coef_group.sort_values(ascending=False)

# grading with a threshold of 2030 (75th quantile)
graded_coef = sorted_coef[sorted_coef > 2030]

# print(graded_coef.index)

# plotting the coef
plt.barh(width=graded_coef, y=graded_coef.index)
plt.title("Feature with score > 1980")
plt.ylabel("Feature")
plt.xlabel("Score")
