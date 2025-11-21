#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 22:47:24 2025

@author: xenon
"""


import pandas as pd
from sklearn.svm import SVC
import random
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import accuracy_score, classification_report


# file paths


# test data file_path
file_path_a = "../../data/titanic/test.csv"
# train data file_path
file_path_b = "../../data/titanic/train.csv"
# prescribed prediction
file_path_c = "../../data/titanic/gender_submission.csv"

# loading data
df1 = pd.read_csv(file_path_a)
df2 = pd.read_csv(file_path_b)
df3 = pd.read_csv(file_path_c)

# filling mixing value


def cabin_handler(df):
    unique = df["Cabin"].unique()
    unique = unique[1:]
    # print(unique)
    # randomly filling cabin
    # if (df["Cabin"] == "nan").any():
    #     df["Cabin"] = random.choice(unique)
    df["Cabin"] = df["Cabin"].replace({"nan": random.choice(unique)})
    # print(df["Cabin"].value_counts())


cabin_handler(df1)
cabin_handler(df2)

df1["Fare"] = df1["Fare"].fillna(df1["Fare"].mean())
df1["Age"] = df1["Age"].fillna(df1["Age"].mean())
df2["Age"] = df2["Age"].fillna(df2["Age"].mean())


# feature encoder function


def encoder(df):
    # df = df.drop("Cabin", axis= 1)
    enc = LabelEncoder()
    cat_data = df.select_dtypes(include="object").apply(enc.fit_transform)
    num_data = df.select_dtypes(exclude="object")

    final_df = num_data.join(cat_data)
    return final_df


# Encoding training data
encoded_df2 = encoder(df2)
X = encoded_df2.drop("Survived", axis=1)
y = encoded_df2["Survived"]

# Encoding testing data
encoded_df1 = encoder(df1)

# Modelling

model = SVC(kernel="linear")

model.fit(X, y)

y_pred = model.predict(encoded_df1)

y_true = df3["Survived"]

# acc = accuracy_score(y_true, y_pred)
# cr = classification_report(y_true, y_pred)


predicted_df = pd.DataFrame()

predicted_df["PassengerId"] = encoded_df1["PassengerId"]
predicted_df["Survived"] = y_pred

predicted_df.to_csv("../../data/titanic/svc.csv", index=False)
