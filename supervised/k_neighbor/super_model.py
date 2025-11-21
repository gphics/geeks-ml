#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 17:46:49 2025

@author: xenon
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import random
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

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


# best_hyperparameters = {
#     "tree": [
#         {'criterion': 'gini', 'max_depth': np.int64(
#             25), 'min_samples_split': np.int64(25), 'splitter': 'random'},
#         0.8204381394764922
#     ],
#     "k_neighbors": [
#         {'algorithm': 'auto', 'n_neighbors': np.int64(
#             51), 'weights': 'distance'},
#         0.652099679869437
#     ]
# }


best_hyperparameters = {
    "tree": [
        {'criterion': 'gini', 'max_depth': np.int64(7), 'min_samples_split': np.int64(
            5), 'random_state': np.int64(10), 'splitter': 'random'},
        0.8260749482141737
    ],
    "k_neighbors": [
        {'algorithm': 'auto', 'n_neighbors': np.int64(
            51), 'weights': 'distance'},
        0.652099679869437
    ],
    "forest": [
        {'criterion': 'log_loss', 'n_estimators': 10000},
        0.8440336450944699
    ]
}

# =====>>>>>
# Model testing

# First model
# model = DecisionTreeClassifier()

# param_grid = {
#     "criterion": ["gini", "log_loss", "entropy"],
#     "splitter": ["best", "random"],
#     "random_state":np.arange(10,50,5),
#     "max_depth": np.arange(3, 30, 2),
#     "min_samples_split": np.arange(3, 30, 2)
# }

# Second model
# model = KNeighborsClassifier()

# param_grid = {
#     "n_neighbors": np.arange(3, 100, 3),
#     "weights": ["uniform", 'distance'],
#     "algorithm":['auto', 'ball_tree', 'kd_tree', 'brute']
# }

# Third model
# model = RandomForestClassifier()
# param_grid = {
#     "criterion":["gini", "entropy", "log_loss"],
#     "n_estimators":[1000, 15000]
#     }


# Cross Validation
# grid = GridSearchCV(estimator=model, param_grid=param_grid)

# model training
# grid.fit(X, y)

# print(grid.best_params_)
# print(grid.best_score_)

# =====>>>>>


def generate_model(name="tree"):
    # getting hyperparameters gotten from cross validations
    tree_params = best_hyperparameters["tree"][0]
    forest_params = best_hyperparameters["forest"][0]
    neighbors_params = best_hyperparameters["k_neighbors"][0]

    # instantiating the model
    model = {}
    if name == "tree":
        model = DecisionTreeClassifier(**tree_params)
    elif name == "forest":
        model = RandomForestClassifier(**forest_params)
    else:
        model = KNeighborsClassifier(**neighbors_params)
    model.fit(X, y)
    return model


tree_model = generate_model()
forest_model = generate_model("forest")
k_model = generate_model("neighbors")

tree_pred = tree_model.predict(encoded_df1)
k_pred = k_model.predict(encoded_df1)
forest_pred = forest_model.predict(encoded_df1)

# print(tree_model.score(X, y))
# print(tree_model.score(encoded_df1, df3["Survived"].values))

print("<<====>>")
predicted_df = pd.DataFrame()
predicted_df["PassengerId"] = encoded_df1["PassengerId"]
predicted_df["tree_pred"] = tree_pred
predicted_df["neighbor_pred"] = k_pred
predicted_df["forest_pred"] = forest_pred
# computation
predicted_df["super_pred"] = predicted_df[["tree_pred", "forest_pred",
                                           "neighbor_pred"]].mode(axis=1).iloc[:, 0].astype(int)
predicted_df["true_pred"] = df3["Survived"].values


def get_performance(y_pred):
    y_true = predicted_df["true_pred"]
    performance = accuracy_score(y_true, y_pred)
    return performance


tree_performance = get_performance(predicted_df["tree_pred"])

neighbor_performance = get_performance(predicted_df["neighbor_pred"])

forest_performance = get_performance(predicted_df["forest_pred"])

super_performance = get_performance(predicted_df["super_pred"])


print("TREE:", tree_performance)
print("NEIGHBOR:", neighbor_performance)
print("FOREST:", forest_performance)
print("SUPER:", super_performance)


def csv_export(name="super_pred"):
    new_df = predicted_df[["PassengerId", name]]
    new_df.columns = ["PassengerId", "Survived"]
    new_df.to_csv(f"../../data/titanic/{name}.csv", index=False)


# csv_export("tree_pred")
# csv_export("forest_pred")
# csv_export("neighbor_pred")
# csv_export()
