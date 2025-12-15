#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 16:57:41 2025

@author: xenon

@step: 4

@description:
    This module is responsible for encoding categorical variables.
    This module finalize all data transformation stage and lastly export to csv the transformed data.
"""

from data_imputation import X_train, X_test
from base import y_train
from sklearn.preprocessing import OneHotEncoder
from category_encoders import OrdinalEncoder
from sklearn.compose import ColumnTransformer


# getting ordinal data columns
ordinal_cat_cols = [
    "ExterQual",
    "ExterCond",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "HeatingQC",
    "KitchenQual",
    "Functional",
    "GarageQual",
    "GarageCond"
]


# setting the categorical orders

# ExterQual, ExterCond, HeatingQC, KitchenQual
first_part_qual_cat = [
    "Po", "Fa", "TA", "Gd", "Ex"
]
# BsmtQual, BsmtCond, GarageQual,GarageCond
second_part_qual_cat = ["NA"] + first_part_qual_cat

# BsmtExposure
third_part_qual_cat = [
    "NA", "No", "Mn", "Av", "Gd"
]

# Functional
fourth_part_qual_cat = [
    "Sal", "Sev", "Maj2", "Maj1", "Mod", "Min1", "Min2", "Typ"
]

# BsmtFinType1
fifth_part_qual_cat = [
    "NA", "Unf", "LwQ", "Rec", 'BLQ', "ALQ", "GLQ"
]

# creating mapping dict for ordinal data
mapping_dict = {
    "BsmtFinType1": fifth_part_qual_cat,
    "Functional": fourth_part_qual_cat,
    "BsmtExposure": third_part_qual_cat,
    "BsmtQual": second_part_qual_cat,
    "BsmtCond": second_part_qual_cat,
    "GarageQual": second_part_qual_cat,
    "GarageCond": second_part_qual_cat,
    "ExterQual": first_part_qual_cat,
    "ExterCond": first_part_qual_cat,
    "HeatingQC": first_part_qual_cat,
    "KitchenQual": first_part_qual_cat,
}

# transforming mapping dict to a format acceptable by the OrdinalEncoder from category_encoders

mapping_list = []

for key, values in mapping_dict.items():
    data = {"col": key, "mapping": {}}

    # getting the index of values contents
    for index, value in enumerate(values):
        data["mapping"][value] = index

    # exporting the data
    mapping_list.append(data)


# getting nominal data columns
nominal_cat_cols = X_train.select_dtypes("O").drop(
    ordinal_cat_cols, axis=1).columns.to_list()


# Encoding values

# OneHotEncoding
ohe_encoding = OneHotEncoder(drop="first", sparse_output=False)

# OrdinalEncodeing
ord_encoding = OrdinalEncoder(mapping=mapping_list)

# pipeline steps
transformers = [
    ("nominal", ohe_encoding, nominal_cat_cols),
    ("ordinal", ord_encoding, ordinal_cat_cols)

]

# Pipeline
ct = ColumnTransformer(
    transformers, verbose_feature_names_out=False, remainder="passthrough")

# setting sklearn output
ct.set_output(transform="pandas")

# transforming data
X_train = ct.fit_transform(X_train)

X_test = ct.transform(X_test)


# exporting to csv
# final_train_df = X_train.copy().join(y_train)
# final_test_df = X_test.copy()

# final_train_df.to_csv("../../../data/house-prices/cleaned/train.csv", index=False)

# final_test_df.to_csv("../../../data/house-prices/cleaned/test.csv", index=False)
