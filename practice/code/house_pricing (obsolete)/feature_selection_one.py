#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 22:54:16 2025

@description: This module is for generating df that has been transformed (encoded). It contains only X.

@export:
    Train data without "SalePrice" column (transformed)
    Test data (transformed)
@author: xenon
"""

import pandas as pd
from data_imputation import cleaned_df as df
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

# importing test dataframe
test_file_path = "../../data/house-prices/test.csv"

test_df = pd.read_csv(test_file_path)

# getting test specific df id
test_id = test_df["Id"]

# Ordinal categorical cols

ordinal_cols = [
    "MSSubClass",
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
    "GarageCond",

    # Removed from the data due to excessive missing values

    # "PoolQC",
    # "FireplaceQu",
    # "Fence"
]

# selecting numerical cols
num_df = df.select_dtypes(exclude=["category"]).copy()

# reseting index
num_df.reset_index(drop=True, inplace=True)

# Because the two columns below have been encoded, I am changing their dtype to category
change_to_cat = ["OverallQual", "OverallCond"]
num_df[change_to_cat] = num_df[change_to_cat].astype("category")

# selecting ordinal cols
ordinal_df = df[ordinal_cols].copy()

# selecting nominal cols
nominal_df = df.select_dtypes(include="category").drop(
    ordinal_cols, axis=1).copy()

# Encoding nominal_df

encoded_nominal_df = pd.get_dummies(nominal_df, dtype="int", drop_first=True)

# reseting index
encoded_nominal_df.reset_index(drop=True, inplace=True)


# Encoding ordinal data

# MSSubClass
ms_sub_cat = [

    20,
    30,
    40,
    45,
    50,
    60,
    70,
    75,
    80,
    85,
    90,
    120,
    150,
    160,
    180,
    190

]

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
    "MSSubClass": ms_sub_cat
}


def buildEncoder(categories, col_name):
    """
    @params:
        categories: orderly arrangement of the categorical values
        col_name: Name of the current column
    @action:
        This function transform the ordinal_df[col_name] using OrdinalEncoder
    @return:
        Return encoder object
    """
    enc = OrdinalEncoder(categories=[categories])

    # getting specific column data
    original_col_data = ordinal_df.loc[:, [col_name]]

    # Transformation
    res = enc.fit_transform(original_col_data).flatten()

    # Return statement
    return [enc, res]


# creating new df
encoded_ordinal_df = pd.DataFrame()
encoder_dict = {}

# Running a for loop for encoding data
for col_name, categories in mapping_dict.items():

    # encoding data
    enc, res = buildEncoder(categories, col_name)

    # storing the result
    encoded_ordinal_df[col_name] = res

    # storing the encoder for probable future use
    encoder_dict[col_name] = enc

# reseting index
encoded_ordinal_df.reset_index(drop=True, inplace=True)


# final product

# The indexes have been reseted for all df to prevent indexError

final_df = pd.concat([num_df, encoded_nominal_df, encoded_ordinal_df], axis=1)

# filtering condition
condition = final_df["Id"].isin(test_id)

# sequestering test data
transformed_test_df = final_df[condition].reset_index(drop=True).copy()

# exporting df
product_df = final_df[~condition].copy()
