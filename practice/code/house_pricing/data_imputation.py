#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 23:26:37 2025

@description:
    This module fills missing values in both train and test data excluding the "SalePrice" column

@author: xenon
"""

import pandas as pd
import numpy as np

file_path = "../../data/house-prices/train.csv"
second_file_path = "../../data/house-prices/test.csv"


df = pd.read_csv(file_path)

# Does not contain SalePrice column
test_df = pd.read_csv(second_file_path)

df = pd.concat([df, test_df])
# print(df.shape)
# print(df["Id"].nunique())
# Dropping the target "SalePrice" before imputation
df.drop("SalePrice", axis=1, inplace=True)


# getting null columns sum
null_cols = np.round(df.isnull().sum() * 100 / df.shape[0], 2)
# print(null_cols.sort_values(ascending=False))

# when there is more than 30% missing values
dropping_condition = null_cols > 30
cols_to_drop = null_cols[dropping_condition].index


# Columns dropped
# This is for refrence purpose
cols_droped = ['Alley', 'MasVnrType',
               'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']

# dropping cols
df.drop(cols_to_drop, axis=1, inplace=True)

# getting null columns sum
null_cols = df.isnull().sum()

# print(null_cols.sort_values(ascending=False))

imputing_conditions = null_cols > 0

cols_to_impute = null_cols[imputing_conditions].index


# # subsetting dataframe
impute_df = df[cols_to_impute]
# print(impute_df.info())


# # Imputing

# getting the cols of numerical data
num_part = impute_df.select_dtypes(exclude=["object"]).copy()

# Dropping "GarageYrBlt" as it contains year and need special imputation

num_part.drop("GarageYrBlt", axis=1, inplace=True)
num_part_cols = num_part.columns


# calculating imputation values
num_imputation_vals = num_part[num_part_cols].apply(lambda x: x.mean())


# filling missing numerical values
num_part = num_part.fillna(num_imputation_vals)

# storing the new num data
df[num_part_cols] = num_part

# getting null columns sum to verify the previous ops
null_cols = df.isnull().sum()


# categorical columns
cat_part = impute_df.select_dtypes(include=["object"]).copy()

cat_part_cols = cat_part.columns

cat_imputation_vals = cat_part[cat_part_cols].apply(lambda x: x.mode()[0])

# Mass fillling for categorical features
cat_part = cat_part.fillna(cat_imputation_vals)

# storing cat imputed cols
df[cat_part_cols] = cat_part

# Dealing with GarageYrBlt
df["GarageYrBlt"] = impute_df["GarageYrBlt"].fillna(
    impute_df["GarageYrBlt"].mode()[0]).astype(str)


# getting null columns sum for confirmation
null_cols = df.isnull().sum()

# Transforming object dtype to category

cat_cols = df.select_dtypes(include="object").columns

df[cat_cols] = df[cat_cols].astype("category")

df["MSSubClass"] = df["MSSubClass"].astype("category")

# Performing deep df copy
cleaned_df = df.copy()


# Assrting the proper cleaning of the data
assert (cleaned_df.isnull().sum() == 0).all(), "The data is not well cleaned"
