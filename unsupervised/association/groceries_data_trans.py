#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 22:54:38 2025

@author: xenon
"""

import pandas as pd

file_path = "../../data/Groceries_dataset.csv"


df = pd.read_csv(file_path)

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month
df["day"] = df["Date"].dt.day
df["day_of_week"] = df["Date"].dt.day_of_week


df.to_csv("../../data/trans_groceries.csv", index=False)
