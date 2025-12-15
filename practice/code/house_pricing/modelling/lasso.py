#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 21:34:52 2025

@author: xenon
"""

from sklearn.linear_model import LassoCV
from base import union_df, intersect_df, y_train, y_true
from sklearn.metrics import r2_score
import pandas as pd

lasso = LassoCV()


lasso.fit(union_df[0], y_train)


y_pred = lasso.predict(union_df[1])


# r_score = r2_score(y_true, y_pred)

# print(r_score)

df_pred = pd.DataFrame()

df_pred["pred"] = y_pred

df_pred.describe()
