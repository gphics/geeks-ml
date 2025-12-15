#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 07:08:04 2025

@author: xenon
"""

from base import y_train, y_true
from reduction import y_pred as decomposed_pred
from regression import y_pred_i as y_pred


import pandas as pd

note = """
    y_train: The target from the train dataset
    y_pred: The target predicted 
    decomposed_pred: The target predicted after decomposing X_train
    y_true: The target provided in the submission sample

"""

print(note)

desc = pd.DataFrame()
desc["y_train"] = y_train.drop(0)
desc["y_pred"] = y_pred
desc["decomposed_pred"] = decomposed_pred
desc["y_true"] = y_true

print(desc.describe())
