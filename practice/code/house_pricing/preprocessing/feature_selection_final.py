#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 19:18:25 2025

@author: xenon

@step: 5c

@description:
    This module finalize the feature selection process.


"""

from feature_selection_two import selected_cols as rfe_cols

from feature_selection_one import selected_cols as mutual_cols


rfe_set = set(rfe_cols)

mutual_set = set(mutual_cols)

# product

intersect_cols = rfe_set & mutual_set

union_cols = rfe_set | mutual_set
