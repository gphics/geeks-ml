#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 20:59:00 2025

@author: xenon
"""

import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression

np.random.seed(42)

X = np.random.rand(50, 1) * 100

Y = 3.5 * X + np.random.randn(50, 1) * 20


model = LinearRegression()
model.fit(X, Y)

y_pred = model.predict(X)

sns.scatterplot(x=X.ravel(), y=Y.flatten(), color="blue", label="Data Points")
sns.lineplot(x=X.reshape(-1), y=y_pred.ravel(),
             color="red", label="Regression Line")

print("Slope", model.coef_)
print("Intercept", model.intercept_)
print(X.ravel())
