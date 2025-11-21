# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt


file_path = "../../data/breast_cancer.csv"

fig, ax = plt.subplots()

df = pd.read_csv(file_path)
enc = LabelEncoder()
df["binary_dx"] =  enc.fit_transform(df["diagnosis"])

X = df.drop(["id", "diagnosis","binary_dx"], axis=1)

y= df["diagnosis"]


# Feature selection

# Tree classifier method
model = ExtraTreesClassifier()

model.fit(X, y)

score = model.feature_importances_

best_df = pd.DataFrame()
best_df["score"] = score
best_df["label"] = [*X.columns]

best_ten = best_df.sort_values(by ="score", ascending=False).head(10)

# sns.barplot(data=best_ten, x = "label", y ="score", ax=ax)

# ax.set_xticklabels(ax.get_xticklabels(), rotation=65)

best_features = best_ten["label"]

# correlation method

# corr_df = df[[*best_features, "binary_dx"]]
# corr = corr_df.corr()

# sns.heatmap(corr, annot =True, ax=ax)

# Classification model

X = df[best_features].values
y = df["diagnosis"].values

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)


# Classification model
logit = LogisticRegression(max_iter=10000)

# fitting
logit.fit(X_train, y_train)

# classifying
y_pred = logit.predict(X_test)

acc = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

cf = classification_report(y_test, y_pred)
print(acc)
print(cm)
print(cf)


