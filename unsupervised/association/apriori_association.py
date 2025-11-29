#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 23:06:16 2025

@author: xenon
"""

import pandas as pd
import seaborn as sns
import networkx as nx
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

file_path = "../../data/trans_groceries.csv"

base_df = pd.read_csv(file_path)


transactions = base_df.groupby(["Member_number", "Date"])[
    "itemDescription"].apply(list).values.tolist()


te = TransactionEncoder()
te_list = te.fit_transform(transactions)

new_df = pd.DataFrame(te_list, columns=te.columns_)


# Running the algorithm

frequent_items = apriori(new_df, min_support=0.01, use_colnames=True)


rules = association_rules(
    frequent_items, metric="confidence", min_threshold=0.1)

rules = rules[rules["antecedents"].apply(lambda x: len(
    x) >= 1) & rules["consequents"].apply(lambda x: len(x) >= 1)]

print(rules)
