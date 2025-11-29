#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 22:40:26 2025

@author: xenon
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

G = nx.Graph()

G.add_nodes_from(np.arange(1, 5))

G.add_edge(1, 2)
G.add_edge(1, 5)
G.add_edge(4, 5)
G.add_edge(3, 5)
G.add_edge(2, 3)

G.remove_node(5)
G.add_node(5)
G.add_edge(1, 5)
G.add_edge(4, 5)
G.add_edge(3, 5)
nx.draw(G, with_labels=True, node_size=1000, node_color="purple")
