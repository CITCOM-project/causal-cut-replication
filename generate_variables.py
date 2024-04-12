#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 14:51:14 2024

@author: michael
"""

import pandas as pd
import networkx as nx
import json

datapath = "data/SWaT_Dataset_Normal_v0.csv"
dagpath = "dag.dot"

dag = nx.DiGraph(nx.nx_pydot.read_dot(dagpath))
df = pd.read_csv(datapath, header=1)

variables = [{
    "name": node,
    "datatype": df.dtypes[node]
    .__class__.__name__.replace("64DType", "")
    .lower()
    if node in df.dtypes
    else "float",
    "typestring": "Input" if len(list(dag.predecessors(node))) == 0 else "Output"
} for node in dag.nodes]

with open("variables.json", 'w') as f:
    json.dump({"variables": variables}, f, indent=2)

