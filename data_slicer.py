#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 14:03:03 2024

@author: michael
"""

import pandas as pd
from config import timesteps
from multiprocessing import Pool

normal_path = "data/SWaT_Dataset_Normal_v0.csv"
attack_path = "data/SWaT_Dataset_Attack_v0.csv"

df = pd.concat([pd.read_csv(normal_path, index_col=0), pd.read_csv(attack_path, index_col=0)])
print(len(df))

time_skip = timesteps
block_size = timesteps + 1


def block_data(i):
    return df.iloc[i : i + block_size]


def time_block(block):
    datum = {}
    for column in block:
        column = column.strip()
        datum = datum | {f"{column}_{j}": val for j, val in enumerate(block[column])}
    return datum


pool = Pool(8)
blocks = pool.map(block_data, range(0, len(df) - block_size, time_skip))
data = pool.map(time_block, blocks)
# for i in range(0, len(df) - block_size, time_skip):
#     block = df.iloc[i : i + block_size]
#     datum = {}
#     for column in block:
#         column = column.strip()
#         if column == "Normal/attack":
#             block[column] = [x.strip() == "attack" for x in block[column]]
#         datum = datum | {f"{column}_{j}": val for j, val in enumerate(block[column])}
#     data.append(datum)

data = pd.DataFrame(data)

attack = data[[col for col in data.columns if col.startswith("attack")]].any(axis=1)
data = data.drop([col for col in data.columns if col.startswith("attack")], axis=1)
data["attack"] = attack
assert data.attack.any()

data.to_csv("data/timestep_data.csv")

normal_data = data.query("not attack")[:50]
attack_data = data.query("attack")[:50]
pd.concat([normal_data, attack_data]).to_csv("data/timestep_data_small.csv")

print(len(data))
