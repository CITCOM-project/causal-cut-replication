import pandas as pd
from actuators import actuators
from itertools import product
import json
from tqdm import tqdm

df = pd.read_csv("../data/long_data.csv")

frequencies = {
    a1: {
        v1: {a2: {v2: {a3: {v3: 1 for v3 in [0, 1]} for a3 in actuators} for v2 in [0, 1]} for a2 in actuators + [None]}
        for v1 in [0, 1]
    }
    for a1 in actuators + [None]
}

for _, group in tqdm(df.groupby("id")):
    group = group[actuators].to_dict(orient="records")
    group = [[(None, 1) for _ in actuators]] + [list(row.items()) for row in group] + [[(None, 1) for _ in actuators]]
    for i in range(0, len(group) - 3):
        previous, current, next = group[i : i + 3]
        for a1, a2 in product(previous, next):
            a1, v1 = a1
            a2, v2 = a2
            for a3 in current:
                a3, v3 = a3
                frequencies[a1][v1][a2][v2][a3][v3] += 1

with open("usage_profile_freqs.json", "w") as f:
    json.dump(frequencies, f, indent=2)

for a1 in tqdm(frequencies):
    for v1 in [0, 1]:
        for a2 in frequencies[a1][v1]:
            for v2 in [0, 1]:
                for a3 in frequencies[a1][v1][a2][v2]:
                    total = float(sum(frequencies[a1][v1][a2][v2][a3].values()))
                    for v3 in [0, 1]:
                        frequencies[a1][v1][a2][v2][a3][v3] /= total


with open("usage_profile_probs.json", "w") as f:
    json.dump(frequencies, f, indent=2)
