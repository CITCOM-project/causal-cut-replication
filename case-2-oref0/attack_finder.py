import pandas as pd
from glob import glob
import json

from safe_ranges import safe_ranges
from aps_digitaltwin.util import TrainingData


for i, fname in enumerate(glob("../data/processed_and_split/traces/*.csv")):
    df = pd.read_csv(fname)
    df["low"] = df["bg"] < safe_ranges["Glucose"]["lo"]
    df["high"] = df["bg"] > safe_ranges["Glucose"]["hi"]

    if df["high"].any() and not df["high"].iloc[0:3].any():
        print(fname)
        break

    # # if not df["safe"].all():
    # td = TrainingData(fname, timesteps=len(df))
    # interventions = list(filter(lambda x: x[2] > 0, td.interventions))
    # if df["low"].any():
    #     print(fname)
    # with open(fname.replace("traces", "attacks").replace(".csv", ".json"), "w") as f:
    #     json.dump(interventions, f)
