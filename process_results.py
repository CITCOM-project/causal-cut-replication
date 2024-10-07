import json
import sys
import pandas as pd
import os
from glob import glob

if len(sys.argv) != 2:
    raise ValueError("Please provide a file (or directory of files) to convert.")

fname = sys.argv[1]
outfile = fname.replace(".json", ".csv")

if os.path.isdir(fname):
    if outfile.endswith("/"):
        outfile = outfile[:-1]
    outfile += ".csv"
    log = []
    for fname in glob(f"{fname}/*.json"):
        with open(fname) as f:
            log += json.load(f)
else:
    with open(fname) as f:
        log = json.load(f)

data = []
for record in log:
    attack = record["attack"]
    outcome = record["outcome"]
    spurious = record.get("spurious", [])
    if "error" in record:
        data += [
            {
                "attack": attack,
                "outcome": outcome,
                "variable": var,
                "value": val,
                "spurious": inx in spurious,
                "error": record["error"],
                "result": False,
            }
            for inx, (t, var, val) in enumerate(record["attack"])
        ]
        continue
    # assert len(record["treatment_strategies"]) == len(record["attack"])

    for inx, ((t, var, val), record) in enumerate(zip(record["attack"], record["treatment_strategies"])):
        if "error" in record:
            data.append(
                {
                    "attack": attack,
                    "outcome": outcome,
                    "variable": var,
                    "value": val,
                    "spurious": inx in spurious,
                    "error": record["error"],
                    "result": False,
                }
            )
            continue
        result = record["result"]
        if "adequacy" in result:
            result = result | result.pop("adequacy")
            result["kurtosis"] = result["kurtosis"]["trtrand"]
        result["effect_estimate"] = result["effect_estimate"]["trtrand"]
        result["ci_low"] = result["ci_low"][0]
        result["ci_high"] = result["ci_high"][0]
        result["significant"] = not (result["ci_low"] < 1 < result["ci_high"])
        data.append(
            {
                "attack": attack,
                "outcome": outcome,
                "variable": var,
                "value": val,
                "spurious": inx in spurious,
                "error": None,
                "result": True,
            }
            | result
        )

data = pd.DataFrame(data)
data["necessary"] = ~data["spurious"]
data.to_csv(outfile)

data["significant"] = data["significant"].astype(bool)
tp = len(data.loc[data["necessary"] & data["significant"]])
tn = len(data.loc[(~data["necessary"]) & ~data["significant"]])
fp = len(data.loc[~data["necessary"] & data["significant"]])
fn = len(data.loc[data["necessary"] & ~data["significant"]])

print("True positives", tp)
print("True negatives", tn)
print("False positives", fp)
print("False negatives", fn)

sensitivity = tp / (tp + fp)
specificity = tn / (tn + fp)
bcr = (sensitivity + specificity) / 2

print()

print("sensitivity", sensitivity)
print("specificity", specificity)
print("bcr", bcr)
