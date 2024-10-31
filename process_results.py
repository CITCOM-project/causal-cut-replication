"""
Process the causal estimation results from the attack sequences.
"""

import os
from glob import glob
import json
import sys
import pandas as pd

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
    with open(outfile.replace(".csv", ".json"), "w") as f:
        json.dump(log, f)
else:
    with open(fname) as f:
        log = json.load(f)

data = []
for record in log:
    attack = record["attack"]
    attack_index = record["attack_index"]
    outcome = record["outcome"]
    spurious = record.get("spurious", [])

    basic_attack = {
        "attack_index": record["attack_index"],
        # "attack": record["attack"],
        "outcome": outcome,
        "failure": record["failure"],
        # "variable": var,
        # "value": val,
    }

    if "error" in record:
        print("WARNING: Error in record")
        data += [
            basic_attack
            | {
                "intervention_index": inx,
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
                basic_attack
                | {
                    "intervention_index": record["intervention_index"],
                    "spurious": record["intervention_index"] in spurious,
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
        result["significant"] = not result["ci_low"] < 1 < result["ci_high"]
        data.append(
            basic_attack
            | {
                "intervention_index": record["intervention_index"],
                "spurious": record["intervention_index"] in spurious,
                "error": None,
                "result": True,
                "len_control_group": record["len_control_group"],
                "len_treatment_group": record["len_treatment_group"],
            }
            | result
        )

data = pd.DataFrame(data).sort_values(["attack_index", "intervention_index"]).reset_index(drop=True)

data["spurious"] = data["spurious"]
data["necessary"] = ~data["spurious"]
data["significant"] = data.pop("significant")
data["correct"] = data["necessary"] == data["significant"]
# data["treatment_value"] = [t[i] for t, i in zip(data["treatment_value"], data["intervention_index"])]
# data["control"] = [t[i] for t, i in zip(data["control_value"], data["intervention_index"])]
data["treatment"] = [t[i] for t, i in zip(data["treatment"], data["intervention_index"])]
data.drop(["adjustment_set", "effect_measure"], axis=1, inplace=True)


def highlight_greaterthan(row):
    """
    Colour the rows based on the attack_index so that the results are easier to read.

    :param row: Pandas object representing a row of data.
    """
    index = pd.Series(data=False, index=row.index)
    index["attack_index"] = attack_indices[row.loc["attack_index"]] % 2 == 0
    return ["background-color: #eee" if index["attack_index"] else "" for v in index]


attack_indices = {v: k for k, v in enumerate(sorted(list(set(data["attack_index"]))))}
styled_df = data.style.apply(highlight_greaterthan, axis=1)
styled_df.to_excel(outfile.replace(".csv", ".xlsx"), engine="openpyxl")

print(data.dtypes)


data.to_csv(outfile, index=False)

# data["attack"] = [tuple(map(tuple, attack)) for attack in data["attack"]]
data = data.groupby("attack_index").filter(lambda gp: gp["necessary"].any())

if len(data) > 0:
    result = data.loc[data["result"]]
    print(result)
    tp = len(result.loc[result["necessary"] & result["significant"]])
    tn = len(result.loc[(~result["necessary"]) & ~result["significant"]])
    fp = len(result.loc[~result["necessary"] & result["significant"]])
    fn = len(result.loc[result["necessary"] & ~result["significant"]])

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

    print("Failed estimates", len(data.loc[~data["result"]]))

else:
    print("No successful estimates")
