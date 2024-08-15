import json
import sys
import pandas as pd

if len(sys.argv) != 2:
    raise ValueError("Please provide a file to convert.")

fname = sys.argv[1]

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
            for inx, (var, val) in enumerate(record["attack"])
        ]
        continue
    assert len(record["treatment_strategies"]) == len(record["attack"])
    for inx, ((var, val), record) in enumerate(zip(record["attack"], record["treatment_strategies"])):
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
        result = result | result.pop("adequacy")
        result["effect_estimate"] = result["effect_estimate"]["trtrand"]
        result["kurtosis"] = result["kurtosis"]["trtrand"]
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
data.to_csv(fname.replace(".json", ".csv"))
