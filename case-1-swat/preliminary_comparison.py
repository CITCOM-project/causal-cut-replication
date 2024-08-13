import json
import pandas as pd

with open("logs/output_no_filter_lo_hi_sim_ctf_attack.json") as f:
    log1 = {(e["outcome"], e["failure"], tuple(tuple(x) for x in e["attack"])): e for e in json.load(f)}

with open("logs/output_no_filter_lo_hi_sim_ctf_normal.json") as f:
    log2 = {(e["outcome"], e["failure"], tuple(tuple(x) for x in e["attack"])): e for e in json.load(f)}

df = []
for attack1 in log1:
    entry1 = log1[attack1]
    if attack1 not in log2:
        continue
    entry2 = log2[attack1]
    print(attack1)
    for strategy in entry1["treatment_strategies"]:
        treatment_strategy = strategy["treatment_strategy"]
        if "effect_estimate" in strategy:
            for strategy2 in entry1["treatment_strategies"]:
                if strategy2["treatment_strategy"] == treatment_strategy:
                    print(
                        " ",
                        [(i["variable"], i["value"]) for i in treatment_strategy],
                        strategy["effect_estimate"]["trtrand"],
                        strategy2["effect_estimate"]["trtrand"],
                        strategy["effect_estimate"]["trtrand"] - strategy2["effect_estimate"]["trtrand"],
                    )
                    print(strategy)
                    df.append(
                        {
                            "outcome": attack1[0],
                            "attack": attack1[1][1:-1],
                            "control_strategy": list(attack1[2]),
                            "treatment_strategies": [(i["variable"], i["value"]) for i in treatment_strategy],
                            "effect_estimate_attack": strategy["effect_estimate"]["trtrand"],
                            "effect_estimate_attack_significant": not strategy["ci_low"][0]
                            < 1
                            < strategy["ci_high"][0],
                            "effect_estimate_normal": strategy2["effect_estimate"]["trtrand"],
                            "effect_estimate_normal_significant": not strategy2["ci_low"][0]
                            < 1
                            < strategy2["ci_high"][0],
                            "diff": strategy["effect_estimate"]["trtrand"] - strategy2["effect_estimate"]["trtrand"],
                        }
                    )

df = pd.DataFrame(df)
print(df)
df.to_csv("logs/results.csv")
