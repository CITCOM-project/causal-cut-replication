import json

with open("logs/output_no_filter_lo_hi_sim_ctf_attack.json") as f:
    log1 = {(e["outcome"], e["failure"], tuple(tuple(x) for x in e["attack"])): e for e in json.load(f)}

with open("logs/output_no_filter_lo_hi_sim_ctf_normal.json") as f:
    log2 = {(e["outcome"], e["failure"], tuple(tuple(x) for x in e["attack"])): e for e in json.load(f)}

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
