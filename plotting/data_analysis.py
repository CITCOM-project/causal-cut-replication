"""
This module processes the causal test logs and draws the figures for openAPS.
"""

import sys
import os
import json
import matplotlib.pyplot as plt

from constants import BASELINE, TOOLNAME, RED, GREEN, BLUE, MAGENTA
from grouped_boxplot import plot_grouped_boxplot

plt.style.use("ggplot")


if len(sys.argv) != 2:
    raise ValueError("Please provide the directory of the log files, e.g. case-2-oref0/logs")
logs = sys.argv[1]
figures = sys.argv[1].replace("/logs", "/figures")

if not os.path.exists(figures):
    os.mkdir(figures)


attacks = []
for root, dirs, files in os.walk(logs):
    for file in files:
        if not file.endswith(".json"):
            continue
        with open(os.path.join(root, file)) as f:
            log = json.load(f)
        for attack in attacks:
            assert "treatment_strategies" in attack, f"No treatment strategies for {os.path.join(root, file)}"
        attacks += log

print(len(attacks))

for attack in attacks:
    for intervention in attack["treatment_strategies"]:
        if "result" in intervention:
            assert len(intervention["result"]["ci_low"]) == 1
            intervention["result"]["ci_low"] = intervention["result"]["ci_low"][0]
            assert len(intervention["result"]["ci_high"]) == 1
            intervention["result"]["ci_high"] = intervention["result"]["ci_high"][0]

        pruned = False
        if "result" in intervention and not (intervention["result"]["ci_low"] < 1 < intervention["result"]["ci_high"]):
            pruned = True
        intervention["pruned"] = pruned

original_attack_lengths = sorted(list(set(len(attack["attack"]) for attack in attacks)))

data_samples = list(range(500, 5000, 500))


# RQ1: Baseline - minimal traces produced by Poskitt [2023]
# (1) Measure the length of the "tool-minimised" traces, comparing to length of original
greedy_attack_lengths = {
    length: [len(attack["greedy_minimal"]) for attack in attacks if len(attack["attack"]) == length]
    for length in original_attack_lengths
}
greedy_attack_lengths_combinatorial = {
    length: [len(attack["minimal"]) for attack in attacks if len(attack["attack"]) == length]
    for length in original_attack_lengths
}
our_attack_lengths = {
    length: [len(attack["extended_interventions"]) for attack in attacks if len(attack["attack"]) == length]
    for length in original_attack_lengths
}
our_attack_lengths_combinatorial = {
    length: [len(attack["minimised_extended_interventions"]) for attack in attacks if len(attack["attack"]) == length]
    # We can't feasibly minimise attacks of length greater than 20 as there's over 16M combinations (16,777,215)
    for length in original_attack_lengths
    if length < 20
}

plot_grouped_boxplot(
    [
        [greedy_attack_lengths[l] for l in original_attack_lengths if l < 20],
        [greedy_attack_lengths_combinatorial[l] for l in original_attack_lengths if l < 20],
        [our_attack_lengths[l] for l in original_attack_lengths if l < 20],
        [our_attack_lengths_combinatorial[l] for l in original_attack_lengths if l < 20],
    ],
    savepath=f"{figures}/rq1-attack-lengths.png",
    labels=[BASELINE, f"{BASELINE} (optimal)", TOOLNAME, f"{TOOLNAME} (optimal)"],
    colours=[RED, BLUE, GREEN, MAGENTA],
    markers=["x", "o", "s", 2],
    title="Pruned Trace Lengths",
    xticklabels=original_attack_lengths,
    xlabel="Original trace length",
    ylabel="Tool-minimised trace length",
)


# (2) Measure the proportion of the "tool-minimise" traces that are spurious. Report as the average proportion again.
greedy_spurious = {
    length: [
        (len(attack["greedy_minimal"]) - len(attack["minimal"])) / len(attack["attack"])
        for attack in attacks
        if len(attack["attack"]) == length
    ]
    for length in original_attack_lengths
}
our_spurious = {
    length: [
        (len(attack["extended_interventions"]) - len(attack["minimised_extended_interventions"]))
        / len(attack["attack"])
        for attack in attacks
        # we can't combinatorially minimise the long traces in reasonable time
        if len(attack["attack"]) == length and length < 20
    ]
    for length in original_attack_lengths
}
plot_grouped_boxplot(
    [
        [greedy_spurious[l] for l in original_attack_lengths],
        [our_spurious[l] for l in original_attack_lengths],
    ],
    savepath=f"{figures}/rq1-proportion-spurious.png",
    labels=[BASELINE, TOOLNAME],
    colours=[RED, GREEN],
    markers=["x", "s"],
    title="Spurious Events",
    xticklabels=original_attack_lengths,
    xlabel="Original trace length",
    ylabel="Proportion of Spurious Events",
)

# RQ2: Baseline - minimal traces produced by Poskitt [2023]
# Measure number of executions required from simulator / CPS.
our_executions = {
    length: [attack["simulator_runs"] for attack in attacks if len(attack["attack"]) == length]
    for length in original_attack_lengths
}
plot_grouped_boxplot(
    [
        [[l] for l in original_attack_lengths],
        [our_executions[l] for l in original_attack_lengths],
    ],
    savepath=f"{figures}/rq2-simulator-executions.png",
    labels=[BASELINE, TOOLNAME],
    colours=[RED, GREEN],
    markers=["x", "s"],
    title="Simulator Executions",
    xticklabels=original_attack_lengths,
    xlabel="Original trace length",
    ylabel="Number of Simulations to Minimise the Trace",
)
# RQ3:
# look into the impact of the different levels of data provision.
# our_adequacy = {
#     length: [
#         attack.get("result", {}).get("adequacy", {}).get("kurtosis", {}).get("trtrand", None)
#         for attack in attacks
#         if len(attack["attack"]) == length
#     ]
#     for length in original_attack_lengths
# }
