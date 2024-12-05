"""
This module processes the causal test logs and draws the figures.
"""

import sys
import os
import json
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")

RED = "#DC3220"
BLUE = "#005AB5"
GREEN = "#009E73"
MAGENTA = "#DC267F"

TOOLNAME = "CausalCut"
BASELINE = "Greedy Heuristic"


def color(color, flierprops={}):
    return dict(
        boxprops=dict(color=color),
        capprops=dict(color=color),
        whiskerprops=dict(color=color),
        flierprops=dict(color=color, markeredgecolor=color) | flierprops,
        medianprops=dict(color=color),
    )


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
        attacks += log

print(len(attacks))

for attack in attacks:
    for intervention in attack["treatment_strategies"]:
        assert len(intervention["result"]["ci_low"]) == 1
        intervention["result"]["ci_low"] = intervention["result"]["ci_low"][0]
        assert len(intervention["result"]["ci_high"]) == 1
        intervention["result"]["ci_high"] = intervention["result"]["ci_high"][0]

        pruned = False
        if "result" in intervention and not (intervention["result"]["ci_low"] < 1 < intervention["result"]["ci_high"]):
            pruned = True
        intervention["pruned"] = pruned


# RQ1: Baseline - minimal traces produced by Poskitt [2023]
# (1) Measure the length of the "tool-minimised" traces, comparing to length of original
original_attack_lengths = sorted(list(set(len(attack["attack"]) for attack in attacks)))

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
    for length in original_attack_lengths
}

fig, ax = plt.subplots()

WIDTH = 0.5
PLOTS = 4
MARKERSIZE = 3

ax.boxplot(
    [greedy_attack_lengths[l] for l in original_attack_lengths],
    positions=np.array(range(len(original_attack_lengths))) * (PLOTS + 1),
    widths=WIDTH,
    label=BASELINE,
    **color(RED, flierprops={"marker": "x", "markersize": MARKERSIZE}),
)
ax.boxplot(
    [greedy_attack_lengths_combinatorial[l] for l in original_attack_lengths],
    positions=np.array(range(len(original_attack_lengths))) * (PLOTS + 1) + 1,
    widths=WIDTH,
    label=f"{BASELINE} (optimal)",
    **color(BLUE, flierprops={"marker": "x", "markersize": MARKERSIZE}),
)
ax.boxplot(
    [our_attack_lengths[l] for l in original_attack_lengths],
    positions=np.array(range(len(original_attack_lengths))) * (PLOTS + 1) + 2,
    widths=WIDTH,
    label=TOOLNAME,
    **color(GREEN, flierprops={"marker": "o", "markersize": MARKERSIZE}),
)
ax.boxplot(
    [our_attack_lengths_combinatorial[l] for l in original_attack_lengths],
    positions=np.array(range(len(original_attack_lengths))) * (PLOTS + 1) + 3,
    widths=WIDTH,
    label=f"{TOOLNAME} (optimal)",
    **color(MAGENTA, flierprops={"marker": "o", "markersize": MARKERSIZE}),
)

ax.set_title("Pruning")
ax.set_xlabel("Original trace length")
ax.set_ylabel("Tool-minimised trace length")

ax.set_xticks(np.array(range(len(original_attack_lengths))) * (PLOTS + 1) + 1 + WIDTH, original_attack_lengths)
ax.legend()
plt.savefig(os.path.join(figures, "rq1-attack-lengths.png"))
plt.clf()

# (2) Measure the proportion of the "tool-minimise" traces that are spurious. Report as the average proportion again.
greedy_spurious = {
    length: [
        len(attack["greedy_minimal"]) - len(attack["minimal"]) for attack in attacks if len(attack["attack"]) == length
    ]
    for length in original_attack_lengths
}
our_spurious = {
    length: [
        len(attack["extended_interventions"]) - len(attack["minimised_extended_interventions"])
        for attack in attacks
        if len(attack["attack"]) == length
    ]
    for length in original_attack_lengths
}

# RQ2: Baseline - minimal traces produced by Poskitt [2023]
# Measure number of executions required from simulator / CPS.
#
# RQ3:
# look into the impact of the different levels of data provision.
