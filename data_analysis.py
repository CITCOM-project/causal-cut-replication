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


def plot_grouped_boxplot(
    groups,
    savepath=None,
    width=0.6,
    labels=None,
    colours=None,
    markers=None,
    title=None,
    xticklabels=None,
    xlabel=None,
    ylabel=None,
):
    _, ax = plt.subplots()
    plots = len(groups)
    if isinstance(labels, list) and len(labels) != plots:
        raise ValueError("If providing labels, please ensure that you provide as many as you have plots")
    if isinstance(colours, list) and len(labels) != plots:
        raise ValueError("If providing colours, please ensure that you provide as many as you have plots")
    for i, boxes in enumerate(groups):
        marker = markers[i] if isinstance(markers, list) else markers if markers is not None else "o"
        ax.boxplot(
            boxes,
            positions=np.array(range(len(original_attack_lengths))) * (plots + 1) + i,
            widths=width,
            label=labels[i] if labels is not None else None,
            **color(
                colours[i] if colours is not None else None, flierprops={"marker": marker, "markersize": width * 2}
            ),
        )
    ax.set_xticks(
        np.array(range(len(xticklabels))) * (plots + 1) + (((plots + (plots / 2) - 1) * width) / 2), xticklabels
    )

    if labels is not None:
        ax.legend()
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(xlabel)

    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.show()
    plt.clf()


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


plot_grouped_boxplot(
    [
        [greedy_attack_lengths[l] for l in original_attack_lengths],
        [greedy_attack_lengths_combinatorial[l] for l in original_attack_lengths],
        [our_attack_lengths[l] for l in original_attack_lengths],
        [our_attack_lengths_combinatorial[l] for l in original_attack_lengths],
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
        if len(attack["attack"]) == length
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
