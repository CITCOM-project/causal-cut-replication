"""
This module processes the causal test logs and draws the figures for openAPS.
"""

import sys
import os
import json
import matplotlib.pyplot as plt

from constants import BASELINE, TOOLNAME, RED, GREEN, BLUE, MAGENTA
from grouped_boxplot import plot_grouped_boxplot, bag_plot

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
        for attack in log:
            assert "greedy_minimal" in attack, f"No greedy_minimal in {os.path.join(root, file)}"
            if "error" in attack:
                if "treatment_strategies" not in attack:
                    assert attack["error"] in [
                        "Missing data for control_strategy",
                        "No faults observed. P(error) = 0",
                    ], f"Bad error {attack['error']} in {os.path.join(root, file)}"
                    attack["treatment_strategies"] = []
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

attack_id_length = {attack["attack_index"]: len(attack["attack"]) for attack in attacks}
original_attack_lengths = sorted(list(set(attack_id_length.values())))
attack_ids = sorted(list(attack_id_length.keys()))

data_samples = list(range(500, 5000, 500))


# RQ1: Baseline - minimal traces produced by Poskitt [2023]
# (1a) Measure the length of the "tool-minimised" traces, comparing to length of original
# Group by trace length
greedy_attack_lengths = [
    [len(attack["greedy_minimal"]) for attack in attacks if len(attack["attack"]) == length]
    for length in original_attack_lengths
]
greedy_attack_lengths_combinatorial = [
    [len(attack["minimal"]) for attack in attacks if len(attack["attack"]) == length]
    for length in original_attack_lengths
]
our_attack_lengths = [
    [len(attack["extended_interventions"]) for attack in attacks if len(attack["attack"]) == length]
    for length in original_attack_lengths
]
our_attack_lengths_combinatorial = [
    # We can't feasibly minimise attacks of length greater than 20 as there's over 16M combinations (16,777,215)
    (
        [len(attack["minimised_extended_interventions"]) for attack in attacks if len(attack["attack"]) == length]
        if length < 20
        else []
    )
    for length in original_attack_lengths
]

plot_grouped_boxplot(
    [greedy_attack_lengths, greedy_attack_lengths_combinatorial, our_attack_lengths, our_attack_lengths_combinatorial],
    savepath=f"{figures}/rq1-attack-lengths.png",
    labels=[BASELINE, f"{BASELINE} (optimal)", TOOLNAME, f"{TOOLNAME} (optimal)"],
    colours=[RED, BLUE, GREEN, MAGENTA],
    markers=["x", "o", "s", 2],
    title="Pruned Trace Lengths",
    xticklabels=original_attack_lengths,
    xlabel="Original trace length",
    ylabel="Tool-minimised trace length",
)

# (1b) Measure the length of the "tool-minimised" traces, comparing to length of original
# Show each trace separately
l_greedy_attack_lengths = [
    [len(attack["greedy_minimal"]) for attack in attacks if attack["attack_id"] == attack_id]
    for attack_id, _ in sorted(list(attack_id_length.items()), key=lambda x: (x[1], x[0]))
]
l_greedy_attack_lengths_combinatorial = [
    [len(attack["minimal"]) for attack in attacks if attack["attack_id"] == attack_id]
    for attack_id, _ in sorted(list(attack_id_length.items()), key=lambda x: (x[1], x[0]))
]
l_our_attack_lengths = [
    [len(attack["extended_interventions"]) for attack in attacks if attack["attack_id"] == attack_id]
    for attack_id, _ in sorted(list(attack_id_length.items()), key=lambda x: (x[1], x[0]))
]
l_our_attack_lengths_combinatorial = [
    # We can't feasibly minimise attacks of length greater than 20 as there's over 16M combinations (16,777,215)
    (
        [len(attack["minimised_extended_interventions"]) for attack in attacks if attack["attack_id"] == attack_id]
        if len(attack["attack"]) < 20
        else []
    )
    for attack_id, _ in sorted(list(attack_id_length.items()), key=lambda x: (x[1], x[0]))
]

plot_grouped_boxplot(
    [
        l_greedy_attack_lengths,
        l_greedy_attack_lengths_combinatorial,
        l_our_attack_lengths,
        l_our_attack_lengths_combinatorial,
    ],
    savepath=f"{figures}/rq1-attack-lengths-per-trace.png",
    labels=[BASELINE, f"{BASELINE} (optimal)", TOOLNAME, f"{TOOLNAME} (optimal)"],
    colours=[RED, BLUE, GREEN, MAGENTA],
    markers=["x", "o", "s", 2],
    title="Pruned Trace Lengths",
    xticklabels=[
        attack_length for _, attack_length in sorted(list(attack_id_length.items()), key=lambda x: (x[1], x[0]))
    ],
    xlabel="Original trace length",
    ylabel="Tool-minimised trace length",
)


# (2) Measure the proportion of the "tool-minimise" traces that are spurious. Report as the average proportion again.
greedy_spurious = [
    [
        (len(attack["greedy_minimal"]) - len(attack["minimal"])) / len(attack["attack"])
        for attack in attacks
        if len(attack["attack"]) == length
    ]
    for length in original_attack_lengths
]
our_spurious = [
    [
        (len(attack["extended_interventions"]) - len(attack["minimised_extended_interventions"]))
        / len(attack["attack"])
        for attack in attacks
        # we can't combinatorially minimise the long traces in reasonable time
        if len(attack["attack"]) == length and length < 20
    ]
    for length in original_attack_lengths
]
plot_grouped_boxplot(
    [greedy_spurious, our_spurious],
    savepath=f"{figures}/rq1-proportion-spurious.png",
    labels=[BASELINE, TOOLNAME],
    colours=[RED, GREEN],
    markers=["x", "s"],
    title="Spurious Events",
    xticklabels=original_attack_lengths,
    xlabel="Original trace length",
    ylabel="Proportion of Remaining Spurious Events",
)

# RQ2: Baseline - minimal traces produced by Poskitt [2023]
# Measure number of executions required from simulator / CPS.
our_executions = [
    [attack["simulator_runs"] for attack in attacks if len(attack["attack"]) == length]
    for length in original_attack_lengths
]
plot_grouped_boxplot(
    [[[l] for l in original_attack_lengths], our_executions],
    savepath=f"{figures}/rq2-simulator-executions.png",
    labels=[BASELINE, TOOLNAME],
    colours=[RED, GREEN],
    markers=["x", "s"],
    title="Simulator Executions",
    xticklabels=original_attack_lengths,
    xlabel="Original trace length",
    ylabel="Number of Simulations to Minimise the Trace",
)

bag_plot(
    [item for y, l in zip(greedy_spurious, original_attack_lengths) if l < 20 for item in [l] * len(y)],
    [item for y, l in zip(greedy_spurious, original_attack_lengths) if l < 20 for item in y],
    colour=GREEN,
    marker="s",
    label=BASELINE,
)
bag_plot(
    [item for x, l in zip(our_executions, original_attack_lengths) if l < 20 for item in x],
    [item for y, l in zip(our_spurious, original_attack_lengths) if l < 20 for item in y],
    colour=RED,
    marker="x",
    label=TOOLNAME,
    title="Remaining Spurious per Simulation",
    xlabel="Number of Simulations to Minimise the Trace",
    ylabel="Proportion of Remaining Spurious Events",
    savepath=f"{figures}/rq2-executions-spurious.png",
)

bag_plot(
    [item for y, l in zip(greedy_attack_lengths, original_attack_lengths) if l < 20 for item in [l] * len(y)],
    [item for y, l in zip(greedy_attack_lengths, original_attack_lengths) if l < 20 for item in y],
    colour=GREEN,
    marker="s",
    label=BASELINE,
)
bag_plot(
    [item for x, l in zip(our_executions, original_attack_lengths) if l < 20 for item in x],
    [item for y, l in zip(our_attack_lengths, original_attack_lengths) if l < 20 for item in y],
    colour=RED,
    marker="x",
    label=TOOLNAME,
    title="Pruning per Simulation",
    xlabel="Number of Simulations to Minimise the Trace",
    ylabel="Length of tool-minimised trace",
    savepath=f"{figures}/rq1-executions-pruning.png",
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
