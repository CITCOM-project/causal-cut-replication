"""
This module processes the causal test logs and draws the figures for openAPS.
"""

import sys
import os
import json
import matplotlib.pyplot as plt
from matplotlib import gridspec

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
        path = os.path.normpath(root).split(os.sep)
        sample_size = [int(x.split("_")[1]) for x in path if x.startswith("sample_")][0]
        ci_alpha = [int(x.split("_")[1]) for x in path if x.startswith("ci_")][0]
        for attack in log:
            assert "greedy_minimal" in attack, f"No greedy_minimal in {os.path.join(root, file)}"
            attack["sample_size"] = sample_size
            attack["ci_alpha"] = ci_alpha
            if "error" in attack:
                if "treatment_strategies" not in attack:
                    assert attack["error"] in [
                        "Missing data for control_strategy",
                        "No faults observed. P(error) = 0",
                    ], f"Bad error {attack['error']} in {os.path.join(root, file)}"
                    attack["treatment_strategies"] = []
        attacks += log

attack_id_length = {attack["attack_index"]: len(attack["attack"]) for attack in attacks}
original_attack_lengths = sorted(list(set(attack_id_length.values())))
attack_ids = sorted(list(attack_id_length.keys()))


sample_sizes = sorted(list(set(attack["sample_size"] for attack in attacks)))
ci_alphas = sorted(list(set(attack["ci_alpha"] for attack in attacks)))


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

l_greedy_attack_lengths = {k: [] for k in attack_ids}
l_greedy_attack_lengths_combinatorial = {k: [] for k in attack_ids}
l_our_attack_lengths = {k: [] for k in attack_ids}
l_our_attack_lengths_combinatorial = {k: [] for k in attack_ids}
for attack in attacks:
    l_greedy_attack_lengths[attack["attack_index"]].append(len(attack["greedy_minimal"]))
    l_greedy_attack_lengths_combinatorial[attack["attack_index"]].append(len(attack["minimal"]))
    l_our_attack_lengths[attack["attack_index"]].append(len(attack["extended_interventions"]))
    if len(attack["attack"]) < 20:
        l_our_attack_lengths_combinatorial[attack["attack_index"]].append(
            len(attack["minimised_extended_interventions"])
        )

fig = plt.figure(figsize=(18, 8))
gs = gridspec.GridSpec(3, 5)
axs = {0: [], 1: [], 2: []}

positions = {
    1: (0, 0, 1),
    2: (0, 1, 1),
    3: (0, 2, 1),
    4: (0, 3, 2),
    5: (1, 0, 2),
    6: (1, 2, 1),
    7: (1, 3, 1),
    8: (1, 4, 1),
    9: (2, 0, 1),
    10: (2, 1, 1),
    11: (2, 2, 1),
    13: (2, 3, 1),
    24: (2, 4, 1),
}

i = 0
for length in original_attack_lengths:
    row, col, size = positions[length]

    inx = gs[i]
    if length in [4, 5]:
        inx = gs[i : i + size]
    ax = fig.add_subplot(inx, sharey=axs[row][0] if len(axs[row]) > 0 else None)
    axs[row].append(ax)
    i += size

    selected_attack_ids = [
        attack_id
        for attack_id, attack_length in sorted(list(attack_id_length.items()), key=lambda x: list(reversed(x)))
        if attack_length == length
    ]

    plot_grouped_boxplot(
        [
            [l_greedy_attack_lengths[a] for a in selected_attack_ids],
            [l_greedy_attack_lengths_combinatorial[a] for a in selected_attack_ids],
            [l_our_attack_lengths[a] for a in selected_attack_ids],
            [l_our_attack_lengths_combinatorial[a] for a in selected_attack_ids],
        ],
        ax=ax,
        title=f"Original length {length}",
        labels=[BASELINE, f"{BASELINE} (optimal)", TOOLNAME, f"{TOOLNAME} (optimal)"] if length == 1 else None,
        colours=[RED, BLUE, GREEN, MAGENTA],
        markers=["x", "o", "s", 2],
        xticklabels=selected_attack_ids,
        xlabel="Attack ID",
        ylabel="Tool-minimised trace length" if length in [1, 5, 9] else None,
    )
    if ax != axs[row][0]:
        ax.tick_params(labelleft=False)


fig.suptitle("Pruned Trace Lengths")
plt.tight_layout()
plt.savefig(f"{figures}/rq1-attack-lengths-per-trace.png")
plt.clf()

# (1c) Measure the length of the "tool-minimised" traces, comparing to length of original
# Show each trace separately

fig, axs = plt.subplots(3, 3, figsize=(18, 8))

for sample, ax in zip(sample_sizes, axs.reshape(-1)):
    sampled_attacks = list(filter(lambda attack: attack["sample_size"] == sample, attacks))

    plot_grouped_boxplot(
        [
            [
                [len(a["greedy_minimal"]) for a in sampled_attacks if len(a["attack"]) == l]
                for l in original_attack_lengths
            ],
            [[len(a["minimal"]) for a in sampled_attacks if len(a["attack"]) == l] for l in original_attack_lengths],
            [
                [len(a["extended_interventions"]) for a in sampled_attacks if len(a["attack"]) == l]
                for l in original_attack_lengths
            ],
            [
                (
                    [len(a["minimised_extended_interventions"]) for a in sampled_attacks if len(a["attack"]) == l]
                    if l < 20
                    else []
                )
                for l in original_attack_lengths
            ],
        ],
        ax=ax,
        title=f"Sample {sample}",
        labels=[BASELINE, f"{BASELINE} (optimal)", TOOLNAME, f"{TOOLNAME} (optimal)"] if sample == 1 else None,
        colours=[RED, BLUE, GREEN, MAGENTA],
        markers=["x", "o", "s", 2],
        xticklabels=original_attack_lengths,
        xlabel="Attack length",
        ylabel="Tool-minimised trace length",
    )


fig.suptitle("Pruned Trace Lengths")
plt.tight_layout()
plt.savefig(f"{figures}/rq1-attack-lengths-per-sample.png")
plt.clf()


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
