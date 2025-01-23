"""
This module processes the causal test logs and draws the figures for openAPS.
"""

import sys
import os
import json
import re
import matplotlib.pyplot as plt
from matplotlib import gridspec
from math import ceil
import pandas as pd
from matplotlib.lines import Line2D

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
        sample_size = re.search(r"sample_(\w+)", root).group(1)
        sample_size = int(sample_size) if sample_size != "None" else 449920
        ci_alpha = int(re.search(r"ci_(\w+)", root).group(1))
        for attack in log:
            assert "greedy_minimal" in attack, f"No greedy_minimal in {os.path.join(root, file)}"
            attack["sample_size"] = sample_size
            attack["ci_alpha"] = ci_alpha
            attack["original_length"] = len(attack["attack"])
            for trace in ["greedy_minimal", "minimal", "extended_interventions", "minimised_extended_interventions"]:
                if trace in attack:
                    attack[trace] = len(attack[trace])
            if "error" in attack:
                if "treatment_strategies" not in attack:
                    assert attack["error"] in [
                        "Missing data for control_strategy",
                        "No faults observed. P(error) = 0",
                    ], f"Bad error {attack['error']} in {os.path.join(root, file)}"
                    attack["treatment_strategies"] = []
        attacks += log

df = pd.DataFrame(attacks)

attack_id_length = pd.Series(df.original_length.values, index=df.attack_index).to_dict()
original_attack_lengths = sorted(list(set(df.original_length)))
attack_ids = sorted(list(set(df.attack_index)))
sample_sizes = sorted(list(set(df.sample_size)))
ci_alphas = sorted(list(set(df.ci_alpha)))

# RQ1: Baseline - minimal traces produced by Poskitt [2023]
# (1a) Measure the length of the "tool-minimised" traces, comparing to length of original
# Group by trace length
greedy_attack_lengths = list(df.groupby("original_length")["greedy_minimal"].apply(list))
greedy_attack_lengths_combinatorial = list(df.groupby("original_length")["minimal"].apply(list))
our_attack_lengths = list(df.groupby("original_length")["extended_interventions"].apply(list))
our_attack_lengths_combinatorial = list(
    df.groupby("original_length")["minimised_extended_interventions"].apply(
        # We can't feasibly minimise attacks of length greater than 20 as there's over 16M combinations (16,777,215)
        lambda group: list(group) if group.name < 20 else []
    )
)

plot_grouped_boxplot(
    [greedy_attack_lengths, greedy_attack_lengths_combinatorial, our_attack_lengths, our_attack_lengths_combinatorial],
    savepath=f"{figures}/rq1-attack-lengths.pgf",
    labels=[BASELINE, f"{BASELINE} (optimal)", TOOLNAME, f"{TOOLNAME} (optimal)"],
    colours=[RED, BLUE, GREEN, MAGENTA],
    markers=["x", "o", "s", 2],
    # title="Pruned Trace Lengths",
    xticklabels=original_attack_lengths,
    xlabel="Original trace length",
    ylabel="Tool-minimised trace length",
)

# (1b) Measure the length of the "tool-minimised" traces, comparing to length of original
# Show each trace separately

if "case-1" in logs:
    sizes = {
        1: 2,
        2: 4,
        3: 4,
        4: 4,
        5: 3,
        6: 3,
        7: 2,
        8: 2,
        9: 2,
        10: 1,
        11: 1,
        12: 2,
    }
    PER_TRACE_COLS = 4
elif "case-2" in logs:
    sizes = {
        1: 2,
        2: 2,
        3: 2,
        4: 4,
        5: 4,
        6: 2,
        7: 2,
        8: 2,
        9: 2,
        10: 2,
        11: 2,
        13: 2,
        24: 2,
    }
    PER_TRACE_COLS = 5
else:
    raise ValueError(f"Could not initialise positions for {logs}")

COLUMNS = 8
ROWS = ceil(sum(sizes.values()) / COLUMNS)
if ROWS * COLUMNS <= sum(sizes.values()):
    ROWS += 1

fig = plt.figure(figsize=(16, 2 * ROWS))
gs = gridspec.GridSpec(ROWS, COLUMNS)
gs.update(wspace=0.1, hspace=0.8)
axs = {r: [] for r in range(ROWS)}

start = 2
end = 2
for length, size in sizes.items():
    end += size
    inx = gs[start:end]
    row = start // COLUMNS
    col = start % COLUMNS
    ax = fig.add_subplot(inx, sharey=axs[row][0] if len(axs[row]) > 0 else None)
    axs[row].append(ax)
    start += size

    plot_grouped_boxplot(
        [
            df.loc[df["original_length"] == length].groupby(["attack_index"])["greedy_minimal"].apply(list),
            df.loc[df["original_length"] == length].groupby(["attack_index"])["minimal"].apply(list),
            df.loc[df["original_length"] == length].groupby(["attack_index"])["extended_interventions"].apply(list),
            df.loc[df["original_length"] == length]
            .groupby(["attack_index"])["minimised_extended_interventions"]
            .apply(
                # We can't feasibly minimise attacks of length greater than 20 as there's over 16M combinations (16,777,215)
                lambda group: list(group) if length < 20 else []
            ),
        ],
        ax=ax,
        title=f"Original length {length}",
        colours=[RED, BLUE, GREEN, MAGENTA],
        markers=["x", "o", "s", 2],
        xticklabels=df.loc[df["original_length"] == length].groupby(["attack_index"]).groups.keys(),
        yticklabels=range(0, 11, 2) if "case-1" in logs else range(0, 21, 4),
        xlabel="Attack ID",
        # ylabel="Tool-minimised\ntrace length" if len(axs[row]) == 1 else None,
    )
    if len(axs[row]) > 1:
        ax.tick_params(labelleft=False)

# fig.align_ylabels()

fig.add_subplot(111, frameon=False)
plt.tick_params(
    labelcolor="none",
    which="both",
    top=False,
    bottom=False,
    left=False,
    right=False,
    labelleft=False,
    labelbottom=False,
)
plt.xticks([])
plt.yticks([])
plt.ylabel("Tool-minimised trace length", labelpad=20)


colours = [RED, BLUE, GREEN, MAGENTA]
lines = [Line2D([0], [0], color=c) for c in colours]
labels = labels = [BASELINE, f"{BASELINE} (optimal)", TOOLNAME, f"{TOOLNAME} (optimal)"]
axs[0][0].legend(lines, labels, bbox_to_anchor=(-1, 1), loc="upper left")

plt.savefig(f"{figures}/rq1-attack-lengths-per-trace.pgf", bbox_inches="tight", pad_inches=0)
plt.clf()

# (1c) Measure the length of the "tool-minimised" traces, comparing to length of original
# Show each trace separately

fig, axs = plt.subplots(3, PER_TRACE_COLS, figsize=(16, 8), sharey="row")

# I suggest we drop original_length==1 out of this since these can't be pruned any more
for i, length in enumerate(original_attack_lengths):
    row = i // PER_TRACE_COLS
    col = i % PER_TRACE_COLS
    plot_grouped_boxplot(
        [
            df.loc[df["original_length"] == length].groupby(["sample_size"])["greedy_minimal"].apply(list),
            df.loc[df["original_length"] == length].groupby(["sample_size"])["minimal"].apply(list),
            df.loc[df["original_length"] == length].groupby(["sample_size"])["extended_interventions"].apply(list),
            df.loc[df["original_length"] == length]
            .groupby(["sample_size"])["minimised_extended_interventions"]
            .apply(
                # We can't feasibly minimise attacks of length greater than 20 as there's over 16M combinations (16,777,215)
                lambda group: list(group) if length < 20 else []
            ),
        ],
        ax=axs[row][col],
        title=f"Original length {length}",
        labels=[BASELINE, f"{BASELINE} (optimal)", TOOLNAME, f"{TOOLNAME} (optimal)"] if length == 1 else None,
        colours=[RED, BLUE, GREEN, MAGENTA],
        markers=["x", "o", "s", 2],
        xticklabels=df.loc[df["original_length"] == length].groupby(["sample_size"]).groups.keys(),
        xlabel="Sample size",
        ylabel="Tool-minimised trace length" if col == 0 else None,
    )
    ax.tick_params(labelleft=col == 0)

col += 1
for col in range(col, PER_TRACE_COLS):
    fig.delaxes(axs[row][col])
fig.align_ylabels()
plt.tight_layout()
plt.savefig(f"{figures}/rq1-attack-lengths-per-sample.pgf", bbox_inches="tight", pad_inches=0)
plt.clf()


# (2) Measure the proportion of the "tool-minimised" traces that are spurious. Report as the average proportion again.
# (a) aggregate
df["greedy_spurious"] = (df["greedy_minimal"] - df["minimal"]) / df["original_length"]
df["our_spurious"] = (df["extended_interventions"] - df["minimised_extended_interventions"]) / df["original_length"]
greedy_spurious = df.groupby("original_length")["greedy_spurious"].apply(list)
our_spurious = df.groupby("original_length")["our_spurious"].apply(list)

plot_grouped_boxplot(
    [greedy_spurious, our_spurious],
    savepath=f"{figures}/rq1-proportion-spurious.pgf",
    labels=[BASELINE, TOOLNAME],
    colours=[RED, GREEN],
    markers=["x", "s"],
    # title="Spurious Events",
    xticklabels=original_attack_lengths,
    xlabel="Original trace length",
    ylabel="Proportion of Remaining Spurious Events",
)

# (b) group by trace id
fig = plt.figure(figsize=(16, 2 * ROWS))
gs = gridspec.GridSpec(ROWS, COLUMNS)
gs.update(wspace=0.1, hspace=0.8)
axs = {r: [] for r in range(ROWS)}

start = 2
end = 2
for length, size in sizes.items():
    end += size
    inx = gs[start:end]
    row = start // COLUMNS
    col = start % COLUMNS
    ax = fig.add_subplot(inx, sharey=axs[row][0] if len(axs[row]) > 0 else None)
    axs[row].append(ax)
    start += size

    plot_grouped_boxplot(
        [
            df.loc[df["original_length"] == length].groupby(["attack_index"])["greedy_spurious"].apply(list),
            df.loc[df["original_length"] == length]
            .groupby(["attack_index"])["our_spurious"]
            .apply(
                # We can't feasibly minimise attacks of length greater than 20 as there's over 16M combinations (16,777,215)
                lambda group: list(group) if length < 20 else []
            ),
        ],
        ax=ax,
        title=f"Original length {length}",
        colours=[RED, GREEN],
        xticklabels=df.loc[df["original_length"] == length].groupby(["attack_index"]).groups.keys(),
        yticklabels=[x / 10 for x in range(0, 10, 2)],
        xlabel="Attack ID",
    )
    if len(axs[row]) > 1:
        ax.tick_params(labelleft=False)

fig.add_subplot(111, frameon=False)
plt.tick_params(
    labelcolor="none",
    which="both",
    top=False,
    bottom=False,
    left=False,
    right=False,
    labelleft=False,
    labelbottom=False,
)
plt.xticks([])
plt.yticks([])
plt.ylabel("Proportion of Remaining Spurious Events")
fig.align_ylabels()

colours = [RED, GREEN]
lines = [Line2D([0], [0], color=c) for c in colours]
labels = [BASELINE, TOOLNAME]
axs[0][0].legend(lines, labels, bbox_to_anchor=(-0.9, 1), loc="upper left")

plt.savefig(f"{figures}/rq1-proportion-spurious-per-trace.pgf", bbox_inches="tight", pad_inches=0)
plt.clf()


# (c) group by sample size
# (1c) Measure the length of the "tool-minimised" traces, comparing to length of original
# Show each trace separately

fig, axs = plt.subplots(3, PER_TRACE_COLS, figsize=(16, 8), sharey="row")

# I suggest we drop original_length==1 out of this since these can't be pruned any more
for i, length in enumerate(original_attack_lengths):
    row = i // PER_TRACE_COLS
    col = i % PER_TRACE_COLS
    plot_grouped_boxplot(
        [
            df.loc[df["original_length"] == length].groupby(["sample_size"])["greedy_spurious"].apply(list),
            df.loc[df["original_length"] == length]
            .groupby(["sample_size"])["our_spurious"]
            .apply(
                # We can't feasibly minimise attacks of length greater than 20 as there's over 16M combinations (16,777,215)
                lambda group: list(group) if length < 20 else []
            ),
        ],
        ax=axs[row][col],
        title=f"Original length {length}",
        labels=[BASELINE, TOOLNAME] if length == 1 else None,
        colours=[RED, GREEN],
        xticklabels=df.loc[df["original_length"] == length].groupby(["sample_size"]).groups.keys(),
        xlabel="Sample size",
        ylabel="Tool-minimised trace length" if col == 0 else None,
    )
    ax.tick_params(labelleft=col == 0)

col += 1
for col in range(col, PER_TRACE_COLS):
    fig.delaxes(axs[row][col])
fig.align_ylabels()
# fig.suptitle("Spurious events")
plt.tight_layout()
plt.savefig(f"{figures}/rq1-proportion-spurious-per-sample.pgf", bbox_inches="tight", pad_inches=0)
plt.clf()


# RQ2: Baseline - minimal traces produced by Poskitt [2023]
# Measure number of executions required from simulator / CPS.
our_executions = df.groupby("original_length")["simulator_runs"].apply(list)
plot_grouped_boxplot(
    [[[l] for l in original_attack_lengths], our_executions],
    savepath=f"{figures}/rq2-simulator-executions.pgf",
    labels=[BASELINE, TOOLNAME],
    colours=[RED, GREEN],
    markers=["x", "s"],
    # title="Simulator Executions",
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
    # title="Remaining Spurious per Simulation",
    xlabel="Number of Simulations to Minimise the Trace",
    ylabel="Proportion of Remaining Spurious Events",
    savepath=f"{figures}/rq2-executions-spurious.pgf",
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
    # title="Pruning per Simulation",
    xlabel="Number of Simulations to Minimise the Trace",
    ylabel="Length of tool-minimised trace",
    savepath=f"{figures}/rq1-executions-pruning.pgf",
)


# RQ3:
# look into the impact of the different levels of data provision.
# our_adequacy = {
#     length: [
#         attack.get("result", {}).get("adequacy", {}).get("kurtosis", {}).get("trtrand", None)
#         for attack in attacks
#         if attack["original_length"] == length
#     ]
#     for length in original_attack_lengths
# }
