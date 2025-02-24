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
import numpy as np
from matplotlib.lines import Line2D
from statsmodels.formula.api import ols

from constants import BASELINE, TOOLNAME, RED, GREEN, BLUE, MAGENTA, GOLD_STANDARD
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
        sample_size = re.search(r"sample_(\w+)", root).group(1)
        sample_size = int(sample_size) if sample_size != "None" else 449920
        ci_alpha = int(re.search(r"ci_(\w+)", root).group(1))
        for attack in log:
            assert "greedy_minimal" in attack, f"No greedy_minimal in {os.path.join(root, file)}"
            attack["sample_size"] = sample_size
            attack["ci_alpha"] = ci_alpha
            attack["original_length"] = len(attack["attack"])
            # assert (
            #     len(attack["extended_interventions"]) < attack["original_length"]
            # ), f'Attack grew from {attack["original_length"]} to {len(attack["extended_interventions"])} in {os.path.join(root, file)}'
            for trace in [
                "greedy_minimal",
                "minimal",
                "extended_interventions",
                "minimised_extended_interventions",
                "reduced_extended_interventions",
            ]:
                if trace in attack:
                    attack[trace] = len(set(map(tuple, attack[trace])))
            if "error" in attack:
                if "treatment_strategies" not in attack:
                    assert attack["error"] in [
                        "Missing data for control_strategy",
                        "No faults observed. P(error) = 0",
                    ], f"Bad error {attack['error']} in {os.path.join(root, file)}"
                    attack["treatment_strategies"] = []
        attacks += log

df = pd.DataFrame(attacks)
df = df.loc[df["original_length"] > 1]

attack_id_length = pd.Series(df.original_length.values, index=df.attack_index).to_dict()
original_attack_lengths = sorted(list(set(df.original_length)))
attack_ids = sorted(list(set(df.attack_index)))
sample_sizes = sorted(list(set(df.sample_size)))
ci_alphas = sorted(list(set(df.ci_alpha)))

if "case-1" in logs:
    sizes = {
        # 1: 2,
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
    POSITION_OFFSETS = None
elif "case-2" in logs:
    sizes = {
        # 1: 2,
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
    POSITION_OFFSETS = ([0] * 10) + [3] + [6]
else:
    raise ValueError(f"Could not initialise positions for {logs}")

# RQ1: Baseline - minimal traces produced by Poskitt [2023]
# (1a) Measure the length of the "tool-minimised" traces, comparing to length of original
# Group by trace length
greedy_attack_lengths = list(df.groupby("original_length")["greedy_minimal"].apply(list))
greedy_attack_lengths_combinatorial = list(df.groupby("original_length")["minimal"].apply(list))
our_attack_lengths = list(df.groupby("original_length")["extended_interventions"].apply(list))
our_greedy_attack_lengths = list(df.groupby("original_length")["reduced_extended_interventions"].apply(list))

fig, ax = plt.subplots()

plot_grouped_boxplot(
    [greedy_attack_lengths, greedy_attack_lengths_combinatorial, our_attack_lengths, our_greedy_attack_lengths],
    ax=ax,
    labels=[BASELINE, GOLD_STANDARD, TOOLNAME, f"{TOOLNAME} + {BASELINE}"],
    colours=[RED, BLUE, GREEN, MAGENTA],
    markers=["x", "o", "s", 2],
    # title="Pruned Trace Lengths",
    xticklabels=original_attack_lengths,
    xlabel="Original trace length",
    ylabel="Tool-minimised trace length",
    position_offsets=POSITION_OFFSETS,
)

# Show the gap in data with a zigzag on x axis
if "case-2" in logs:
    kwargs = {
        "marker": "^",
        "markersize": 12,
        "linestyle": "none",
        "color": "w",
        "mec": "w",
        "mew": 1,
        "clip_on": False,
    }
    ax.plot([0.78, 0.91], [0, 0], transform=ax.transAxes, **kwargs)
    ax.plot([0.79, 0.92], [0, 0], transform=ax.transAxes, **kwargs)

plt.savefig(f"{figures}/rq1-attack-lengths.png", bbox_inches="tight", pad_inches=0)
plt.clf()

print("maximum_minimal_attack_length", np.max([np.max(x) for x in greedy_attack_lengths_combinatorial]))

print(
    "greedy_attack_lengths_optimal",
    [round(np.median(y) / np.median(x), 3) for x, y in zip(greedy_attack_lengths, greedy_attack_lengths_combinatorial)],
)
print(
    "greedy_attack_lengths_original",
    [round(np.median(x) / y, 3) for x, y in zip(greedy_attack_lengths, original_attack_lengths)],
)
print(
    "our_attack_lengths_original",
    [round(np.median(x) / y, 3) for x, y in zip(our_attack_lengths, original_attack_lengths)],
)


# (1b) Separate out amounts of test data
fig, ax = plt.subplots(figsize=(16, 6))

sample_sizes = [50, 500, 1000, 5000] if "case-2" in logs else [500, 1000, 5000, 449920]

our_attack_lengths = [
    (
        f"{TOOLNAME} ({sample_size} test runs)",
        list(df.loc[df.sample_size == sample_size].groupby("original_length")["extended_interventions"].apply(list)),
        f"{TOOLNAME} + {BASELINE} ({sample_size} test runs)",
        list(
            df.loc[df.sample_size == sample_size]
            .groupby("original_length")["reduced_extended_interventions"]
            .apply(list)
        ),
    )
    for sample_size in sample_sizes
]
greedy_attack_lengths = list(df.groupby("original_length")["greedy_minimal"].apply(list))
greedy_attack_lengths_combinatorial = list(df.groupby("original_length")["minimal"].apply(list))

causal_cut_labels, causal_cut_data, causal_cut_greedy_labels, causal_cut_greedy_data = zip(*our_attack_lengths)

plot_grouped_boxplot(
    [greedy_attack_lengths, greedy_attack_lengths_combinatorial] + list(causal_cut_data) + list(causal_cut_greedy_data),
    ax=ax,
    labels=[BASELINE, GOLD_STANDARD] + list(causal_cut_labels) + list(causal_cut_greedy_labels),
    colours=[RED, BLUE] + [GREEN] * len(causal_cut_data) + [MAGENTA] * len(causal_cut_greedy_data),
    # markers=["x", "o", "s", 2],
    # title=f"{sample_size} test runs",
    xticklabels=original_attack_lengths,
    xlabel="Original trace length" if ax.get_subplotspec().is_last_row() else None,
    ylabel="Tool-minimised trace length" if ax.get_subplotspec().is_first_col() else None,
    position_offsets=POSITION_OFFSETS,
)

# Show the gap in data with a zigzag on x axis
if "case-2" in logs:
    kwargs = {
        "marker": "^",
        "markersize": 12,
        "linestyle": "none",
        "color": "w",
        "mec": "w",
        "mew": 1,
        "clip_on": False,
    }
    ax.plot([0.78, 0.91], [0, 0], transform=ax.transAxes, **kwargs)
    ax.plot([0.79, 0.92], [0, 0], transform=ax.transAxes, **kwargs)

plt.savefig(f"{figures}/rq1-attack-lengths-by-data-size.png", bbox_inches="tight", pad_inches=0)
plt.clf()

# (2) Measure the proportion of the "tool-minimised" traces that are spurious. Report as the average proportion again.
# (a) aggregate
df["greedy_spurious"] = (df["greedy_minimal"] - df["minimal"]) / df["original_length"]
df["our_spurious"] = (df["extended_interventions"] - df["minimised_extended_interventions"]) / df["original_length"]
greedy_spurious = df.groupby("original_length")["greedy_spurious"].apply(list)
our_spurious = df.groupby("original_length")["our_spurious"].apply(list)

fig, ax = plt.subplots()
plot_grouped_boxplot(
    [greedy_spurious, our_spurious],
    ax=ax,
    labels=[BASELINE, TOOLNAME],
    colours=[RED, GREEN],
    markers=["x", "s"],
    # title="Spurious Events",
    xticklabels=original_attack_lengths,
    xlabel="Original trace length",
    ylabel="Proportion of Remaining Spurious Events",
)

plt.savefig(f"{figures}/rq1-proportion-spurious.pgf", bbox_inches="tight", pad_inches=0)
plt.clf()

# RQ2: Baseline - minimal traces produced by Poskitt [2023]
# Measure number of executions required from simulator / CPS.
our_executions = df.groupby("original_length")["simulator_runs"].apply(list)
our_executions_extra = df.groupby("original_length")["reduced_simulator_runs"].apply(list)
greedy_executions = [[l] for l in original_attack_lengths]
fig, ax = plt.subplots()
plot_grouped_boxplot(
    [greedy_executions, our_executions, our_executions_extra],
    ax=ax,
    labels=[BASELINE, TOOLNAME, f"{TOOLNAME} + {BASELINE}"],
    colours=[RED, GREEN, MAGENTA],
    markers=["x", "s", 2],
    # title="Simulator Executions",
    xticklabels=original_attack_lengths,
    xlabel="Original trace length",
    ylabel="Number of Simulations to Minimise the Trace",
    position_offsets=POSITION_OFFSETS,
)

# Greedy fit
ax.plot(ax.get_xticks()[:11] - 0.5, np.median(greedy_executions[:11], axis=1), color=RED, alpha=0.5)
# CausalCut fit
flat_data = [(x, y) for x, Y in zip(original_attack_lengths, our_executions) for y in Y]
x, y = zip(*flat_data)
model = ols(
    "executions ~ np.log(trace_length) + trace_length - 1",
    pd.DataFrame({"trace_length": x, "executions": y}),
).fit()
ax.plot(
    ax.get_xticks()[:11] + 0.5,
    model.predict(pd.DataFrame({"trace_length": original_attack_lengths[:11]})),
    color=GREEN,
    alpha=0.5,
)
# CausalCut + Greedy fit
flat_data = [(x, y) for x, Y in zip(original_attack_lengths, our_executions_extra) for y in Y]
x, y = zip(*flat_data)
model = ols(
    "executions ~ np.log(trace_length) + trace_length - 1",
    pd.DataFrame({"trace_length": x, "executions": y}),
).fit()
ax.plot(
    ax.get_xticks()[:11] + 0.5,
    model.predict(pd.DataFrame({"trace_length": original_attack_lengths[:11]})),
    color=MAGENTA,
    alpha=0.5,
)

if "case-2" in logs:
    kwargs = {
        "marker": "^",
        "markersize": 12,
        "linestyle": "none",
        "color": "w",
        "mec": "w",
        "mew": 1,
        "clip_on": False,
    }
    ax.plot([0.755, 0.905], [0, 0], transform=ax.transAxes, **kwargs)
    ax.plot([0.765, 0.915], [0, 0], transform=ax.transAxes, **kwargs)
plt.savefig(f"{figures}/rq2-simulator-executions.png", bbox_inches="tight", pad_inches=0)
plt.clf()

# 2b group by data size
fig, ax = plt.subplots(figsize=(16, 6))

assert (df["reduced_simulator_runs"] >= df["simulator_runs"]).all()

our_executions = [
    (
        f"{TOOLNAME} ({sample_size} test runs)",
        df.loc[df.sample_size == sample_size].groupby("original_length")["simulator_runs"].apply(list),
        f"{TOOLNAME} + {BASELINE} ({sample_size} test runs)",
        df.loc[df.sample_size == sample_size].groupby("original_length")["reduced_simulator_runs"].apply(list),
    )
    for sample_size in sample_sizes
]
greedy_executions = [[l] for l in original_attack_lengths]

causal_cut_labels, causal_cut_data, causal_cut_greedy_labels, causal_cut_greedy_data = zip(*our_executions)

plot_grouped_boxplot(
    [greedy_executions] + list(causal_cut_data) + list(causal_cut_greedy_data),
    ax=ax,
    labels=[BASELINE] + list(causal_cut_labels) + list(causal_cut_greedy_labels),
    colours=[RED] + [GREEN] * len(causal_cut_data) + [MAGENTA] * len(causal_cut_greedy_data),
    # markers=["x", "s", 2],
    # title=f"{sample_size} test runs",
    xticklabels=original_attack_lengths,
    xlabel="Original trace length" if ax.get_subplotspec().is_last_row() else None,
    ylabel="Simulator runs" if ax.get_subplotspec().is_first_col() else None,
    position_offsets=POSITION_OFFSETS,
    legend_args={"loc": "upper left"},
)

# Show the gap in data with a zigzag on x axis
if "case-2" in logs:
    kwargs = {
        "marker": "^",
        "markersize": 12,
        "linestyle": "none",
        "color": "w",
        "mec": "w",
        "mew": 1,
        "clip_on": False,
    }
    ax.plot([0.78, 0.91], [0, 0], transform=ax.transAxes, **kwargs)
    ax.plot([0.79, 0.92], [0, 0], transform=ax.transAxes, **kwargs)

plt.savefig(f"{figures}/rq2-simulator-executions-by-data-size.png", bbox_inches="tight", pad_inches=0)
plt.clf()
