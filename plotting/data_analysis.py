"""
This module processes the causal test logs and draws the figures.
"""

import sys
import os
import json
import re
from itertools import combinations
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
from scipy.stats import mannwhitneyu
from cliffs_delta import cliffs_delta
from collections import Counter

from constants import BASELINE, TOOLNAME, RED, GREEN, BLUE, MAGENTA, GOLD_STANDARD, RANGE_1
from grouped_boxplot import plot_grouped_boxplot

plt.style.use("ggplot")

# Setup IO
if len(sys.argv) != 2:
    raise ValueError("Please provide the directory of the log files, e.g. case-2-oref0/logs")
logs = sys.argv[1]
figures = sys.argv[1].replace("/logs", "/figures")
stats_dir = sys.argv[1].replace("/logs", "/stats")

if not os.path.exists(figures):
    os.mkdir(figures)
if not os.path.exists(stats_dir):
    os.mkdir(stats_dir)

# Read in data and convert to dataframe
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
            for trace in [
                "greedy_minimal",
                "minimal",
                "extended_interventions",
                # "minimised_extended_interventions",
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
df["attack"] = df["attack"].apply(lambda a: tuple(map(tuple, a)))

lengths = [
    (len(attack), len_minimal)
    for attack, len_minimal in df[["attack", "minimal"]].groupby(["attack", "minimal"]).groups.keys()
]

minimised_lengths = {}
for original_length, minimised_length in lengths:
    minimised_lengths[minimised_length] = sorted(minimised_lengths.get(minimised_length, []) + [original_length])
minimised_lengths = {k: dict(Counter(v)) for k, v in minimised_lengths.items()}
pd.Series(minimised_lengths).sort_index().to_latex(f"{stats_dir}/attack_lengths.tex")

assert False

# Add extra columns
df["greedy_executions"] = df["original_length"]
df["greedy_executions_per_event"] = df["greedy_executions"] / df["original_length"]
df["simulator_runs_per_event"] = df["simulator_runs"] / df["original_length"]
df["reduced_simulator_runs_per_event"] = df["reduced_simulator_runs"] / df["original_length"]

df["minimal_per_event"] = df["minimal"] / df["original_length"]
df["greedy_minimal_per_event"] = df["greedy_minimal"] / df["original_length"]
df["extended_interventions_per_event"] = df["extended_interventions"] / df["original_length"]
df["reduced_extended_interventions_per_event"] = df["reduced_extended_interventions"] / df["original_length"]


def calculate_percentage_reduction(data):
    return pd.DataFrame(
        {
            col: ((data["original_length"] - data[col]) / data["original_length"]) * 100
            for col in ["greedy_minimal", "minimal", "extended_interventions", "reduced_extended_interventions"]
        }
    ).replace(float("inf"), 100)


def calculate_percentage_removed(data):
    return pd.DataFrame(
        {
            col: ((data["original_length"] - data[col]) / (data["original_length"] - data["minimal"])) * 100
            for col in ["greedy_minimal", "minimal", "extended_interventions", "reduced_extended_interventions"]
        }
    ).replace(np.nan, 100)


percentage_reduction = calculate_percentage_reduction(df)
percentage_removed = calculate_percentage_removed(df)

pd.DataFrame(
    {
        "Mean": [f"{mean:.2f}" for mean in percentage_reduction.median()],
        "Median": [f"{median:.2f}" for median in percentage_reduction.mean()],
    },
    index=[BASELINE, GOLD_STANDARD, TOOLNAME, f"{TOOLNAME} + {BASELINE}"],
).to_csv(f"{stats_dir}/percentage_reduction.tex")

pd.DataFrame(
    {
        "Mean": [f"{mean:.2f}" for mean in percentage_removed.median()],
        "Median": [f"{median:.2f}" for median in percentage_removed.mean()],
    },
    index=[BASELINE, GOLD_STANDARD, TOOLNAME, f"{TOOLNAME} + {BASELINE}"],
).to_csv(f"{stats_dir}/percentage_removed.tex")

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
greedy_attack_lengths = list(df.groupby("original_length")["greedy_minimal_per_event"].apply(list))
greedy_attack_lengths_combinatorial = list(df.groupby("original_length")["minimal_per_event"].apply(list))
our_attack_lengths = list(df.groupby("original_length")["extended_interventions_per_event"].apply(list))
our_greedy_attack_lengths = list(df.groupby("original_length")["reduced_extended_interventions_per_event"].apply(list))


fig, ax = plt.subplots()
plot_grouped_boxplot(
    [
        list(df.groupby("sample_size")["greedy_minimal_per_event"].apply(list)),
        list(df.groupby("sample_size")["extended_interventions_per_event"].apply(list)),
        list(df.groupby("sample_size")["reduced_extended_interventions_per_event"].apply(list)),
    ],
    labels=[BASELINE, TOOLNAME, f"{TOOLNAME} + {BASELINE}"],
    colours=[RED, GREEN, MAGENTA],
    markers=["x", "s", 2],
    ax=ax,
    xticklabels=sample_sizes,
    xlabel="Number of executions",
    ylabel="Tool-minimised trace length\n(normalised by original length)",
)
ax.legend(loc="upper left")
plt.savefig(f"{figures}/rq1-attack-lengths-by-data-size.pgf", bbox_inches="tight", pad_inches=0)
plt.clf()

spurious_events = df["original_length"] - df["minimal"]

our_prune = df["original_length"] - df["extended_interventions"]
our_greedy_prune = df["original_length"] - df["reduced_extended_interventions"]

print("Fraction of spurius events removed")
print("our_prune", (our_prune / spurious_events).mean(), (our_prune / spurious_events).median())
print("our_greedy_prune", (our_greedy_prune / spurious_events).mean(), (our_greedy_prune / spurious_events).median())

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

plt.savefig(f"{figures}/rq1-attack-lengths.pgf", bbox_inches="tight", pad_inches=0)
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


def plot_grouped(our_groups, greedy_groups, xlabel, ylabel, fname, cc_alternative="greater", ccg_alternative="greater"):
    _, [ax1, ax2] = plt.subplots(2, 1, figsize=(16, 6), sharex=True)

    causal_cut_labels, causal_cut_data, causal_cut_greedy_labels, causal_cut_greedy_data = zip(*our_groups)

    stats = []
    for g1, g2 in combinations(our_groups, 2):
        ccl1, ccd1, ccgl1, ccgd1 = g1
        ccl2, ccd2, ccgl2, ccgd2 = g2
        ccd1_overall = np.array([x for xs in ccd1 for x in xs])
        ccd2_overall = np.array([x for xs in ccd2 for x in xs])
        ccgd1_overall = np.array([x for xs in ccgd1 for x in xs])
        ccgd2_overall = np.array([x for xs in ccgd2 for x in xs])

        cc_results = {
            l: mannwhitneyu(x, y, alternative=cc_alternative).pvalue
            for l, x, y in zip(original_attack_lengths, ccd1, ccd2)
        }
        cc_results["overall"] = mannwhitneyu(ccd1_overall, ccd2_overall, alternative=cc_alternative).pvalue
        cc_results["overall_effect"], cc_results["overall_effect_class"] = cliffs_delta(ccd1_overall, ccd2_overall)
        ccg_results = {
            l: mannwhitneyu(x, y, alternative=ccg_alternative).pvalue
            for l, x, y in zip(original_attack_lengths, ccgd1, ccgd2)
        }
        ccg_results["overall"] = mannwhitneyu(ccgd1_overall, ccgd2_overall, alternative=ccg_alternative).pvalue
        ccg_results["overall_effect"], ccg_results["overall_effect_class"] = cliffs_delta(ccgd1_overall, ccgd2_overall)
        stats.append(
            {
                "System": "CausalCut",
                "Pair": ccl1[10:] + ("<" if cc_alternative == "less" else ">") + ccl2[10:],
            }
            | cc_results
        )
        stats.append(
            {
                "System": "CausalCut+Greedy",
                "Pair": ccgl1[29:] + ("<" if ccg_alternative == "less" else ">") + ccgl2[29:],
            }
            | ccg_results
        )
    stats = pd.DataFrame(stats)
    if len(causal_cut_data) == 4:
        by = "data-sample"
    else:
        by = "confidence-intervals"
    stats.T.to_latex(os.path.join(stats_dir, ylabel.lower().replace(" ", "-") + "-by-" + by + ".tex"), escape=True)

    baseline_labels = [BASELINE]
    baseline_colours = [RED]
    if len(greedy_groups) == 2:
        baseline_labels.append(GOLD_STANDARD)
        baseline_colours.append(BLUE)

    plot_grouped_boxplot(
        greedy_groups + list(causal_cut_data),
        ax=ax1,
        labels=baseline_labels + list(causal_cut_labels),
        colours=baseline_colours + RANGE_1[: len(causal_cut_data)],
        xticklabels=original_attack_lengths,
        ylabel=ylabel,
        position_offsets=POSITION_OFFSETS,
    )
    plot_grouped_boxplot(
        greedy_groups + list(causal_cut_greedy_data),
        ax=ax2,
        labels=baseline_labels + list(causal_cut_greedy_labels),
        colours=baseline_colours + RANGE_1[: len(causal_cut_data)],
        xticklabels=original_attack_lengths,
        xlabel=xlabel,
        ylabel=ylabel,
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
        ax1.plot([0.78, 0.91], [0, 0], transform=ax.transAxes, **kwargs)
        ax1.plot([0.79, 0.92], [0, 0], transform=ax.transAxes, **kwargs)
        ax2.plot([0.78, 0.91], [0, 0], transform=ax.transAxes, **kwargs)
        ax2.plot([0.79, 0.92], [0, 0], transform=ax.transAxes, **kwargs)

    plt.savefig(os.path.join(figures, fname), bbox_inches="tight", pad_inches=0)
    plt.clf()


sample_sizes = [50, 500, 1000, 5000] if "case-2" in logs else [500, 1000, 5000, 449920]
confidence_intervals = [80, 90]

# (1b) Separate out amounts of test data
plot_grouped(
    [
        (
            f"{TOOLNAME} ({sample_size} test runs)",
            list(
                df.loc[df.sample_size == sample_size].groupby("original_length")["extended_interventions"].apply(list)
            ),
            f"{TOOLNAME} + {BASELINE} ({sample_size} test runs)",
            list(
                df.loc[df.sample_size == sample_size]
                .groupby("original_length")["reduced_extended_interventions"]
                .apply(list)
            ),
        )
        for sample_size in sample_sizes
    ],
    [greedy_attack_lengths, greedy_attack_lengths_combinatorial],
    "Original trace length",
    "Tool-minimised trace length",
    "rq1-attack-lengths-by-data-size-and-length.pgf",
)


# Define a function to apply the formatting
def format_and_bold(val):
    # Check if the value is a number (int or float)
    if isinstance(val, (int, float)):
        # If it's a number, format it and apply the bolding logic
        if val < 0.05:
            return f"\\textbf{{{val:.3f}}}"
        else:
            return f"{val:.3f}"
    else:
        # If it's not a number (e.g., a string or list),
        # just return the value as a string to avoid errors.
        return str(val)


def test_u_shape(causal_cut_col, causal_cut_plus_col, outfile, sizes=None):
    if sizes is None:
        sizes = combinations(sample_sizes, 2)
    test_sample_sizes = {}
    for x, y in sizes:
        test_sample_sizes[f"{x} > {y}"] = {
            "causal_cut_longer": mannwhitneyu(
                df.loc[df.sample_size == x, causal_cut_col],
                df.loc[df.sample_size == y, causal_cut_col],
                alternative="greater",
            ).pvalue,
            "causal_cut_shorter": mannwhitneyu(
                df.loc[df.sample_size == x, causal_cut_col],
                df.loc[df.sample_size == y, causal_cut_col],
                alternative="less",
            ).pvalue,
            "causal_cut+_longer": mannwhitneyu(
                df.loc[df.sample_size == x, causal_cut_plus_col],
                df.loc[df.sample_size == y, causal_cut_plus_col],
                alternative="greater",
            ).pvalue,
            "causal_cut+_shorter": mannwhitneyu(
                df.loc[df.sample_size == x, causal_cut_plus_col],
                df.loc[df.sample_size == y, causal_cut_plus_col],
                alternative="less",
            ).pvalue,
        }
    pd.DataFrame(test_sample_sizes).round(3).map(format_and_bold).to_latex(f"{stats_dir}/{outfile}.tex")


test_u_shape("extended_interventions", "reduced_extended_interventions", "minimisation_by_sample_size")
test_u_shape(
    "extended_interventions",
    "reduced_extended_interventions",
    "rq1-minimisation_by_sample_size",
    sizes=[(50, 100), (100, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 5000)],
)


# 1c. Group by confidence intervals
plot_grouped(
    [
        (
            f"{TOOLNAME} ({confidence_interval}% confidence)",
            list(
                df.loc[df.ci_alpha == confidence_interval]
                .groupby("original_length")["extended_interventions"]
                .apply(list)
            ),
            f"{TOOLNAME} + {BASELINE} ({confidence_interval}% confidence)",
            list(
                df.loc[df.ci_alpha == confidence_interval]
                .groupby("original_length")["reduced_extended_interventions"]
                .apply(list)
            ),
        )
        for confidence_interval in confidence_intervals
    ],
    [greedy_attack_lengths, greedy_attack_lengths_combinatorial],
    "Original trace length",
    "Tool-minimised trace length",
    "rq1-attack-lengths-by-confidence.pgf",
    cc_alternative="greater",
    ccg_alternative="less",
)

# RQ2: Baseline - minimal traces produced by Poskitt [2023]
# Measure number of executions required from simulator / CPS.
our_executions = df.groupby("original_length")["simulator_runs"].apply(list)
our_executions_extra = df.groupby("original_length")["reduced_simulator_runs"].apply(list)
greedy_executions = df.groupby("original_length")["greedy_executions"].apply(list)

pd.concat(
    [
        df.groupby("original_length")[["greedy_executions", "simulator_runs", "reduced_simulator_runs"]]
        .apply(lambda x: x.mean())
        .rename(
            {
                "simulator_runs": "causal_cut",
                "reduced_simulator_runs": "causal_cut_plus",
                "greedy_executions": "greedy",
            },
            axis=1,
        ),
        pd.DataFrame(
            {
                "greedy": {"mean": df["greedy_executions"].mean(), "median": df["greedy_executions"].median()},
                "causal_cut": {"mean": df["simulator_runs"].mean(), "median": df["simulator_runs"].median()},
                "causal_cut_plus": {
                    "mean": df["reduced_simulator_runs"].mean(),
                    "median": df["reduced_simulator_runs"].median(),
                },
            }
        ),
    ]
).T.round(3).to_latex(f"{stats_dir}/simulator_runs.tex")

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
    ylabel="Simulation runs",
    position_offsets=POSITION_OFFSETS,
)

# Greedy fit
ax.plot(
    ax.get_xticks()[:11] - 0.5,
    df.groupby("original_length")["greedy_executions"].apply(np.median)[:11],
    color=RED,
    alpha=0.5,
)
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
plt.savefig(f"{figures}/rq2-simulator-executions.pgf", bbox_inches="tight", pad_inches=0)
plt.clf()

# 2b. Group by data size
plot_grouped(
    [
        (
            f"{TOOLNAME} ({sample_size} test runs)",
            list(df.loc[df.sample_size == sample_size].groupby("original_length")["simulator_runs"].apply(list)),
            f"{TOOLNAME} + {BASELINE} ({sample_size} test runs)",
            list(
                df.loc[df.sample_size == sample_size].groupby("original_length")["reduced_simulator_runs"].apply(list)
            ),
        )
        for sample_size in sample_sizes
    ],
    [greedy_executions],
    "Original trace length",
    "Simulation runs",
    "rq2-simulator-executions-by-sample-size.pgf",
)

test_u_shape("simulator_runs", "reduced_simulator_runs", "simulator_runs_by_sample_size")


# 2c. Group by confidence
plot_grouped(
    [
        (
            f"{TOOLNAME} ({confidence_interval}% confidence)",
            list(df.loc[df.ci_alpha == confidence_interval].groupby("original_length")["simulator_runs"].apply(list)),
            f"{TOOLNAME} + {BASELINE} ({confidence_interval}% confidence)",
            list(
                df.loc[df.ci_alpha == confidence_interval]
                .groupby("original_length")["reduced_simulator_runs"]
                .apply(list)
            ),
        )
        for confidence_interval in confidence_intervals
    ],
    [greedy_executions],
    "Original trace length",
    "Simulation runs",
    "rq2-simulator-executions-by-confidence.pgf",
    cc_alternative="less",
    ccg_alternative="greater",
)
