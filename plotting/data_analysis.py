"""
This module processes the causal test logs and draws the figures.
"""

import sys
import os
import json
import re
from itertools import combinations
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
from scipy.stats import mannwhitneyu, spearmanr
from cliffs_delta import cliffs_delta

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
            for test in [
                "greedy_minimal",
                "minimal",
                "extended_interventions",
                "reduced_extended_interventions",
            ]:
                if test in attack:
                    attack[test] = len(set(map(tuple, attack[test])))
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
# Make the names easier to read
df.rename(
    {
        "simulator_runs": "causal_cut_executions",
        "reduced_simulator_runs": "causal_cut_plus_greedy_minimal_executions",
        "reduced_extended_interventions": "causal_cut_plus_greedy_minimal",
        "extended_interventions": "causal_cut",
    },
    axis=1,
    inplace=True,
)

# Add extra columns
df["greedy_minimal_executions"] = df["original_length"]
df["minimal_per_event"] = df["minimal"] / df["original_length"]

TECHNIQUES = ["greedy_minimal", "causal_cut", "causal_cut_plus_greedy_minimal"]

for technique in TECHNIQUES + ["minimal"]:
    if technique != "minimal":
        df[f"{technique}_executions_per_event"] = df[f"{technique}_executions"] / df["original_length"]
    df[f"{technique}_per_event"] = df[technique] / df["original_length"]
    df[f"{technique}_removed"] = df["original_length"] - df[technique]
    df[f"{technique}_removed_per_event"] = (df["original_length"] - df[technique]) / df["original_length"]
    df[f"{technique}_percentage_reduction"] = ((df["original_length"] - df[technique]) / df["original_length"]) * 100
    df[f"{technique}_percentage_removed"] = (
        (df["original_length"] - df[technique]) / (df["original_length"] - df["minimal"])
    ).replace(np.nan, 1) * 100


ORIGINAL_ATTACK_LENGTHS = sorted(list(set(df.original_length)))
POSITION_OFFSETS = ([0] * 10) + [3] + [6] if "case-2" in logs else None
SAMPLE_SIZES = sorted(list(set(df.sample_size)))
CONFIDENCE_INTERVALS = [80, 90]

# Original and gold standard attack lengths (Table 1)
lengths = [
    (len(attack), len_minimal)
    for attack, len_minimal in df[["attack", "minimal"]].groupby(["attack", "minimal"]).groups.keys()
]

minimised_lengths = {}
for original_length, minimised_length in lengths:
    minimised_lengths[minimised_length] = sorted(minimised_lengths.get(minimised_length, []) + [original_length])
minimised_lengths = {k: dict(Counter(v)) for k, v in minimised_lengths.items()}
minimised_lengths = pd.DataFrame(minimised_lengths).sort_index().T.fillna(0).astype(int).replace(0, "")
minimised_lengths.sort_index().to_latex(f"{stats_dir}/attack-lengths.tex")


def bold_if_significant(val):
    if isinstance(val, (int, float)):
        if val < 0.05:
            return f"\\textbf{{{round_format(val)}}}"
        return f"{round_format(val)}"
    return str(val)


def round_format(val):
    if isinstance(val, str):
        return val
    if val == 0:
        return "0"
    if abs(val) < 0.001:
        return f"{val:.2e}"
    return f"{val:.3f}"


# RQ1: Factors that influence pruning ability
pd.concat(
    [
        pd.concat(
            [
                df[[f"{technique}_percentage_reduction" for technique in TECHNIQUES + ["minimal"]]].median(),
                df[[f"{technique}_percentage_reduction" for technique in TECHNIQUES + ["minimal"]]].mean(),
            ],
            axis=1,
        )
        .reset_index(drop=True)
        .rename(dict(enumerate(TECHNIQUES + ["minimal"])))
        .rename({0: "Mean reduction", 1: "Median reduction"}, axis=1),
        pd.concat(
            [
                df[[f"{technique}_percentage_removed" for technique in TECHNIQUES + ["minimal"]]].median(),
                df[[f"{technique}_percentage_removed" for technique in TECHNIQUES + ["minimal"]]].mean(),
            ],
            axis=1,
        )
        .reset_index(drop=True)
        .rename(dict(enumerate(TECHNIQUES + ["minimal"])))
        .rename({0: "Mean spurious events removed", 1: "Median spurious events removed"}, axis=1),
    ],
    axis=1,
).to_latex(f"{stats_dir}/rq1-pruning.tex", float_format="%.2f")


# 1a. original test length
fig, ax = plt.subplots()

plot_grouped_boxplot(
    [
        list(df.groupby("original_length")["greedy_minimal_per_event"].apply(list)),
        list(df.groupby("original_length")["minimal_per_event"].apply(list)),
        list(df.groupby("original_length")["causal_cut_per_event"].apply(list)),
        list(df.groupby("original_length")["causal_cut_plus_greedy_minimal_per_event"].apply(list)),
    ],
    ax=ax,
    labels=[BASELINE, GOLD_STANDARD, TOOLNAME, f"{TOOLNAME} + {BASELINE}"],
    colours=[RED, BLUE, GREEN, MAGENTA],
    xticklabels=ORIGINAL_ATTACK_LENGTHS,
    xlabel="Original test length",
    ylabel="Tool-minimised test length\n(normalised by original length)",
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

plt.savefig(f"{figures}/rq1-test-length.pgf", bbox_inches="tight", pad_inches=0)
plt.clf()

# Correlation between original test length and pruned test length
test_length_stats = []
for technique in ["minimal", "greedy_minimal", "causal_cut", "causal_cut_plus_greedy_minimal"]:
    raw_stat, raw_p_value = spearmanr(df["original_length"], df[technique])
    normalised_stat, normalised_p_value = spearmanr(df["original_length"], df[technique + "_per_event"])
    test_length_stats.append(
        {
            "technique": technique,
            "raw_stat": raw_stat,
            "raw_p_value": raw_p_value,
            "normalised_stat": normalised_stat,
            "normalised_p_value": normalised_p_value,
        }
    )


pd.DataFrame(test_length_stats).map(round_format).to_latex(os.path.join(stats_dir, "rq1-test-length.tex"), index=False)


# 1b. data available
plot_grouped_boxplot(
    [
        list(df.groupby("sample_size")["greedy_minimal_per_event"].apply(list)),
        list(df.groupby("sample_size")["causal_cut_per_event"].apply(list)),
        list(df.groupby("sample_size")["causal_cut_plus_greedy_minimal_per_event"].apply(list)),
    ],
    labels=[BASELINE, TOOLNAME, f"{TOOLNAME} + {BASELINE}"],
    colours=[RED, GREEN, MAGENTA],
    xticklabels=SAMPLE_SIZES,
    xlabel="Number of executions",
    ylabel="Tool-minimised test length\n(normalised by original length)",
    legend_args={"loc": "upper left"},
    savepath=f"{figures}/rq1-data-size.pgf",
)

# Test whether adding each increment of data makes the minimised test longer or shorter
test_sample_sizes = {}
size_intervals = [
    (50, 100),
    (100, 250),
    (250, 500),
    (500, 1000),
    (1000, 2000),
    (2000, 3000),
    (3000, 4000),
    (4000, 5000),
]
for x, y in size_intervals:
    test_sample_sizes[f"{x} > {y}"] = {
        "causal_cut_shorter": mannwhitneyu(
            df.loc[df.sample_size == x, "causal_cut"],
            df.loc[df.sample_size == y, "causal_cut"],
            alternative="greater",
        ).pvalue,
        "causal_cut_longer": mannwhitneyu(
            df.loc[df.sample_size == x, "causal_cut"],
            df.loc[df.sample_size == y, "causal_cut"],
            alternative="less",
        ).pvalue,
        "causal_cut+_shorter": mannwhitneyu(
            df.loc[df.sample_size == x, "causal_cut_plus_greedy_minimal"],
            df.loc[df.sample_size == y, "causal_cut_plus_greedy_minimal"],
            alternative="greater",
        ).pvalue,
        "causal_cut+_longer": mannwhitneyu(
            df.loc[df.sample_size == x, "causal_cut_plus_greedy_minimal"],
            df.loc[df.sample_size == y, "causal_cut_plus_greedy_minimal"],
            alternative="less",
        ).pvalue,
    }
pd.DataFrame(test_sample_sizes).map(bold_if_significant).to_latex(f"{stats_dir}/rq1-data-size.tex")

# 1c. confidence intervals
plot_grouped_boxplot(
    [
        list(df.groupby("ci_alpha")["greedy_minimal_per_event"].apply(list)),
        list(df.groupby("ci_alpha")["causal_cut_per_event"].apply(list)),
        list(df.groupby("ci_alpha")["causal_cut_plus_greedy_minimal_per_event"].apply(list)),
    ],
    labels=[BASELINE, TOOLNAME, f"{TOOLNAME} + {BASELINE}"],
    colours=[RED, GREEN, MAGENTA],
    xticklabels=CONFIDENCE_INTERVALS,
    xlabel="CI alpha",
    ylabel="Tool-minimised test length\n(normalised by original length)",
    legend_args={"loc": "upper right"},
    savepath=f"{figures}/rq1-confidence-intervals.pgf",
)

# Test whether 80% or 90% confidence intervals make the minimised test longer or shorter
test_sample_sizes = {
    f"{toolname}_{x}_less": [
        mannwhitneyu(
            df.loc[df.ci_alpha == x, toolname],
            df.loc[df.ci_alpha == y, toolname],
            alternative="less",
        ).pvalue
    ]
    + list(cliffs_delta(df.loc[df.ci_alpha == x, toolname], df.loc[df.ci_alpha == y, toolname]))
    for toolname in ["causal_cut", "causal_cut_plus_greedy_minimal"]
    for x, y in [(80, 90), (90, 80)]
}

pd.DataFrame().from_dict(test_sample_sizes, orient="index", columns=["p_value", "effect_size", "effect_class"]).map(
    round_format
).to_latex(f"{stats_dir}/rq1-confidence-inverval.tex")

# RQ2: Factors that influence number of executions


pd.concat(
    [
        df.groupby("original_length")[
            ["greedy_minimal_executions", "causal_cut_executions", "causal_cut_plus_greedy_minimal_executions"]
        ]
        .apply(lambda x: x.mean())
        .rename(
            {
                "greedy_minimal_executions": BASELINE,
                "causal_cut_executions": TOOLNAME,
                "causal_cut_plus_greedy_minimal_executions": f"{TOOLNAME} + {BASELINE}",
            },
            axis=1,
        ),
        pd.DataFrame(
            {
                BASELINE: {
                    "Mean": df["greedy_minimal_executions"].mean(),
                    "Median": df["greedy_minimal_executions"].median(),
                },
                TOOLNAME: {
                    "Mean": df["causal_cut_executions"].mean(),
                    "Median": df["causal_cut_executions"].median(),
                },
                f"{TOOLNAME} + {BASELINE}": {
                    "Mean": df["causal_cut_plus_greedy_minimal_executions"].mean(),
                    "Median": df["causal_cut_plus_greedy_minimal_executions"].median(),
                },
            }
        ),
    ]
).T.round(2).to_latex(f"{stats_dir}/rq2-executions.tex", float_format="%.2f")

# 2a. original test length
fig, ax = plt.subplots()

plot_grouped_boxplot(
    [
        list(df.groupby("original_length")["greedy_minimal_executions"].apply(list)),
        list(df.groupby("original_length")["causal_cut_executions"].apply(list)),
        list(df.groupby("original_length")["causal_cut_plus_greedy_minimal_executions"].apply(list)),
    ],
    ax=ax,
    labels=[BASELINE, TOOLNAME, f"{TOOLNAME} + {BASELINE}"],
    colours=[RED, GREEN, MAGENTA],
    xticklabels=ORIGINAL_ATTACK_LENGTHS,
    xlabel="Original test length",
    ylabel="Minimisation executions",
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

# Greedy fit
ax.plot(
    ax.get_xticks()[:11] - 0.5,
    df.groupby("original_length")["greedy_minimal_executions"].apply(np.median)[:11],
    color=RED,
    alpha=0.5,
)
# CausalCut fit
model = ols(
    "causal_cut_executions ~ np.log(original_length) + original_length - 1",
    df[["original_length", "causal_cut_executions"]],
).fit()
ax.plot(
    ax.get_xticks()[:11] + 0.5,
    model.predict(pd.DataFrame({"original_length": ORIGINAL_ATTACK_LENGTHS[:11]})),
    color=GREEN,
    alpha=0.5,
)
# CausalCut + Greedy fit
model = ols(
    "causal_cut_plus_greedy_minimal_executions ~ np.log(original_length) + original_length - 1",
    df[["original_length", "causal_cut_plus_greedy_minimal_executions"]],
).fit()
ax.plot(
    ax.get_xticks()[:11] + 0.5,
    model.predict(pd.DataFrame({"original_length": ORIGINAL_ATTACK_LENGTHS[:11]})),
    color=MAGENTA,
    alpha=0.5,
)

plt.savefig(f"{figures}/rq2-test-length.pgf", bbox_inches="tight", pad_inches=0)
plt.clf()

# Correlation between original test length and pruned test length
test_length_stats = []
for technique in ["greedy_minimal_executions", "causal_cut_executions", "causal_cut_plus_greedy_minimal_executions"]:
    raw_stat, raw_p_value = spearmanr(df["original_length"], df[technique])
    normalised_stat, normalised_p_value = spearmanr(df["original_length"], df[technique + "_per_event"])
    test_length_stats.append(
        {
            "technique": technique,
            "raw_stat": raw_stat,
            "raw_p_value": raw_p_value,
            "normalised_stat": normalised_stat,
            "normalised_p_value": normalised_p_value,
        }
    )


pd.DataFrame(test_length_stats).map(round_format).to_latex(os.path.join(stats_dir, "rq2-test-length.tex"), index=False)


# 2b. data available
plot_grouped_boxplot(
    [
        list(df.groupby("sample_size")["greedy_minimal_executions_per_event"].apply(list)),
        list(df.groupby("sample_size")["causal_cut_executions_per_event"].apply(list)),
        list(df.groupby("sample_size")["causal_cut_plus_greedy_minimal_executions_per_event"].apply(list)),
    ],
    labels=[BASELINE, TOOLNAME, f"{TOOLNAME} + {BASELINE}"],
    colours=[RED, GREEN, MAGENTA],
    xticklabels=SAMPLE_SIZES,
    xlabel="Number of executions",
    ylabel="Minimisation executions\n(normalised by original length)",
    legend_args={"loc": "upper left"},
    savepath=f"{figures}/rq2-data-size.pgf",
)

# Test whether adding each increment of data makes the minimised test longer or shorter
test_sample_sizes = {}
size_intervals = [
    (50, 100),
    (100, 250),
    (250, 500),
    (500, 1000),
    (1000, 2000),
    (2000, 3000),
    (3000, 4000),
    (4000, 5000),
]
for x, y in size_intervals:
    test_sample_sizes[f"{x} > {y}"] = {
        "causal_cut_shorter": mannwhitneyu(
            df.loc[df.sample_size == x, "causal_cut_executions"],
            df.loc[df.sample_size == y, "causal_cut_executions"],
            alternative="greater",
        ).pvalue,
        "causal_cut_longer": mannwhitneyu(
            df.loc[df.sample_size == x, "causal_cut_executions"],
            df.loc[df.sample_size == y, "causal_cut_executions"],
            alternative="less",
        ).pvalue,
        "causal_cut+_shorter": mannwhitneyu(
            df.loc[df.sample_size == x, "causal_cut_plus_greedy_minimal_executions"],
            df.loc[df.sample_size == y, "causal_cut_plus_greedy_minimal_executions"],
            alternative="greater",
        ).pvalue,
        "causal_cut+_longer": mannwhitneyu(
            df.loc[df.sample_size == x, "causal_cut_plus_greedy_minimal_executions"],
            df.loc[df.sample_size == y, "causal_cut_plus_greedy_minimal_executions"],
            alternative="less",
        ).pvalue,
    }
pd.DataFrame(test_sample_sizes).map(bold_if_significant).to_latex(f"{stats_dir}/rq2-data-size.tex")

# 2c. confidence intervals
plot_grouped_boxplot(
    [
        list(df.groupby("ci_alpha")["greedy_minimal_executions"].apply(list)),
        list(df.groupby("ci_alpha")["causal_cut_executions"].apply(list)),
        list(df.groupby("ci_alpha")["causal_cut_plus_greedy_minimal_executions"].apply(list)),
    ],
    labels=[BASELINE, TOOLNAME, f"{TOOLNAME} + {BASELINE}"],
    colours=[RED, GREEN, MAGENTA],
    xticklabels=CONFIDENCE_INTERVALS,
    xlabel="CI alpha",
    ylabel="Minimisation executions",
    legend_args={"loc": "upper right"},
    savepath=f"{figures}/rq2-confidence-intervals.pgf",
)

# Test whether 80% or 90% confidence intervals make the minimised test longer or shorter
test_sample_sizes = {
    f"{toolname}_{x}_less": [
        mannwhitneyu(
            df.loc[df.ci_alpha == x, toolname],
            df.loc[df.ci_alpha == y, toolname],
            alternative="less",
        ).pvalue
    ]
    + list(cliffs_delta(df.loc[df.ci_alpha == x, toolname], df.loc[df.ci_alpha == y, toolname]))
    for toolname in ["causal_cut_executions", "causal_cut_plus_greedy_minimal_executions"]
    for x, y in [(80, 90), (90, 80)]
}

pd.DataFrame().from_dict(test_sample_sizes, orient="index", columns=["p_value", "effect_size", "effect_class"]).map(
    round_format
).to_latex(f"{stats_dir}/rq2-confidence-inverval.tex")

# RQ3 practicality
max_x = df[[f"{technique}_executions_per_event" for technique in TECHNIQUES]].max().max()
fig, ax = plt.subplots()
for technique, marker, color in zip(TECHNIQUES, ["o", "x", "+"], [RED, GREEN, MAGENTA]):
    ax.scatter(
        df[f"{technique}_executions_per_event"],
        df[f"{technique}_removed_per_event"],
        label=technique,
        marker=marker,
        color=color,
    )
    if technique != "greedy_minimal":
        ax.plot(
            np.linspace(0, max_x),
            ols(f"{technique}_removed_per_event ~ {technique}_executions_per_event", df)
            .fit()
            .predict(pd.DataFrame({f"{technique}_executions_per_event": np.linspace(0, max_x)}))
            .values,
            color=color,
        )
# ax.legend(loc="lower left")
ax.vlines(1, ymin=0, ymax=1, color=RED)
ax.set_ylabel("Proportion of test removed")
ax.set_xlabel("Executions per event")
ax.set_xlim(0)
ax.set_ylim(0, 1)
plt.savefig(f"{figures}/rq3.png", bbox_inches="tight", pad_inches=0)

fig, axs = plt.subplots(3, 4, sharex=True, sharey=True)
max_x = df[[f"{technique}_executions_per_event" for technique in TECHNIQUES]].max().max()
for original_length, ax in zip(ORIGINAL_ATTACK_LENGTHS, axs.reshape(-1)):
    for technique, marker, color in zip(TECHNIQUES, ["o", "x", "+"], [RED, GREEN, MAGENTA]):
        ax.set_title(f"Length {original_length}")
        ax.scatter(
            df.loc[df.original_length == original_length, f"{technique}_executions_per_event"],
            df.loc[df.original_length == original_length, f"{technique}_removed_per_event"],
            label=technique,
            marker=marker,
            color=color,
        )
        ax.set_xlim(0)
        ax.set_ylim(0, 1)

# plt.legend(loc="upper left")
fig.text(0.5, 0, "Executions per event", ha="center", va="center")
fig.text(0, 0.5, "Proportion of test removed", ha="center", va="center", rotation="vertical")
plt.tight_layout()
plt.savefig(f"{figures}/rq3_subplots_per_event.png", bbox_inches="tight", pad_inches=0)
