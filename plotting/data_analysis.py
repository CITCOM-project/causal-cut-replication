"""
This module processes the causal test logs and draws the figures.
"""

import sys
import os
import json
import re
from scipy.stats import spearmanr
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from constants import BASELINE, TOOLNAME, RED, GREEN, BLUE, MAGENTA, DDMIN
from grouped_boxplot import plot_grouped_boxplot

import warnings

warnings.filterwarnings("ignore")

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


def find_necessary_prefix(m, t):
    max_intervention = t.index(max(m))
    return t[: max_intervention + 1]


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
            assert "greedy_minimal" in attack, f"No greedy_heuristic in {os.path.join(root, file)}"
            attack["sample_size"] = sample_size
            attack["ci_alpha"] = ci_alpha
            attack["original_length"] = len(attack["attack"])
            attack["necessary_prefix"] = find_necessary_prefix(attack["minimal"], attack["attack"])
            if "error" in attack:
                if "treatment_strategies" not in attack:
                    assert attack["error"] in [
                        "Missing data for control_strategy",
                        "No faults observed. P(error) = 0",
                    ], f"Bad error {attack['error']} in {os.path.join(root, file)}"
                    attack["treatment_strategies"] = []
        attacks += log

# FIT101 False
# FIT201 False
# FIT301 True # No test with FIT301 as an outcome can be failure-inducing on our dataset
# FIT401 False
# FIT501 False
# FIT601 False
# DPIT301 False
# LIT101 False
# LIT301 False
# LIT401 False

df = pd.DataFrame(attacks)
for col in ["attack", "minimal", "estimated_interventions", "necessary_prefix"]:
    df[col] = df[col].apply(lambda x: tuple(map(tuple, x)))

# print(df.loc[df["error"] == "Missing data for control_strategy", ["attack_index", "outcome"]])  # FIT401
df = df.query("outcome != 'FIT301'")
df = df.query("outcome != 'FIT401'")
df["minimal_no_time"] = df["minimal"].apply(lambda t: tuple(map(lambda i: i[1:], t)))
df["last_necessary_intervention"] = [max(x)[0] for x in df["minimal"]]
df["total_time"] = [max(x)[0] for x in df["attack"]]


# print("\n".join(map(str, sorted(df["minimal"].apply(lambda t: tuple(map(lambda i: i[1:], t))).unique()))))
df["estimable"] = [sum("result" in intervention for intervention in test) for test in df["treatment_strategies"]]
df["estimable_per_event"] = df["estimable"] / df["original_length"]

df = df.loc[df["original_length"] > 1]
# Make field names consistent with the paper
df.rename(
    {
        "simulator_runs": "causal_cut_executions",
        "reduced_simulator_runs": "causal_cut_plus_greedy_heuristic_executions",
        "reduced_extended_interventions": "causal_cut_plus_greedy_heuristic",
        "extended_interventions": "causal_cut",
        "greedy_minimal": "greedy_heuristic",
    },
    axis=1,
    inplace=True,
)

df["greedy_heuristic_executions"] = df["original_length"]

TECHNIQUES = ["greedy_heuristic", "ddmin", "causal_cut", "causal_cut_plus_greedy_heuristic"]

for technique in ["minimal"] + TECHNIQUES:
    df[technique] = df[technique].apply(lambda a: len(tuple(map(tuple, a))))
    df[f"{technique}_per_event"] = df[technique] / df["original_length"]
    df[f"{technique}_removed"] = df["original_length"] - df[technique]
    df[f"{technique}_removed_per_event"] = (df["original_length"] - df[technique]) / df["original_length"]
    df[f"{technique}_percentage_reduction"] = ((df["original_length"] - df[technique]) / df["original_length"]) * 100
    df[f"{technique}_percentage_spurious_removed"] = (
        (df["original_length"] - df[technique]) / (df["original_length"] - df["minimal"])
    ).replace(np.nan, 1) * 100
    df[f"{technique}_ppv"] = df["minimal"] / df[technique]
    df[f"{technique}_specificity"] = (df[technique] - (df["original_length"] - df["minimal"])) / (
        df["original_length"] - df["minimal"]
    )
    if technique != "minimal":
        df[f"{technique}_executions_per_event"] = df[f"{technique}_executions"] / df["original_length"]
        df[f"{technique}_cost_efficiency"] = df["minimal"] / (df[technique] * df[f"{technique}_executions"])

df["estimated_interventions"] = df["estimated_interventions"].apply(len)


if "case-1" in logs:
    df = df.query("sample_size <= 5000 | sample_size > 9000")


def position_offsets(feature):
    if "case-2" in logs and feature == "original_length":
        return ([0] * 10) + [3] + [6]
    if "case-1" in logs and feature == "sample_size":
        return ([0] * 6) + [1] + [2]
    return None


def zigzag(feature):
    if "case-2" in logs and feature == "original_length":
        return [[0.77, 0.905], [0.78, 0.915]]
    if "case-1" in logs and feature == "sample_size":
        return [[0.725, 0.88], [0.735, 0.89]]
    return []


ORIGINAL_TEST_LENGTHS = sorted(list(set(df.original_length)))
NECESSARY_INTERVENTIONS = sorted(list(set(df.minimal)))
SAMPLE_SIZES = sorted(list(set(df.sample_size)))
CONFIDENCE_INTERVALS = [80, 90]

OUTCOMES = {
    "_cost_efficiency": ["original_length", "minimal", "estimable_per_event"],  # RQ1
    "": ["original_length", "minimal", "sample_size"],  # RQ2
    "_executions": ["original_length", "minimal", "sample_size"],  # RQ3
}
y_labels = {
    "_cost_efficiency": "Cost efficiency",
    "": "Tool-minimised test length",
    "_executions": "Executions",
    "_reinstatement": "Reinstatement rate",
}
FEATURES = ["original_length", "minimal", "sample_size", "ci_alpha", "estimable_per_event"]
x_labels = {
    "original_length": "Original test length",
    "minimal": "Proportion of necessary interventions",
    "sample_size": "Executions available",
    "estimable_per_event": "Proportion of estimable interventions",
    "ci_alpha": "CI alpha",
}
x_ticks = {k: sorted(list(set(df[k]))) for k in FEATURES}
technique_labels = {
    "greedy_heuristic": "\\greedy",
    "ddmin": "\\ddmin",
    "causal_cut": "\\toolname",
    "causal_cut_plus_greedy_heuristic": "\\toolnamePlus",
}
technique_colours = {
    "greedy_heuristic": RED,
    "ddmin": BLUE,
    "causal_cut": GREEN,
    "causal_cut_plus_greedy_heuristic": MAGENTA,
}
technique_markers = {
    "greedy_heuristic": "x",
    "ddmin": "^",
    "causal_cut": "o",
    "causal_cut_plus_greedy_heuristic": "+",
}


# Original and gold standard attack lengths (Table 1)
lengths = [
    (len(attack), necessary_prefix, len_minimal)
    for attack, necessary_prefix, len_minimal in df[["attack", "necessary_prefix", "minimal"]]
    .groupby(["attack", "necessary_prefix", "minimal"])
    .groups.keys()
]

minimised_lengths = {}
necessary_prefixes = {}
for original_length, necessary_prefix, minimised_length in lengths:
    minimised_lengths[minimised_length] = sorted(minimised_lengths.get(minimised_length, []) + [original_length])
    necessary_prefixes[necessary_prefix] = sorted(necessary_prefixes.get(necessary_prefix, []) + [original_length])
minimised_lengths = {k: dict(Counter(v)) for k, v in minimised_lengths.items()}
minimised_lengths = pd.DataFrame(minimised_lengths).sort_index().T.fillna(0).astype(int).replace(0, "")
minimised_lengths.sort_index().to_latex(f"{stats_dir}/attack-lengths.tex")

necessary_prefixes = {k: dict(Counter(v)) for k, v in necessary_prefixes.items()}
necessary_prefixes = pd.DataFrame(necessary_prefixes).sort_index().T.fillna(0).astype(int).replace(0, "")
necessary_prefixes.sort_index().to_latex(f"{stats_dir}/necessary-prefixes.tex")


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


def format_latex(df, filename, **kwargs):
    pad_lengths = df.map(lambda x: len(str(x))).apply(max)
    output = []
    for line in df.to_latex(**kwargs).split("\n"):
        if "&" in line:
            output.append(
                "& " + "&".join(val.ljust(pad + 2) for val, pad in zip(line[:-2].split("&"), pad_lengths)) + "\\\\"
            )
    with open(filename, "w") as f:
        f.write("\n".join(output))


print("=" * 40, "Cost efficiency", "=" * 40)
print(df[[f"{t}_cost_efficiency" for t in TECHNIQUES]].max())
print(
    "MAX DIFFERENCE GREEDY median", (df["causal_cut_cost_efficiency"] - df["greedy_heuristic_cost_efficiency"]).median()
)
print("MAX DIFFERENCE GREEDY mean", (df["causal_cut_cost_efficiency"] - df["greedy_heuristic_cost_efficiency"]).mean())
print("MAX DIFFERENCE DDmin median", (df["causal_cut_cost_efficiency"] - df["ddmin_cost_efficiency"]).median())
print("MAX DIFFERENCE DDmin mean", (df["causal_cut_cost_efficiency"] - df["ddmin_cost_efficiency"]).mean())

print("=" * 40, "Spurious events", "=" * 40)
print("Median percentage of spurious events removed")
print(df[[f"{t}_percentage_spurious_removed" for t in TECHNIQUES]].median())
print("Mean percentage of spurious events removed")
print(df[[f"{t}_percentage_spurious_removed" for t in TECHNIQUES]].mean())

print("=" * 40, "Executions", "=" * 40)
print("Median executions")
print(df[[f"{t}_executions_per_event" for t in TECHNIQUES]].median())
print("Mean executions")
print(df[[f"{t}_executions_per_event" for t in TECHNIQUES]].mean())
print("=" * 80)

# Estimable events in the dataset by sample size
fig, ax = plt.subplots(figsize=(6.5, 4))
plot_grouped_boxplot(
    [
        list(df.groupby("sample_size")["estimable_per_event"].apply(list)),
    ],
    ax=ax,
    colours=[GREEN],
    xticklabels=x_ticks["sample_size"],
    xlabel=x_labels["sample_size"],
    ylabel="Proportion of interventions estimable",
    savepath=f"{figures}/estimable.pdf",
)

# Position of last necessary intervention
plt.hist(df["last_necessary_intervention"] / df["total_time"], bins=25, color=GREEN)
plt.savefig(f"{figures}/last_necessary.pdf")

legend_args = {
    "ncol": 4,
    "loc": "upper center",
    "bbox_to_anchor": (0.5, 1.11),
    "columnspacing": 0.7,
    "handletextpad": 0.5,
    "handlelength": 1,
}

df["minimal"] = df["minimal"] / df["original_length"]


def plot_feature(feature, ax):
    if feature in {"estimable_per_event", "minimal"}:
        for technique in TECHNIQUES:
            if f"{technique}{outcome}" in df:
                ax.scatter(
                    df[feature],
                    df[f"{technique}{outcome}"],
                    color=technique_colours[technique],
                    marker=technique_markers[technique],
                    alpha=0.1,
                )
        ax.set_xlabel(x_labels[feature])
        ax.set_xticks(np.linspace(0, 1, 11))
        ax.set_xlim(-0.01)

    else:
        groups = [
            list(df.groupby(feature)[f"{technique}{outcome}"].apply(list))
            for technique in TECHNIQUES
            if f"{technique}{outcome}" in df
        ]
        colours = (
            [technique_colours[technique] for technique in TECHNIQUES]
            if len(groups) == len(TECHNIQUES)
            else [GREEN, MAGENTA]
        )
        plot_grouped_boxplot(
            groups,
            ax=ax,
            colours=colours,
            xticklabels=x_ticks[feature],
            xlabel=x_labels[feature],
            ylabel=y_labels[outcome],
            position_offsets=position_offsets(feature),
            # Highlight the gap in data with a zigzag on x axis
            zigzag=zigzag(feature),
            # logscale=(rq == 3 and feature == "sample_size"),
        )
    ax.set_ylim(0)


for rq, outcome in enumerate(OUTCOMES, 1):
    rq_stats = [pd.DataFrame({"technique": [technique_labels[t] for t in TECHNIQUES]})]
    if rq > 1:
        for technique in TECHNIQUES:
            df[f"technique{outcome}"] = df[f"{technique}{outcome}"] / df["original_length"]
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(6.5 * 3, 4), gridspec_kw={"wspace": 0.05, "hspace": 0})
    for feature, ax in zip(OUTCOMES[outcome], axs.reshape(-1)):
        plot_feature(feature, ax)
    axs[0].set_ylabel(y_labels[outcome])
    axs[1].legend(
        handles=[
            plt.scatter([None], [None], marker="x", color=RED),
            plt.scatter([None], [None], marker="^", color=BLUE),
            plt.scatter([None], [None], marker="o", color=GREEN),
            plt.scatter([None], [None], marker="+", color=MAGENTA),
        ],
        labels=[BASELINE, DDMIN, TOOLNAME, f"{TOOLNAME} + {BASELINE}"],
        **legend_args,
    )
    plt.savefig(f"{figures}/rq{rq}.pdf", bbox_inches="tight", pad_inches=0)

    for feature in FEATURES:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        plot_feature(feature, ax)
        ax.set_ylabel(y_labels[outcome])
        ax.legend(
            handles=[
                plt.scatter([None], [None], marker="x", color=RED),
                plt.scatter([None], [None], marker="^", color=BLUE),
                plt.scatter([None], [None], marker="o", color=GREEN),
                plt.scatter([None], [None], marker="+", color=MAGENTA),
            ],
            labels=[BASELINE, DDMIN, TOOLNAME, f"{TOOLNAME} + {BASELINE}"],
            **legend_args,
        )
        plt.tight_layout()
        plt.savefig(f"{figures}/rq{rq}-{feature}.pdf", bbox_inches="tight", pad_inches=0)

        stats = []
        for technique in [f"{t}{outcome}" for t in TECHNIQUES]:
            raw_stat, raw_p_value = spearmanr(df[feature], df[technique])
            stats.append(
                {
                    f"{feature}_stat": round_format(raw_stat),
                    f"{feature}_p_value": bold_if_significant(raw_p_value),
                }
            )
        rq_stats.append(pd.DataFrame(stats))
    format_latex(pd.concat(rq_stats, axis=1), f"{stats_dir}/rq{rq}.tex", index=False, header=False)


# Reinstatement ratio
df["causal_cut_reinstatement"] = (df["causal_cut"] - df["estimated_interventions"]) / df["causal_cut"]
df["causal_cut_plus_greedy_heuristic_reinstatement"] = (
    df["causal_cut_plus_greedy_heuristic"] - df["estimated_interventions"]
) / df["causal_cut_plus_greedy_heuristic"]
outcome = "_reinstatement"
rq = "2a"
for feature in FEATURES:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    plot_feature(feature, ax)
    ax.set_ylabel(y_labels[outcome])
    ax.legend(
        handles=[
            plt.scatter([None], [None], marker="o", color=GREEN),
            plt.scatter([None], [None], marker="+", color=MAGENTA),
        ],
        labels=[TOOLNAME, f"{TOOLNAME} + {BASELINE}"],
        **legend_args,
    )
    ax.set_yticks([x / 10 for x in range(0, 11, 2)])
    ax.set_ylim(-0.01, 1.01)
    plt.tight_layout()
    plt.savefig(f"{figures}/rq{rq}-{feature}.pdf", bbox_inches="tight", pad_inches=0)

    stats = []
    for technique in [f"{t}{outcome}" for t in ["causal_cut", "causal_cut_plus_greedy_heuristic"]]:
        raw_stat, raw_p_value = spearmanr(df[feature], df[technique])
        stats.append(
            {
                f"{feature}_stat": round_format(raw_stat),
                f"{feature}_p_value": bold_if_significant(raw_p_value),
            }
        )
    rq_stats.append(pd.DataFrame(stats))
format_latex(pd.concat(rq_stats, axis=1), f"{stats_dir}/rq{rq}.tex", index=False, header=False)
