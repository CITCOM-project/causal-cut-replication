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
            latest_minimal_event = max(attack["minimal"])
            attack["necessary_prefix"] = find_necessary_prefix(attack["minimal"], attack["attack"])
            for test in [
                "greedy_minimal",
                "minimal",
                "necessary_prefix",
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

# Add extra columns
df["greedy_heuristic_executions"] = df["original_length"]
df["minimal_per_event"] = df["minimal"] / df["original_length"]

TECHNIQUES = ["greedy_heuristic", "causal_cut", "causal_cut_plus_greedy_heuristic"]

for technique in TECHNIQUES + ["minimal"]:
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

OUTCOMES = ["_cost_efficiency", "", "_executions"]
y_labels = {"_cost_efficiency": "Cost efficiency", "": "Tool-minimised test length", "_executions": "Executions"}
FEATURES = ["original_length", "minimal", "sample_size", "ci_alpha"]
x_labels = {
    "original_length": "Original test length",
    "minimal": "Necessary interventions",
    "sample_size": "Executions available",
    "ci_alpha": "CI alpha",
}
x_ticks = {k: sorted(list(set(df[k]))) for k in FEATURES}
technique_labels = {
    "greedy_heuristic": "\\greedy",
    "causal_cut": "\\toolname",
    "causal_cut_plus_greedy_heuristic": "\\toolnamePlus",
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


for rq, outcome in enumerate(OUTCOMES, 1):
    rq_stats = [pd.DataFrame({"technique": [technique_labels[t] for t in TECHNIQUES]})]
    for feature in FEATURES:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        plot_grouped_boxplot(
            [
                list(df.groupby(feature)[f"greedy_heuristic{outcome}"].apply(list)),
                list(df.groupby(feature)[f"causal_cut{outcome}"].apply(list)),
                list(df.groupby(feature)[f"causal_cut_plus_greedy_heuristic{outcome}"].apply(list)),
            ],
            ax=ax,
            labels=[BASELINE, TOOLNAME, f"{TOOLNAME} + {BASELINE}"],
            colours=[RED, GREEN, MAGENTA],
            xticklabels=x_ticks[feature],
            xlabel=x_labels[feature],
            ylabel=y_labels[outcome],
            legend_args={
                "ncol": 3,
                "loc": "upper center",
                "bbox_to_anchor": (0.5, 1.11),
                "columnspacing": 0.7,
                "handletextpad": 0.5,
                "handlelength": 1,
            },
            position_offsets=position_offsets(feature),
            # Highlight the gap in data with a zigzag on x axis
            zigzag=zigzag(feature),
        )
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
