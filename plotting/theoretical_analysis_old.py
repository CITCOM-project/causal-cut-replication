import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants import BASELINE, TOOLNAME, RED, GREEN, BLUE, MAGENTA, GOLD_STANDARD, RANGE_1

MIN_LENGTH = 1
MAX_LENGTH = 10

cc = []
for original_length in range(MIN_LENGTH, MAX_LENGTH):
    for m_prefix in range(1, original_length + 1):  # m_prefix must contain at least one necessary cause
        for spurious in range(original_length - m_prefix + 1):
            cc.append(
                {"original_length": original_length, "pruned_length": m_prefix + spurious, "executions": m_prefix + 1}
            )
cc = pd.DataFrame(cc)
cc["executions_per_event"] = cc["executions"] / cc["original_length"]
cc["proportion_removed"] = (cc["original_length"] - cc["pruned_length"]) / cc["original_length"]


cc_plus_best = []
for original_length in range(MIN_LENGTH, MAX_LENGTH):
    for m_prefix in range(1, original_length + 1):  # m_prefix must contain at least one necessary cause
        for necessary_causes in range(1, m_prefix + 1):
            cc_plus_best.append(
                {"original_length": original_length, "pruned_length": necessary_causes, "executions": m_prefix + 1}
            )
cc_plus_best = pd.DataFrame(cc_plus_best)
cc_plus_best["executions_per_event"] = cc_plus_best["executions"] / cc_plus_best["original_length"]
cc_plus_best["proportion_removed"] = (cc_plus_best["original_length"] - cc_plus_best["pruned_length"]) / cc_plus_best[
    "original_length"
]


plt.scatter(
    np.ones(len(set(cc["proportion_removed"]))),
    list(set(cc["proportion_removed"])),
    color=RED,
    label="Greedy heuristic",
)
plt.scatter(cc["executions_per_event"], cc["proportion_removed"], color=GREEN, marker="x", label="\toolname")
plt.scatter(
    cc_plus_best["executions_per_event"],
    cc_plus_best["proportion_removed"],
    color=MAGENTA,
    marker="+",
    label="\toolnamePlus",
)

plt.legend()
plt.ylabel("Proportion of test removed")
plt.xlabel("Executions per event")
plt.savefig("/tmp/analysis.png")

print(cc_plus_best.loc[cc_plus_best["executions_per_event"] > 1, "original_length"].max())
