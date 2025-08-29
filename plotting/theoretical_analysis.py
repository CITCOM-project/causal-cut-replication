import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants import BASELINE, TOOLNAME, RED, GREEN, BLUE, MAGENTA, GOLD_STANDARD, RANGE_1

plt.style.use("ggplot")

MIN_LENGTH = 2
MAX_LENGTH = 25

TRACE_LENGTHS = np.linspace(MIN_LENGTH, MAX_LENGTH)

# Greedy
(greedy,) = plt.plot(TRACE_LENGTHS, np.ones(len(TRACE_LENGTHS)) / TRACE_LENGTHS, color=RED, label=BASELINE)

# CausalCut best
(cc_best,) = plt.plot(
    TRACE_LENGTHS, np.ones(len(TRACE_LENGTHS)), color=GREEN, label=TOOLNAME + " best", linestyle="dashed"
)


# CausalCut worst
(cc_worst_n1,) = plt.plot(
    TRACE_LENGTHS,
    (1 / TRACE_LENGTHS) / (TRACE_LENGTHS + 1),
    color=GREEN,
    label=TOOLNAME + " worst (|n|=1)",
    linestyle=(0, (5, 3)),
)
(cc_worst_nt,) = plt.plot(
    TRACE_LENGTHS,
    np.ones(len(TRACE_LENGTHS)) / (TRACE_LENGTHS + 1),
    color=GREEN,
    label=TOOLNAME + " worst (n=t)",
    linestyle=(2, (1, 7)),
    zorder=10,
)

plt.fill_between(
    TRACE_LENGTHS,
    np.ones(len(TRACE_LENGTHS)),
    (1 / TRACE_LENGTHS) / (TRACE_LENGTHS + 1),
    color="none",
    edgecolor=GREEN,
    alpha=0.5,
    hatch="\\",
)

# CausalCut+ best
(cc_plus_best,) = plt.plot(
    TRACE_LENGTHS,
    np.ones(len(TRACE_LENGTHS)) / 2,
    color=MAGENTA,
    label=f"{TOOLNAME}+{BASELINE} best (|n|=1)",
    linestyle="dashed",
)
(cc_plus_best_nt,) = plt.plot(
    TRACE_LENGTHS,
    np.ones(len(TRACE_LENGTHS)) / (TRACE_LENGTHS + 1),
    color=MAGENTA,
    label=f"{TOOLNAME}+{BASELINE} best (n=t)",
    linestyle=(0, (5, 3)),
)
# CausalCut+ worst
(cc_plus_worst_n1,) = plt.plot(
    TRACE_LENGTHS,
    (1 / TRACE_LENGTHS) / (TRACE_LENGTHS + 1),
    color=MAGENTA,
    label=f"{TOOLNAME}+{BASELINE} worst (|n|=1)",
    linestyle=(2, (1, 7)),
)


plt.fill_between(
    TRACE_LENGTHS,
    np.ones(len(TRACE_LENGTHS)) / 2,
    (1 / TRACE_LENGTHS) / (TRACE_LENGTHS + 1),
    color="none",
    edgecolor=MAGENTA,
    alpha=0.5,
    hatch="/",
)
plt.xlabel("Test length")
plt.ylabel("PPV per execution")
handles = [cc_best, cc_plus_best, greedy, (cc_worst_nt, cc_plus_best_nt), (cc_worst_n1, cc_plus_worst_n1)]
plt.legend(
    handles=handles,
    labels=[h.get_label() if hasattr(h, "get_label") else "\n".join(h1.get_label() for h1 in h) for h in handles],
    loc="upper right",
    bbox_to_anchor=(0.95, 0.95),
)
plt.savefig("technique-bounds.pgf", bbox_inches="tight", pad_inches=0)

# Numbers
df = []
for original_length in range(MIN_LENGTH, MAX_LENGTH + 1):
    for necessary_prefix in range(1, original_length + 1):
        for necessary_interventions in range(1, necessary_prefix + 1):
            for correct_estimates in range(necessary_interventions + 1):
                simulations = (necessary_interventions - correct_estimates) + 1
                df.append(
                    {
                        "original_length": original_length,
                        "necessary_prefix": necessary_prefix,
                        "necessary_interventions": necessary_interventions,
                        "correct_estimates": correct_estimates,
                        "simulations": simulations,
                    }
                )
df = pd.DataFrame(df)
df["proportion_correctly_estimated"] = df["correct_estimates"] / df["necessary_interventions"]
df["cc_ppv"] = df["necessary_interventions"] / df["necessary_prefix"]
df["cc_cost_effectiveness"] = df["cc_ppv"] / df["simulations"]
df["greedy_cost_effectiveness"] = 1 / df["original_length"]
df["cc_better"] = df["greedy_cost_effectiveness"] < df["cc_cost_effectiveness"]
df.to_csv("/tmp/analysis.csv")

better = df.query("cc_better")
worse = df.query("not cc_better")

analysis = pd.concat(
    [
        df.groupby("original_length")["cc_better"].apply(lambda x: x.sum() / len(x)),
        df.query("correct_estimates >= 1").groupby("original_length")["cc_better"].apply(lambda x: x.sum() / len(x)),
    ],
    axis=1,
)
analysis.columns = ["p(cc_better)", "p(cc_better | at least one correct estimate)"]
print(analysis)
