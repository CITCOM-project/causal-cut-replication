import pandas as pd
from glob import glob
import matplotlib.pyplot as plt


def analyse(df):
    errors = df.loc[pd.isnull(df.significant)]
    results = df.loc[~pd.isnull(df.significant)]
    TP = results.loc[~results.spurious & results.significant]
    TN = results.loc[results.spurious & ~results.significant]
    FP = results.loc[results.spurious & results.significant]
    FN = results.loc[~results.spurious & ~results.significant]

    assert sum([len(x) for x in [TP, TN, FP, FN]]) == len(
        results
    ), f"{sum([len(x) for x in [TP, TN, FP, FN]])} != {len(results)}"
    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN, "Results": results, "Errors": errors}


totals = {"TP": [], "TN": [], "FP": [], "FN": []}

for fname in glob("logs/*.csv"):
    df = pd.read_csv(fname)
    df["spurious"] = df["spurious"].astype(bool)
    df["significant"] = df["significant"].astype(bool)
    print(fname)
    results = analyse(df)
    print(
        "TP:",
        len(results["TP"]),
        "TN:",
        len(results["TN"]),
        "FP:",
        len(results["FP"]),
        "FN:",
        len(results["FN"]),
        "Results:",
        len(results["Results"]),
        "Errors:",
        len(results["Errors"]),
    )

    for k in ["TP", "TN", "FP", "FN"]:
        totals[k] += results[k]["kurtosis"].tolist()

    if (results["TP"]["kurtosis"] > 50).any():
        print("HIGH", fname)

    for trace, group in df.groupby("outcome"):
        results = analyse(group)
        print(
            " ",
            trace.ljust(66),
            "TP:",
            len(results["TP"]),
            "TN:",
            len(results["TN"]),
            "FP:",
            len(results["FP"]),
            "FN:",
            len(results["FN"]),
            "Results:",
            len(results["Results"]),
            "Errors:",
            len(results["Errors"]),
        )

# vps = [plt.violinplot(totals[k], [i], showmedians=True, vert=False) for i, k in enumerate(["TP", "TN", "FP", "FN"])]
# plt.legend([vp["bodies"][0] for vp in vps], ["TP", "TN", "FP", "FN"])
vps = plt.boxplot([totals[k] for i, k in enumerate(["TP", "TN", "FP", "FN"])])
plt.xticks(ticks=range(1, 5), labels=["TP", "TN", "FP", "FN"])
# plt.show()
