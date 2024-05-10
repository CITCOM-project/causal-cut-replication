import pandas as pd
import statsmodels.formula.api as smf
from numpy import exp, append, nan
from lifelines import CoxPHFitter, CoxTimeVaryingFitter
import matplotlib.pyplot as plt
import config

novCEA = pd.read_csv("data/long_preprocessed.csv")
novCEA["time2"] = novCEA["time"] ** 2

# Use logistic regression to predict switching given baseline covariates
fitBLswitch = smf.logit("xo_t_do ~ time + time2", data=novCEA[novCEA.trtrand == 0]).fit()

# Estimate the probability of switching for each patient-observation included in the regression.
novCEA["pxo1"] = fitBLswitch.predict(novCEA)

# Use logistic regression to predict switching given baseline and time-updated covariates (model S12)
fitBLTDswitch = smf.logit(
    "xo_t_do ~ time + time2 + FIT201 + MV101 + LIT101",
    data=novCEA[(novCEA.trtrand == 0) & (novCEA["recent_prog_t_dc"] == 1)],
).fit()

# Estimate the probability of switching for each patient-observation included in the regression.
novCEA["pxo2"] = fitBLTDswitch.predict(novCEA)

# set prob of switching to zero where progtypetdc==0
novCEA.loc[(novCEA.trtrand == 0) & (novCEA["recent_prog_t_dc"] == 0), "pxo2"] = 0

# IPCW step 3: For each individual at each time, compute the inverse probability of remaining uncensored
# Estimate the probabilities of remaining ‘un-switched’ and hence the weights

novCEA["num"] = nan
novCEA["denom"] = nan
for subj, temp_subset in novCEA.groupby("id"):
    temp_subset = novCEA.loc[novCEA.id == subj].sort_values("time")
    temp_subset.at[temp_subset.index[0], "num"] = 1 - temp_subset.at[temp_subset.index[0], "pxo1"]
    temp_subset.at[temp_subset.index[0], "denom"] = 1 - temp_subset.at[temp_subset.index[0], "pxo2"]

    if len(temp_subset) > 1:
        for i in temp_subset.index[1:]:
            temp_subset.at[i, "num"] = temp_subset.at[i - 1, "num"] * (1 - temp_subset.at[i, "pxo1"])
            temp_subset.at[i, "denom"] = temp_subset.at[i - 1, "denom"] * (1 - temp_subset.at[i, "pxo2"])
    num = temp_subset["num"]
    denom = temp_subset["denom"]
    novCEA.loc[novCEA.id == subj] = temp_subset

novCEA.to_csv("/tmp/novCEA.csv")
assert not novCEA["num"].isnull().any(), f"{len(novCEA['num'].isnull())} null numerator values"
assert not novCEA["denom"].isnull().any(), f"{len(novCEA['denom'].isnull())} null denom values"

isControl = novCEA.trtrand == 0
novCEA.loc[isControl, "weight"] = 1 / novCEA.denom[isControl]
novCEA.loc[isControl, "sweight"] = novCEA.num[isControl] / novCEA.denom[isControl]

# set the weights to 1 in the treatment arm
novCEA.loc[~isControl, "weight"] = 1
novCEA.loc[~isControl, "sweight"] = 1

novCEA_KM = novCEA.loc[novCEA.xo_t_do == 0].copy()
novCEA_KM["tin"] = novCEA_KM.time
novCEA_KM["tout"] = pd.concat([(novCEA_KM.time + config.timesteps_per_intervention), novCEA_KM.fault_time], axis=1).min(
    axis=1
)

novCEA_KM["ok"] = novCEA_KM["tin"] < novCEA_KM["tout"]

novCEA_KM.to_csv("/tmp/novCEA_KM.csv")

assert (
    novCEA_KM["tin"] <= novCEA_KM["tout"]
).all(), f"Left before joining\n{novCEA_KM.loc[novCEA_KM['tin'] >= novCEA_KM['tout']]}"

novCEA_KM.dropna(axis=1, inplace=True)
novCEA_KM.replace([float("inf")], 100, inplace=True)

#  IPCW step 4: Use these weights in a weighted analysis of the outcome model
# Estimate the KM graph and IPCW hazard ratio using Cox regression.
cox_ph = CoxPHFitter()
cox_ph.fit(
    df=novCEA_KM,
    duration_col="tout",
    event_col="fault_t_do",
    weights_col="weight",
    cluster_col="id",
    robust=True,
    formula="trtrand",
    entry_col="tin",
)
cox_ph.print_summary()
