import pandas as pd
import statsmodels.formula.api as smf
from numpy import exp, append, nan
from lifelines import CoxPHFitter, CoxTimeVaryingFitter
import matplotlib.pyplot as plt
import argparse
from safe_ranges import safe_ranges
import random
from collections import OrderedDict


def setup_xo_t_do(strategy_assigned, strategy_followed):
    censored = False
    result = []
    for x, y in zip(strategy_assigned, strategy_followed):
        if censored:
            result.append(None)
        else:
            result.append((not censored) and x != y)
            censored = x != y
    # First and last can't be censored
    return [0] + [int(x) if x is not None else None for x in result] + [0 if result[-1] is not None else None]


def setup_fault_t_do(values, min, max):
    fault_occurred = False
    result = []
    for value in values:
        fault = not (min <= value <= max)
        result.append((not fault_occurred) and fault)
        if (not fault_occurred) and fault:
            fault_occurred == True
    return [int(x) for x in result]


def setup_recent_prog_t_dc(now_prog_t_dc):
    result = []
    prog = False
    for value in now_prog_t_dc:
        prog = prog or value
        result.append(prog)
    return result


def preprocess_data(df, control_strategy, treatment_strategy, outcome):
    df["trtrand"] = None  # treatment/control arm
    df["fault_t_do"] = None  # did a fault occur here?
    df["xo_t_do"] = None  # did the individual deviate from the treatment of interest here?
    df["now_prog_t_dc"] = None  # has the situation progressed now?
    df["recent_prog_t_dc"] = None  # has the situation progressed in the past?
    df["fault_time"] = None  # when did a fault occur?

    individuals = []
    new_id = 0
    for id, individual in df.groupby("id"):
        strategy = [
            (var, time, individual.loc[individual.time == time, var].values[0]) for var, time, _ in treatment_strategy
        ]
        individual["fault_t_do"] = setup_fault_t_do(
            individual[outcome], safe_ranges[outcome]["lolo"], safe_ranges[outcome]["hihi"]
        )
        individual["now_prog_t_dc"] = setup_fault_t_do(
            individual[outcome], safe_ranges[outcome]["lo"], safe_ranges[outcome]["hi"]
        )
        individual["recent_prog_t_dc"] = setup_recent_prog_t_dc(individual["now_prog_t_dc"])
        faulty = individual.loc[
            ~individual[outcome].between(safe_ranges[outcome]["lolo"], safe_ranges[outcome]["hihi"]),
            "time",
        ]
        fault_time = individual.time.max()
        if len(faulty) > 0:
            fault_time = faulty.min()  # TODO: update this to take values from the full dataset
        individual["fault_time"] = random.randint(fault_time + 1, fault_time + 15)

        if fault_time <= 0:
            continue

        # Control flow:
        # Individuals that start off in both arms, need cloning (hence incrementing the ID within the if statements)
        # Individuals that don't start off in either arm need leaving out (hence two ifs rather than elif or else)
        if strategy[0] == control_strategy[0]:
            individual["id"] = id
            id += 1
            individual["trtrand"] = 0
            individual["xo_t_do"] = setup_xo_t_do(strategy, control_strategy)
            individuals.append(individual.loc[individual.time <= fault_time].copy())
        if strategy[0] == treatment_strategy[0]:
            individual["id"] = id
            id += 1
            individual["trtrand"] = 1
            individual["xo_t_do"] = setup_xo_t_do(strategy, treatment_strategy)
            individuals.append(individual.loc[individual.time <= fault_time].copy())
    df = pd.concat(individuals)
    df.to_csv("data/long_preprocessed.csv")


def estimate_hazard_ratio(novCEA, timesteps_per_intervention):
    # Use logistic regression to predict switching given baseline covariates
    fitBLswitch = smf.logit("xo_t_do ~ time + I(time**2)", data=novCEA[novCEA.trtrand == 0]).fit()

    # Estimate the probability of switching for each patient-observation included in the regression.
    novCEA["pxo1"] = fitBLswitch.predict(novCEA)

    # Use logistic regression to predict switching given baseline and time-updated covariates (model S12)
    fitBLTDswitch = smf.logit(
        "xo_t_do ~ time + I(time**2) + FIT201 + MV101 + LIT101",
        data=novCEA[(novCEA.trtrand == 0) & (novCEA["recent_prog_t_dc"] == 1)],
    ).fit()

    # Estimate the probability of switching for each patient-observation included in the regression.
    novCEA["pxo2"] = fitBLTDswitch.predict(novCEA)

    # set prob of switching to zero where progtypetdc==0
    novCEA.loc[(novCEA.trtrand == 0) & (novCEA["recent_prog_t_dc"] == 0), "pxo2"] = 0

    # IPCW step 3: For each individual at each time, compute the inverse probability of remaining uncensored
    # Estimate the probabilities of remaining ‘un-switched’ and hence the weights

    novCEA["num"] = 1 - novCEA["pxo1"]
    novCEA["denom"] = 1 - novCEA["pxo2"]
    prod = novCEA.sort_values(["id", "time"]).groupby("id")[["num", "denom"]].cumprod()
    novCEA["num"] = prod["num"]
    novCEA["denom"] = prod["denom"]

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
    novCEA_KM["tout"] = pd.concat([(novCEA_KM.time + timesteps_per_intervention), novCEA_KM.fault_time], axis=1).min(
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


if __name__ == "__main__":
    df = pd.read_csv("data/long_data.csv")
    control = OrderedDict([("MV101", 1), ("P101", 0), ("P102", 0)])
    timesteps_per_intervention = 15
    timesteps = (len(control) + 1) * timesteps_per_intervention
    outcome = "LIT101"

    treatment = control.copy()
    treatment["P101"] = 1

    control_strategy = list(
        zip(
            control,
            range(timesteps_per_intervention, timesteps + 1, timesteps_per_intervention),
            control.values(),
        )
    )
    treatment_strategy = list(
        zip(
            treatment,
            range(timesteps_per_intervention, timesteps + 1, timesteps_per_intervention),
            treatment.values(),
        )
    )
    preprocess_data(df, control_strategy, treatment_strategy, outcome)
    novCEA = pd.read_csv("data/long_preprocessed.csv")
    estimate_hazard_ratio(novCEA, timesteps_per_intervention)
