import pandas as pd
import statsmodels.formula.api as smf
from numpy import exp, append, nan
from lifelines import CoxPHFitter, CoxTimeVaryingFitter
from lifelines.exceptions import ConvergenceError
import matplotlib.pyplot as plt
import argparse
from safe_ranges import safe_ranges
from collections import OrderedDict
from tqdm import tqdm
from scipy.stats import kurtosis
import networkx as nx
import numpy as np
from multiprocessing import Pool
import warnings
import jsonpickle
from patsy import dmatrix


class Capability:
    def __init__(self, variable, value, time):
        self.variable = variable
        self.value = value
        self.time = time

    def __eq__(self, other):
        return (
            self.variable == other.variable
            and self.value == other.value
            and self.time == other.time
        )

    def __repr__(self):
        return f"({self.variable}, {self.value}, {self.time})"


class CapabilityList:
    def __init__(self, timesteps_per_intervention, capabilities):
        self.timesteps_per_intervention = timesteps_per_intervention
        self.capabilities = [
            Capability(var, val, t)
            for (var, val), t in zip(
                capabilities,
                range(
                    timesteps_per_intervention,
                    (len(capabilities) * timesteps_per_intervention) + 1,
                    timesteps_per_intervention,
                ),
            )
        ]

    def set_value(self, inx, value):
        self.capabilities[inx].value = value

    def treatment_strategy(self, inx, value):
        strategy = CapabilityList(
            self.timesteps_per_intervention,
            [(c.variable, c.value) for c in self.capabilities],
        )
        strategy.set_value(inx, value)
        return strategy

    def total_time(self):
        return (len(self.capabilities) + 1) * self.timesteps_per_intervention


def setup_xo_t_do(strategy_assigned, strategy_followed):
    mask = (pd.Series(strategy_assigned) != pd.Series(strategy_followed)).astype(
        "boolean"
    )
    false = mask.loc[mask == True]
    if false.empty:
        return np.zeros(len(mask) + 2)
    else:
        mask.loc[false.index[0] + 1 :] = None
        mask = (mask * 1).replace(np.nan, None).to_list()
        return [0] + mask + [None]


def setup_fault_t_do(values, min, max):
    fault_occurred = False
    result = []
    for value in values:
        fault = not (min <= value <= max)
        result.append((not fault_occurred) and fault)
        fault_occurred = fault_occurred or fault
    return [int(x) for x in result]


def setup_fault_time(values, min, max):
    fault_occurred = False
    result = []
    fault_time = None
    for inx, value in values.items():
        if not (min <= value <= max):
            return inx
    return fault_time


def setup_fault_time_fast(
    individual, timesteps_per_intervention=15, perturbation=-0.001
):
    fault = individual[individual["within_safe_range"] == False]
    fault_time = (
        individual.time.loc[fault.index[0]]
        if not fault.empty
        else (individual.time.max() + timesteps_per_intervention)
    )
    return pd.DataFrame(
        {"fault_time": np.repeat(fault_time + perturbation, len(individual))}
    )


def setup_recent_prog_t_dc(now_prog_t_dc):
    result = []
    prog = False
    for value in now_prog_t_dc:
        prog = prog or value
        result.append(prog)
    return result


def preprocess_data(
    df, control_strategy, treatment_strategy, outcome, timesteps_per_intervention
):
    df["trtrand"] = None  # treatment/control arm
    df["fault_t_do"] = None  # did a fault occur here?
    df[
        "xo_t_do"
    ] = None  # did the individual deviate from the treatment of interest here?
    df["now_prog_t_dc"] = None  # has the situation progressed now?
    df["recent_prog_t_dc"] = None  # has the situation progressed in the past?
    # df["fault_time"] = None  # when did a fault occur?

    df["within_safe_range"] = df[outcome].between(
        safe_ranges[outcome]["lo"], safe_ranges[outcome]["hi"]
    )
    df["fault_time"] = (
        df.groupby("id")[["within_safe_range", "time"]]
        .apply(setup_fault_time_fast)
        .values
    )
    assert not pd.isnull(df["fault_time"]).any()

    individuals = []
    new_id = 0
    print("  Preprocessing groups")
    for id, individual in tqdm(df.query("fault_time > 0").groupby("id")):
        individual = individual.loc[
            (individual.time % timesteps_per_intervention == 0)
            & (individual.time <= control_strategy.total_time())
        ].copy()

        individual["fault_t_do"] = setup_fault_t_do(
            individual[outcome], safe_ranges[outcome]["lo"], safe_ranges[outcome]["hi"]
        )
        assert (
            sum(individual["fault_t_do"]) <= 1
        ), f"Error initialising fault_t_do for id {id} with fault at {individual.fault_time.iloc[0]}"
        individual["now_prog_t_dc"] = setup_fault_t_do(
            individual[outcome], safe_ranges[outcome]["lo"], safe_ranges[outcome]["hi"]
        )
        individual["recent_prog_t_dc"] = setup_recent_prog_t_dc(
            individual["now_prog_t_dc"]
        )
        faulty = individual.loc[
            ~individual[outcome].between(
                safe_ranges[outcome]["lo"], safe_ranges[outcome]["hi"]
            ),
            "time",
        ]

        # Control flow:
        # Individuals that start off in both arms, need cloning (hence incrementing the ID within the if statements)
        # Individuals that don't start off in either arm need leaving out (hence two ifs rather than elif or else)
        strategy = [
            Capability(
                c.variable,
                individual.loc[individual.time == c.time, c.variable].values[0],
                c.time,
            )
            for c in treatment_strategy.capabilities
        ]
        if strategy[0] == control_strategy.capabilities[0]:
            individual["id"] = id
            id += 1
            individual["trtrand"] = 0
            individual["xo_t_do"] = setup_xo_t_do(
                strategy, control_strategy.capabilities
            )
            individuals.append(
                individual.loc[individual.time <= individual.fault_time].copy()
            )
        if strategy[0] == treatment_strategy.capabilities[0]:
            individual["id"] = id
            id += 1
            individual["trtrand"] = 1
            individual["xo_t_do"] = setup_xo_t_do(
                strategy, treatment_strategy.capabilities
            )
            individuals.append(
                individual.loc[individual.time <= individual.fault_time].copy()
            )
    print(df[[outcome, "within_safe_range", "fault_time"]])
    df = pd.concat(individuals)
    df.to_csv("data/long_preprocessed.csv", index=False)


def estimate_hazard_ratio(
    novCEA,
    timesteps_per_intervention,
    fitBLswitch_formula,
    fitBLTDswitch_formula,
    print_summary=False,
):
    # Use logistic regression to predict switching given baseline covariates
    print(f"  predict switching given baseline covariates: {fitBLswitch_formula}")
    fitBLswitch = smf.logit(fitBLswitch_formula, data=novCEA).fit()

    # Estimate the probability of switching for each patient-observation included in the regression.
    novCEA["pxo1"] = fitBLswitch.predict(novCEA)

    # Use logistic regression to predict switching given baseline and time-updated covariates (model S12)
    print(
        f"  predict switching given baseline and time-updated covariates: {fitBLTDswitch_formula}"
    )
    # Covariance matrix to examine colinearities
    # relevant_features = fitBLTDswitch_formula.split(" ~ ")[1].split(" + ")
    # novCEA[relevant_features].corr().round(3).to_csv("/tmp/corr.csv")

    try:
        fitBLTDswitch = smf.logit(
            fitBLTDswitch_formula,
            data=novCEA,  # [(novCEA.trtrand == 0) & (novCEA["recent_prog_t_dc"] == 1)],
        ).fit()
    except np.linalg.LinAlgError:
        return None, None

    # Estimate the probability of switching for each patient-observation included in the regression.
    novCEA["pxo2"] = fitBLTDswitch.predict(novCEA)

    # set prob of switching to zero where progtypetdc==0
    # novCEA.loc[(novCEA.trtrand == 0) & (novCEA["recent_prog_t_dc"] == 0), "pxo2"] = 0

    # IPCW step 3: For each individual at each time, compute the inverse probability of remaining uncensored
    # Estimate the probabilities of remaining ‘un-switched’ and hence the weights

    novCEA["num"] = 1 - novCEA["pxo1"]
    novCEA["denom"] = 1 - novCEA["pxo2"]
    prod = novCEA.sort_values(["id", "time"]).groupby("id")[["num", "denom"]].cumprod()
    novCEA["num"] = prod["num"]
    novCEA["denom"] = prod["denom"]

    assert (
        not novCEA["num"].isnull().any()
    ), f"{len(novCEA['num'].isnull())} null numerator values"
    assert (
        not novCEA["denom"].isnull().any()
    ), f"{len(novCEA['denom'].isnull())} null denom values"

    isControl = novCEA.trtrand == 0
    novCEA.loc[isControl, "weight"] = 1 / novCEA.denom[isControl]
    novCEA.loc[isControl, "sweight"] = novCEA.num[isControl] / novCEA.denom[isControl]

    # set the weights to 1 in the treatment arm
    novCEA.loc[~isControl, "weight"] = 1
    novCEA.loc[~isControl, "sweight"] = 1

    novCEA_KM = novCEA.loc[novCEA.xo_t_do == 0].copy()
    novCEA_KM["tin"] = novCEA_KM.time
    novCEA_KM["tout"] = pd.concat(
        [(novCEA_KM.time + timesteps_per_intervention), novCEA_KM.fault_time], axis=1
    ).min(axis=1)

    novCEA_KM.to_csv("/tmp/novCEA_KM.csv")

    assert (
        novCEA_KM["tin"] <= novCEA_KM["tout"]
    ).all(), (
        f"Left before joining\n{novCEA_KM.loc[novCEA_KM['tin'] >= novCEA_KM['tout']]}"
    )

    novCEA_KM.dropna(axis=1, inplace=True)
    novCEA_KM.replace([float("inf")], 100, inplace=True)

    #  IPCW step 4: Use these weights in a weighted analysis of the outcome model
    # Estimate the KM graph and IPCW hazard ratio using Cox regression.
    cox_ph = CoxPHFitter()
    try:
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
    except ConvergenceError:
        print("ConvergenceError: Unable to estimate hazard ratio")
        return None, None
    if print_summary:
        cox_ph.print_summary()
    return cox_ph.params_, cox_ph.confidence_intervals_


if __name__ == "__main__":
    df = pd.read_csv("data/long_data.csv")
    dag = nx.nx_pydot.read_dot("flow_raw.dot")
    num_repeats = 100
    with open("successful_attacks.json") as f:
        successful_attacks = jsonpickle.decode("".join(f.readlines()))
    data = []
    adequacy = False

    timesteps_per_intervention = 15

    # These all work
    # successful_attacks = {
    #     "LIT101 (High)": [[["MV101", 1]], [["MV101", 1], ["P101", 0], ["P102", 0]], [["MV101", 1], ["MV201", 0]]],
    # }

    for outcome, attacks in successful_attacks.items():
        print()
        print("OUTCOME", outcome)
        outcome = outcome.split(" ")[0]

        if outcome not in df:
            print(f"Missing data for {outcome}")
            continue

        if (
            len(
                df.loc[
                    (df[outcome] < safe_ranges[outcome]["lo"])
                    | (df[outcome] > safe_ranges[outcome]["hi"])
                ]
            )
            == 0
        ):
            print(
                f"  No faults with {outcome}. Cannot perform estimation.\n"
                f"  Observed range [{df[outcome].min()}, {df[outcome].max()}].\n"
                f"  Safe range {safe_ranges[outcome]}"
            )
            continue
        if len(
            df.loc[
                (df[outcome] < safe_ranges[outcome]["lo"])
                | (df[outcome] > safe_ranges[outcome]["hi"])
            ]
        ) == len(df):
            print(
                f"  All faults with {outcome}. Cannot perform estimation.\n"
                f"  Observed range [{df[outcome].min()}, {df[outcome].max()}].\n"
                f"  Safe range {safe_ranges[outcome]}"
            )
            continue

        for capabilities in attacks:
            control_strategy = CapabilityList(timesteps_per_intervention, capabilities)
            print("  CONTROL STRATEGY", control_strategy.capabilities)

            if any(c.variable not in df for c in control_strategy.capabilities):
                print("  Missing data for control_strategy")
                continue
            if any(c.variable not in dag.nodes for c in control_strategy.capabilities):
                print("  Missing node for control_strategy")
                break

            for i, capability in enumerate(control_strategy.capabilities):
                datum = {"control_strategy": control_strategy.capabilities}
                datum = {"outcome": outcome}

                # Treatment strategy is the same, but with one capability negated
                # i.e. we examine the counterfactual "What if we had not done that?"
                treatment_strategy = control_strategy.treatment_strategy(
                    i, int(not capability.value)
                )
                print("  TREATMENT STRATEGY", treatment_strategy.capabilities)
                datum["treatment_strategy"] = treatment_strategy.capabilities
                print(
                    "SAFE RANGE", safe_ranges[outcome]["lo"], safe_ranges[outcome]["hi"]
                )
                preprocess_data(
                    df,
                    control_strategy,
                    treatment_strategy,
                    outcome,
                    timesteps_per_intervention,
                )

                novCEA = pd.read_csv("data/long_preprocessed.csv")

                print(
                    f"  {novCEA['fault_t_do'].sum()}/{len(novCEA.groupby('id'))} faulty runs observed"
                )

                for id, group in novCEA.groupby("id"):
                    assert (
                        sum(group["fault_t_do"]) <= 1
                    ), f"Multiple fault times for id {id}\n{group}"

                fitBLswitch_formula = "xo_t_do ~ time"

                neighbours = list(dag.predecessors(capability.variable))
                neighbours += list(dag.successors(capability.variable))

                if capability.variable == "P402":
                    neighbours.remove("FIT501")
                    neighbours.remove("AIT402")
                    neighbours.remove("FIT401")

                assert (
                    len(neighbours) > 0
                ), f"No neighbours for node {capability.variable}"

                fitBLTDswitch_formula = (
                    f"{fitBLswitch_formula} + {' + '.join(neighbours)}"
                )
                datum["fitBLTDswitch_formula"] = fitBLTDswitch_formula

                params, confidence_intervals = estimate_hazard_ratio(
                    novCEA,
                    timesteps_per_intervention,
                    fitBLswitch_formula,
                    fitBLTDswitch_formula,
                )
                if params is None:
                    print("  FAILURE: Params was None")
                    continue
                datum["risk_ratio"] = params.to_dict()
                datum["risk_ratio"] = confidence_intervals.to_dict()

                if adequacy:

                    def estimate_hazard_ratio_parallel(ids):
                        try:
                            params, _ = estimate_hazard_ratio(
                                novCEA[novCEA["id"].isin(ids)].copy(),
                                timesteps_per_intervention,
                                fitBLswitch_formula,
                                fitBLTDswitch_formula,
                            )
                            return params
                        except np.linalg.LinAlgError:
                            return None
                        except ValueError:
                            return None

                    ids = list(set(novCEA["id"]))

                    pool = Pool()
                    params_repeats = pool.map(
                        estimate_hazard_ratio_parallel,
                        [
                            np.random.choice(ids, len(ids), replace=True)
                            for x in range(num_repeats)
                        ],
                    )
                    params_repeats = [x for x in params_repeats if x is not None]
                    datum["params_repeats"] = [p.to_dict() for p in params_repeats]

                    datum["mean_risk_ratio"] = (
                        sum(params_repeats) / len(params_repeats)
                    ).to_dict()
                    datum["kurtosis"] = kurtosis(params_repeats).tolist()
                    datum["successes"] = len(params_repeats)
                data.append(datum)
                print(" ", datum)
                with open("logs/output_no_filter_lo_hi_sim.json", "w") as f:
                    print(jsonpickle.encode(data, indent=2, unpicklable=False), file=f)
