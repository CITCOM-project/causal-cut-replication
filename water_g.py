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
import logging
from math import ceil

logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    handlers=[logging.FileHandler("logs/debug.log"), logging.StreamHandler()],
)


class Capability:
    def __init__(self, variable, value, time):
        self.variable = variable
        self.value = value
        self.time = time

    def __eq__(self, other):
        return self.variable == other.variable and self.value == other.value and self.time == other.time

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


def setup_xo_t_do(strategy_assigned: list, strategy_followed: list, eligible: pd.Series):
    strategy_assigned = [1] + strategy_assigned + [1]
    strategy_followed = [1] + strategy_followed + [1]

    mask = (
        pd.Series(strategy_assigned, index=eligible.index) != pd.Series(strategy_followed, index=eligible.index)
    ).astype("boolean")
    mask = mask | ~eligible
    mask.reset_index(inplace=True, drop=True)
    false = mask.loc[mask == True]
    if false.empty:
        return np.zeros(len(mask))
    else:
        mask = (mask * 1).tolist()
        cutoff = false.index[0] + 1
        return mask[:cutoff] + ([None] * (len(mask) - cutoff))


def setup_fault_t_do(individual, timesteps_per_intervention=15, perturbation=-0.001):
    fault = individual[individual["within_safe_range"] == False]
    fault_t_do = pd.Series(np.zeros(len(individual)), index=individual.index)

    if not fault.empty:
        fault_time = individual["time"].loc[fault.index[0]]
        # Ceiling to nearest observation point
        fault_time = ceil(fault_time / timesteps_per_intervention) * timesteps_per_intervention
        # Set the correct observation point to be the fault time of doing (fault_t_do)
        observations = individual.loc[
            (individual.time % timesteps_per_intervention == 0) & (individual.time < fault_time)
        ]
        if not observations.empty:
            fault_t_do.loc[observations.index[0]] = 1

    return pd.DataFrame({"fault_t_do": fault_t_do})


def setup_fault_time(individual, timesteps_per_intervention=15, perturbation=-0.001):
    fault = individual[individual["within_safe_range"] == False]
    fault_time = (
        individual.time.loc[fault.index[0]] if not fault.empty else (individual.time.max() + timesteps_per_intervention)
    )
    return pd.DataFrame({"fault_time": np.repeat(fault_time + perturbation, len(individual))})


def preprocess_data(
    df, control_strategy, treatment_strategy, outcome, min, max, timesteps_per_intervention, eligibility=None
):
    df["trtrand"] = None  # treatment/control arm
    df["xo_t_do"] = None  # did the individual deviate from the treatment of interest here?
    df["eligible"] = df.eval(eligibility) if eligibility is not None else True

    # when did a fault occur?
    df["within_safe_range"] = df[outcome].between(min, max)
    df["fault_time"] = df.groupby("id")[["within_safe_range", "time"]].apply(setup_fault_time).values
    df["fault_t_do"] = df.groupby("id")[["id", "time", "within_safe_range"]].apply(setup_fault_t_do).values
    assert not pd.isnull(df["fault_time"]).any()

    living_runs = df.query("fault_time > 0").loc[
        (df.time % timesteps_per_intervention == 0) & (df.time <= control_strategy.total_time())
    ]

    individuals = []
    new_id = 0
    logging.debug("  Preprocessing groups")
    for id, individual in living_runs.groupby("id"):
        assert (
            sum(individual["fault_t_do"]) <= 1
        ), f"Error initialising fault_t_do for individual\n{individual[['id', 'time', 'fault_time', 'fault_t_do']]}\nwith fault at {individual.fault_time.iloc[0]}"

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

        if strategy[0] == control_strategy.capabilities[0] and individual.eligible.iloc[0]:
            individual["old_id"] = individual["id"]
            individual["id"] = new_id
            new_id += 1
            individual["trtrand"] = 0
            individual["xo_t_do"] = setup_xo_t_do(control_strategy.capabilities, strategy, individual.eligible)
            individuals.append(individual.loc[individual.time <= individual.fault_time].copy())
        if strategy[0] == treatment_strategy.capabilities[0] and individual.eligible.iloc[0]:
            individual["old_id"] = individual["id"]
            individual["id"] = new_id
            new_id += 1
            individual["trtrand"] = 1
            individual["xo_t_do"] = setup_xo_t_do(treatment_strategy.capabilities, strategy, individual.eligible)
            individuals.append(individual.loc[individual.time <= individual.fault_time].copy())
    if len(individuals) == 0:
        logging.debug("No individuals followed either strategy.")
        return None
    df = pd.concat(individuals)
    df.to_csv("data/long_preprocessed.csv", index=False)
    return df


def estimate_hazard_ratio(
    novCEA,
    timesteps_per_intervention,
    fitBLswitch_formula,
    fitBLTDswitch_formula,
    print_summary=False,
):
    # Use logistic regression to predict switching given baseline covariates
    logging.debug(f"  predict switching given baseline covariates: {fitBLswitch_formula}")
    try:
        fitBLswitch = smf.logit(fitBLswitch_formula, data=novCEA).fit()
    except np.linalg.LinAlgError:
        logging.error("Could not predict switching given baseline covariates")
        return None, None

    # Estimate the probability of switching for each patient-observation included in the regression.
    novCEA["pxo1"] = fitBLswitch.predict(novCEA)

    # Use logistic regression to predict switching given baseline and time-updated covariates (model S12)
    logging.debug(f"  predict switching given baseline and time-updated covariates: {fitBLTDswitch_formula}")
    # Covariance matrix to examine colinearities
    # relevant_features = fitBLTDswitch_formula.split(" ~ ")[1].split(" + ")
    # novCEA[relevant_features].corr().round(3).to_csv("/tmp/corr.csv")

    try:
        fitBLTDswitch = smf.logit(
            fitBLTDswitch_formula,
            data=novCEA,
        ).fit()
    except np.linalg.LinAlgError:
        logging.error("Could not predict switching given baseline and time-updated covariates")
        return None, None

    # Estimate the probability of switching for each patient-observation included in the regression.
    novCEA["pxo2"] = fitBLTDswitch.predict(novCEA)

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

    novCEA_KM.to_csv("/tmp/novCEA_KM.csv")

    assert (
        novCEA_KM["tin"] <= novCEA_KM["tout"]
    ).all(), f"Left before joining\n{novCEA_KM.loc[novCEA_KM['tin'] >= novCEA_KM['tout']]}"

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
        logging.error("ConvergenceError: Unable to estimate hazard ratio")
        return None, None
    if print_summary:
        cox_ph.print_summary()
    return cox_ph.params_, cox_ph.confidence_intervals_


if __name__ == "__main__":
    # df = pd.read_csv("data/long_data.csv")
    df = pd.read_parquet("data/long_data.pqt")
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
        outcome = outcome.split(" ")[0]
        logging.debug(f"\nOUTCOME: {outcome}")
        if outcome not in df:
            logging.warning(f"Missing data for {outcome}")
            continue

        min, max = safe_ranges[outcome]["lo"], safe_ranges[outcome]["hi"]
        # if not (~df[outcome].between(min, max)).any():
        #     logging.warning("USING WEAKER BOUNDS")
        #     min, max = safe_ranges[outcome]["lo"], safe_ranges[outcome]["hi"]
        if not (~df[outcome].between(min, max)).any():
            logging.error(
                f"  No faults with {outcome}. Cannot perform estimation.\n"
                f"  Observed range [{df[outcome].min()}, {df[outcome].max()}].\n"
                f"  Safe range {safe_ranges[outcome]}"
            )
            continue
        if df[outcome].between(min, max).all():
            logging.error(
                f"  All faults with {outcome}. Cannot perform estimation.\n"
                f"  Observed range [{df[outcome].min()}, {df[outcome].max()}].\n"
                f"  Safe range {safe_ranges[outcome]}"
            )
            continue

        for capabilities in attacks:
            control_strategy = CapabilityList(timesteps_per_intervention, capabilities)
            logging.debug(f"  CONTROL STRATEGY   {control_strategy.capabilities}")

            if any(c.variable not in df for c in control_strategy.capabilities):
                logging.error("  Missing data for control_strategy")
                continue
            if any(c.variable not in dag.nodes for c in control_strategy.capabilities):
                logging.error("  Missing node for control_strategy")
                break

            for i, capability in enumerate(control_strategy.capabilities):
                datum = {"control_strategy": control_strategy.capabilities}
                datum = {"outcome": outcome}

                # Treatment strategy is the same, but with one capability negated
                # i.e. we examine the counterfactual "What if we had not done that?"
                treatment_strategy = control_strategy.treatment_strategy(i, int(not capability.value))
                logging.debug(f"  TREATMENT STRATEGY {treatment_strategy.capabilities}")
                datum["treatment_strategy"] = treatment_strategy.capabilities
                logging.debug(f"  OUTCOME {outcome}")
                logging.debug(f"  SAFE RANGE {min} {max}")
                novCEA = preprocess_data(
                    df,
                    control_strategy,
                    treatment_strategy,
                    outcome,
                    min,
                    max,
                    timesteps_per_intervention,
                    None
                    # safe_ranges[outcome].get("eligibility", None),
                )
                if novCEA is None:
                    logging.error("No eligible individuals")
                    continue

                # novCEA = pd.read_csv("data/long_preprocessed.csv")
                logging.debug(
                    f'  NOVCEA\n{novCEA[["id", "time",capability.variable,outcome, "xo_t_do", "fault_time", "fault_t_do"]]}'
                )

                logging.debug(f"  {int(novCEA['fault_t_do'].sum())}/{len(novCEA.groupby('id'))} faulty runs observed")
                if novCEA["fault_t_do"].sum() == 0:
                    break

                for id, group in novCEA.groupby("id"):
                    assert sum(group["fault_t_do"]) <= 1, f"Multiple fault times for id {id}\n{group}"

                fitBLswitch_formula = "xo_t_do ~ time"

                neighbours = list(dag.predecessors(capability.variable))
                neighbours += list(dag.successors(capability.variable))

                if capability.variable == "P402":
                    neighbours.remove("FIT501")
                    neighbours.remove("AIT402")
                    neighbours.remove("FIT401")

                assert len(neighbours) > 0, f"No neighbours for node {capability.variable}"

                fitBLTDswitch_formula = f"{fitBLswitch_formula} + {' + '.join(neighbours)}"
                datum["fitBLTDswitch_formula"] = fitBLTDswitch_formula

                params, confidence_intervals = estimate_hazard_ratio(
                    novCEA,
                    timesteps_per_intervention,
                    fitBLswitch_formula,
                    fitBLTDswitch_formula,
                )
                if params is None:
                    logging.error("  FAILURE: Params was None")
                    continue
                datum["risk_ratio"] = params.to_dict()
                datum["risk_ratio"] = confidence_intervals.to_dict()
                datum["capability"] = {"index": i} | capability.__dict__

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
                        [np.random.choice(ids, len(ids), replace=True) for x in range(num_repeats)],
                    )
                    params_repeats = [x for x in params_repeats if x is not None]
                    datum["params_repeats"] = [p.to_dict() for p in params_repeats]

                    datum["mean_risk_ratio"] = (sum(params_repeats) / len(params_repeats)).to_dict()
                    datum["kurtosis"] = kurtosis(params_repeats).tolist()
                    datum["successes"] = len(params_repeats)
                data.append(datum)
                logging.debug(f"  {datum}")
                with open("logs/output_no_filter_lo_hi_sim.json", "w") as f:
                    print(jsonpickle.encode(data, indent=2, unpicklable=False), file=f)
