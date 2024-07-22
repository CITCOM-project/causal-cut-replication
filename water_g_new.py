import pandas as pd
import statsmodels.formula.api as smf
from numpy import exp, append, nan
from lifelines import CoxPHFitter
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
from copy import deepcopy


from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.estimators import Estimator
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.base_test_case import BaseTestCase


np.random.seed(0)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    handlers=[logging.FileHandler("logs/debug.log"), logging.StreamHandler()],
)


class Capability:
    """
    Data class to encapsulate temporal interventions.
    """

    def __init__(self, variable, value, start_time, end_time):
        self.variable = variable
        self.value = value
        self.start_time = start_time
        self.end_time = end_time

    def __eq__(self, other):
        return (
            type(other) == type(self)
            and self.variable == other.variable
            and self.value == other.value
            and self.start_time == other.start_time
            and self.end_time == other.end_time
        )

    def __repr__(self):
        return f"({self.variable}, {self.value}, {self.start_time}-{self.end_time})"


class TreatmentSequence:
    """
    Class to represent a list of capabilities, i.e. a treatment regime.
    """

    def __init__(self, timesteps_per_intervention, capabilities):
        self.timesteps_per_intervention = timesteps_per_intervention
        self.capabilities = [
            Capability(var, val, t, t + timesteps_per_intervention)
            for (var, val), t in zip(
                capabilities,
                range(
                    timesteps_per_intervention,
                    (len(capabilities) * timesteps_per_intervention) + 1,
                    timesteps_per_intervention,
                ),
            )
        ]

    def set_value(self, index: int, value: float):
        """
        Set the value of capability at the given index.
        :param index - the index of the element to update.
        :param value - the desired value of the capability.
        """
        self.capabilities[index].value = value

    def copy(self):
        """
        Return a deep copy of the capability list.
        """
        strategy = TreatmentSequence(
            self.timesteps_per_intervention,
            [(c.variable, c.value) for c in self.capabilities],
        )
        return strategy

    def total_time(self):
        """
        Calculate the total duration of the treatment strategy.
        """
        return (len(self.capabilities) + 1) * self.timesteps_per_intervention


class IPCWEstimator(Estimator):
    """
    Class to perform inverse probability of censoring weighting (IPCW) estimation
    for sequences of treatments over time-varying data.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        timesteps_per_intervention: int,
        control_strategy: TreatmentSequence,
        treatment_strategy: TreatmentSequence,
        outcome: str,
        min: float,
        max: float,
        fitBLswitch_formula: str,
        fitBLTDswitch_formula: str,
        eligibility=None,
    ):
        self.timesteps_per_intervention = timesteps_per_intervention
        self.control_strategy = control_strategy
        self.treatment_strategy = treatment_strategy
        self.outcome = outcome
        self.min = min
        self.max = max
        self.timesteps_per_intervention = timesteps_per_intervention
        self.fitBLswitch_formula = fitBLswitch_formula
        self.fitBLTDswitch_formula = fitBLTDswitch_formula
        self.eligibility = eligibility
        self.df = self.preprocess_data(df)

    def add_modelling_assumptions(self):
        self.modelling_assumptions.append("The variables in the data vary over time.")

    def setup_xo_t_do(self, strategy_assigned: list, strategy_followed: list, eligible: pd.Series):
        """
        Return a binary sequence with each bit representing whether the current
        index is the time point at which the individual diverted from the
        assigned treatment strategy (and thus should be censored).

        :param strategy_assigned - the assigned treatment strategy
        :param strategy_followed - the strategy followed by the individual
        :param eligible - binary sequence represnting the eligibility of the individual at each time step
        """
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

    def setup_fault_t_do(self, individual: pd.DataFrame):
        """
        Return a binary sequence with each bit representing whether the current
        index is the time point at which the event of interest (i.e. a fault)
        occurred.
        """
        fault = individual[individual["within_safe_range"] == False]
        fault_t_do = pd.Series(np.zeros(len(individual)), index=individual.index)

        if not fault.empty:
            fault_time = individual["time"].loc[fault.index[0]]
            # Ceiling to nearest observation point
            fault_time = ceil(fault_time / self.timesteps_per_intervention) * self.timesteps_per_intervention
            # Set the correct observation point to be the fault time of doing (fault_t_do)
            observations = individual.loc[
                (individual["time"] % self.timesteps_per_intervention == 0) & (individual["time"] < fault_time)
            ]
            if not observations.empty:
                fault_t_do.loc[observations.index[0]] = 1
        assert sum(fault_t_do) <= 1, f"Multiple fault times for\n{individual}"

        return pd.DataFrame({"fault_t_do": fault_t_do})

    def setup_fault_time(self, individual, perturbation=-0.001):
        """
        Return the time at which the event of interest (i.e. a fault) occurred.
        """
        fault = individual[individual["within_safe_range"] == False]
        fault_time = (
            individual["time"].loc[fault.index[0]]
            if not fault.empty
            else (individual["time"].max() + self.timesteps_per_intervention)
        )
        return pd.DataFrame({"fault_time": np.repeat(fault_time + perturbation, len(individual))})

    def preprocess_data(self, df):
        """
        Set up the treatment-specific columns in the data that are needed to estimate the hazard ratio.
        """
        df["trtrand"] = None  # treatment/control arm
        df["xo_t_do"] = None  # did the individual deviate from the treatment of interest here?
        df["eligible"] = df.eval(self.eligibility) if self.eligibility is not None else True

        # when did a fault occur?
        df["within_safe_range"] = df[self.outcome].between(self.min, self.max)
        df["fault_time"] = df.groupby("id")[["within_safe_range", "time"]].apply(self.setup_fault_time).values
        df["fault_t_do"] = df.groupby("id")[["id", "time", "within_safe_range"]].apply(self.setup_fault_t_do).values
        assert not pd.isnull(df["fault_time"]).any()

        living_runs = df.query("fault_time > 0").loc[
            (df["time"] % self.timesteps_per_intervention == 0) & (df["time"] <= self.control_strategy.total_time())
        ]

        individuals = []
        new_id = 0
        logging.debug("  Preprocessing groups")
        for id, individual in living_runs.groupby("id"):
            assert (
                sum(individual["fault_t_do"]) <= 1
            ), f"Error initialising fault_t_do for individual\n{individual[['id', 'time', 'fault_time', 'fault_t_do']]}\nwith fault at {individual.fault_time.iloc[0]}"

            strategy_followed = [
                Capability(
                    c.variable,
                    individual.loc[individual["time"] == c.start_time, c.variable].values[0],
                    c.start_time,
                    c.end_time,
                )
                for c in self.treatment_strategy.capabilities
            ]

            # Control flow:
            # Individuals that start off in both arms, need cloning (hence incrementing the ID within the if statement)
            # Individuals that don't start off in either arm are left out
            for inx, strategy_assigned in [(0, control_strategy), (1, self.treatment_strategy)]:
                if strategy_assigned.capabilities[0] == strategy_followed[0] and individual.eligible.iloc[0]:
                    individual["id"] = new_id
                    new_id += 1
                    individual["trtrand"] = inx
                    individual["xo_t_do"] = self.setup_xo_t_do(
                        strategy_assigned.capabilities, strategy_followed, individual["eligible"]
                    )
                    individuals.append(individual.loc[individual["time"] <= individual["fault_time"]].copy())
        if len(individuals) == 0:
            raise ValueError("No individuals followed either strategy.")
        return pd.concat(individuals)

    def estimate_hazard_ratio(self):
        """
        Estimate the hazard ratio.
        """

        if self.df["fault_t_do"].sum() == 0:
            raise ValueError("No recorded faults")

        logging.debug(f"  {int(self.df['fault_t_do'].sum())}/{len(self.df.groupby('id'))} faulty runs observed")

        # Use logistic regression to predict switching given baseline covariates
        logging.debug(f"  predict switching given baseline covariates: {self.fitBLswitch_formula}")
        try:
            print(self.df)
            print(self.fitBLswitch_formula)
            fitBLswitch = smf.logit(self.fitBLswitch_formula, data=self.df).fit()
        except np.linalg.LinAlgError:
            logging.error("Could not predict switching given baseline covariates")
            return None

        # Estimate the probability of switching for each patient-observation included in the regression.
        novCEA = pd.DataFrame()
        novCEA["pxo1"] = fitBLswitch.predict(self.df)

        # Use logistic regression to predict switching given baseline and time-updated covariates (model S12)
        logging.debug(f"  predict switching given baseline and time-updated covariates: {self.fitBLTDswitch_formula}")

        try:
            fitBLTDswitch = smf.logit(
                self.fitBLTDswitch_formula,
                data=self.df,
            ).fit()
        except np.linalg.LinAlgError:
            logging.error("Could not predict switching given baseline and time-updated covariates")
            return None

        # Estimate the probability of switching for each patient-observation included in the regression.
        novCEA["pxo2"] = fitBLTDswitch.predict(self.df)

        # IPCW step 3: For each individual at each time, compute the inverse probability of remaining uncensored
        # Estimate the probabilities of remaining ‘un-switched’ and hence the weights

        novCEA["num"] = 1 - novCEA["pxo1"]
        novCEA["denom"] = 1 - novCEA["pxo2"]
        prod = (
            pd.concat([self.df, novCEA], axis=1).sort_values(["id", "time"]).groupby("id")[["num", "denom"]].cumprod()
        )
        novCEA["num"] = prod["num"]
        novCEA["denom"] = prod["denom"]

        assert not novCEA["num"].isnull().any(), f"{len(novCEA['num'].isnull())} null numerator values"
        assert not novCEA["denom"].isnull().any(), f"{len(novCEA['denom'].isnull())} null denom values"

        novCEA["weight"] = 1 / novCEA["denom"]
        novCEA["sweight"] = novCEA["num"] / novCEA["denom"]

        novCEA_KM = novCEA.loc[self.df["xo_t_do"] == 0].copy()
        novCEA_KM["tin"] = self.df["time"]
        novCEA_KM["tout"] = pd.concat(
            [(self.df["time"] + self.timesteps_per_intervention), self.df["fault_time"]], axis=1
        ).min(axis=1)

        # novCEA_KM.to_csv("/tmp/novCEA_KM.csv")

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
                df=pd.concat([self.df, novCEA_KM], axis=1),
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
            return None
        return pd.concat([cox_ph.hazard_ratios_, np.exp(cox_ph.confidence_intervals_)], axis=1)


if __name__ == "__main__":
    # df = pd.read_csv("data/long_data.csv")
    df = pd.read_parquet("data/long_data.pqt")
    dag = nx.nx_pydot.read_dot("flow_raw.dot")
    num_repeats = 100
    with open("successful_attacks.json") as f:
        successful_attacks = jsonpickle.decode("".join(f.readlines()))
    data = []
    adequacy = True

    timesteps_per_intervention = 15

    for outcome, attacks in successful_attacks.items():
        outcome, attack = outcome.split(" ")
        logging.debug(f"\nOUTCOME: {outcome}")
        if outcome not in df:
            logging.warning(f"Missing data for {outcome}")
            continue

        min, max = safe_ranges[outcome]["lo"], safe_ranges[outcome]["hi"]
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
            control_strategy = TreatmentSequence(timesteps_per_intervention, capabilities)
            logging.debug(f"  CONTROL STRATEGY   {control_strategy.capabilities}")

            if any(c.variable not in df for c in control_strategy.capabilities):
                logging.error("  Missing data for control_strategy")
                continue
            if any(c.variable not in dag.nodes for c in control_strategy.capabilities):
                logging.error("  Missing node for control_strategy")
                break

            for i, capability in enumerate(control_strategy.capabilities):
                # Treatment strategy is the same, but with one capability negated
                # i.e. we examine the counterfactual "What if we had not done that?"
                treatment_strategy = control_strategy.copy()
                treatment_strategy.set_value(i, int(not capability.value))

                logging.debug(f"  TREATMENT STRATEGY {treatment_strategy.capabilities}")
                logging.debug(f"  OUTCOME {outcome}")
                logging.debug(f"  SAFE RANGE {min} {max}")

                neighbours = list(dag.predecessors(capability.variable))
                neighbours += list(dag.successors(capability.variable))

                if capability.variable == "P402":
                    neighbours.remove("FIT501")
                    neighbours.remove("AIT402")
                    neighbours.remove("FIT401")

                assert len(neighbours) > 0, f"No neighbours for node {capability.variable}"

                fitBLswitch_formula = "xo_t_do ~ time"

                estimator = IPCWEstimator(
                    df,
                    timesteps_per_intervention,
                    control_strategy,
                    treatment_strategy,
                    outcome,
                    min,
                    max,
                    fitBLswitch_formula=fitBLswitch_formula,
                    fitBLTDswitch_formula=f"{fitBLswitch_formula} + {' + '.join(neighbours)}",
                    eligibility=None
                    # elligibility = safe_ranges[outcome].get("eligibility", None),,
                )

                datum = {
                    "outcome": outcome,
                    "attack": attack,
                    "safe_range": (min, max),
                    "control_strategy": control_strategy.capabilities,
                    "treatment_strategy": treatment_strategy.capabilities,
                    "fitBLTDswitch_formula": estimator.fitBLTDswitch_formula,
                }

                hazard_ratio = estimator.estimate_hazard_ratio()
                if hazard_ratio is None:
                    logging.error("  FAILURE: Params was None")
                    continue
                datum["hazard_ratio"] = hazard_ratio.T.to_dict()
                datum["significant"] = (
                    datum["hazard_ratio"]["trtrand"]["95% lower-bound"]
                    < 1
                    < datum["hazard_ratio"]["trtrand"]["95% upper-bound"]
                )

                if adequacy:

                    def estimate_hazard_ratio_parallel(ids):
                        try:
                            par_estimator = deepcopy(estimator)
                            par_estimator.df = par_estimator.df[par_estimator.df["id"].isin(ids)].copy()
                            ratio = par_estimator.estimate_hazard_ratio()
                            return ratio["exp(coef)"]["trtrand"] if ratio is not None else None
                        except np.linalg.LinAlgError:
                            return None
                        except ZeroDivisionError:
                            return None

                    ids = list(set(estimator.df["id"]))

                    pool = Pool()
                    params_repeats = pool.map(
                        estimate_hazard_ratio_parallel,
                        [np.random.choice(ids, len(ids), replace=True) for x in range(num_repeats)],
                    )
                    params_repeats = [float(x) for x in params_repeats if x is not None]
                    datum["params_repeats"] = params_repeats

                    datum["mean_risk_ratio"] = sum(params_repeats) / len(params_repeats)
                    datum["kurtosis"] = float(kurtosis(params_repeats))
                    datum["successes"] = len(params_repeats)
                data.append(datum)
                logging.debug(f"  {datum}")
                with open("logs/output_no_filter_lo_hi_sim_ctf.json", "w") as f:
                    print(jsonpickle.encode(data, indent=2, unpicklable=False), file=f)
