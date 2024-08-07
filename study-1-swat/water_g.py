import pandas as pd
import statsmodels.formula.api as smf
from numpy import exp, append, nan
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
import matplotlib.pyplot as plt
import argparse
from safe_ranges import safe_ranges
from collections import OrderedDict
from scipy.stats import kurtosis
import networkx as nx
import numpy as np
from multiprocessing import Pool
import jsonpickle
from patsy import dmatrix
import logging
from math import ceil
from copy import deepcopy
import os
import argparse


from causal_testing.testing.estimators import IPCWEstimator
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.data_collection.data_collector import ObservationalDataCollector
from causal_testing.testing.causal_test_outcome import SomeEffect, NoEffect
from causal_testing.testing.base_test_case import BaseTestCase
from causal_testing.specification.capabilities import TreatmentSequence
from causal_testing.testing.causal_test_adequacy import DataAdequacy

parser = argparse.ArgumentParser(
    prog="water_g", description="Causal testing for the water system."
)
parser.add_argument("-o", "--outfile", type=str, help="Path JSON results file.")
parser.add_argument("datafile", type=str, help="Path to the long format data file.")

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    handlers=[logging.FileHandler("logs/debug.log"), logging.StreamHandler()],
)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.datafile.endswith(".pqt"):
        df = pd.read_parquet(args.datafile)
    elif args.datafile.endswith(".csv"):
        df = pd.read_csv(args.datafile)
    else:
        raise ValueError("datafile must be .csv or .pqt")
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

        min, max = safe_ranges[outcome]["lo"], safe_ranges[outcome]["hi"]

        datum = {
            "outcome": outcome,
            "attack": attack,
            "safe_range": (min, max),
        }
        if not (~df[outcome].between(min, max)).any():
            logging.error(
                f"  No faults with {outcome}. Cannot perform estimation.\n"
                f"  Observed range [{df[outcome].min()}, {df[outcome].max()}].\n"
                f"  Safe range {safe_ranges[outcome]}"
            )
            datum["error"] = "No faults observed. P(error) = 0"
            data.append(datum)
            continue
        if df[outcome].between(min, max).all():
            logging.error(
                f"  All faults with {outcome}. Cannot perform estimation.\n"
                f"  Observed range [{df[outcome].min()}, {df[outcome].max()}].\n"
                f"  Safe range {safe_ranges[outcome]}"
            )
            datum["error"] = "Only faults observed. P(error) = 1"
            data.append(datum)
            continue

        for capabilities in attacks:
            control_strategy = TreatmentSequence(
                timesteps_per_intervention, capabilities
            )
            logging.debug(f"  CONTROL STRATEGY   {control_strategy.capabilities}")
            datum["control_strategy"] = control_strategy.capabilities

            if any(c.variable not in df for c in control_strategy.capabilities):
                logging.error("  Missing data for control_strategy")
                datum["error"] = "Missing data for control_strategy"
                data.append(datum)
                continue
            if any(c.variable not in dag.nodes for c in control_strategy.capabilities):
                logging.error("  Missing node for control_strategy")
                datum["error"] = "Missing node for control_strategy"
                data.append(datum)
                break

            for i, capability in enumerate(control_strategy.capabilities):
                # Treatment strategy is the same, but with one capability negated
                # i.e. we examine the counterfactual "What if we had not done that?"
                treatment_strategy = control_strategy.copy()
                treatment_strategy.set_value(i, int(not capability.value))
                datum["treatment_strategy"] = treatment_strategy.capabilities

                base_test_case = BaseTestCase(
                    treatment_variable=control_strategy,
                    outcome_variable=outcome,
                    effect="temporal",
                )

                causal_test_case = CausalTestCase(
                    base_test_case=base_test_case,
                    expected_causal_effect=SomeEffect(),
                    control_value=control_strategy,
                    treatment_value=treatment_strategy,
                    estimate_type="hazard_ratio",
                )

                logging.debug(f"  TREATMENT STRATEGY {treatment_strategy.capabilities}")
                logging.debug(f"  OUTCOME {outcome}")
                logging.debug(f"  SAFE RANGE {min} {max}")

                neighbours = list(dag.predecessors(capability.variable))
                neighbours += list(dag.successors(capability.variable))

                if capability.variable == "P402":
                    neighbours.remove("FIT501")
                    neighbours.remove("AIT402")
                    neighbours.remove("FIT401")

                assert (
                    len(neighbours) > 0
                ), f"No neighbours for node {capability.variable}"

                fitBLswitch_formula = "xo_t_do ~ time"

                df["within_safe_range"] = df[outcome].between(min, max)

                estimation_model = IPCWEstimator(
                    df,
                    timesteps_per_intervention,
                    control_strategy,
                    treatment_strategy,
                    outcome,
                    "within_safe_range",
                    fit_bl_switch_formula=fitBLswitch_formula,
                    fit_bltd_switch_formula=f"{fitBLswitch_formula} + {' + '.join(neighbours)}",
                    eligibility=None,
                    # elligibility = safe_ranges[outcome].get("eligibility", None),
                )

                causal_test_result = causal_test_case.execute_test(
                    estimation_model, None
                )

                if causal_test_result.test_value.value is None:
                    logging.error("Error: Causal effect not estimated.")
                    datum["error"] = "Failed to estimate hazard_ratio."
                    data.append(datum)
                    continue

                if adequacy:
                    adequacy_metric = DataAdequacy(
                        causal_test_case, estimation_model, group_by="id"
                    )
                    adequacy_metric.measure_adequacy()
                    causal_test_result.adequacy = adequacy_metric
                datum = datum | causal_test_result.to_dict(json=True)
                datum["passed"] = causal_test_case.expected_causal_effect.apply(
                    causal_test_result
                )

                datum["fit_bltd_switch_formula"] = (
                    estimation_model.fit_bltd_switch_formula
                )

                data.append(datum)
                logging.debug(f"  {datum}")
                with open(args.outfile, "w") as f:
                    print(jsonpickle.encode(data, indent=2, unpicklable=False), file=f)
    with open(args.outfile, "w") as f:
        print(jsonpickle.encode(data, indent=2, unpicklable=False), file=f)
