import pandas as pd
import statsmodels.formula.api as smf
from numpy import exp, append, nan
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
import matplotlib.pyplot as plt
import argparse
from collections import OrderedDict
from scipy.stats import kurtosis
import networkx as nx
import numpy as np
import jsonpickle
from patsy import dmatrix
import logging
from math import ceil
from copy import deepcopy
import os
import argparse
import lifelines
import time


from causal_testing.estimation.ipcw_estimator import IPCWEstimator
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.data_collection.data_collector import ObservationalDataCollector
from causal_testing.testing.causal_test_outcome import SomeEffect, NoEffect
from causal_testing.testing.base_test_case import BaseTestCase
from causal_testing.testing.causal_test_adequacy import DataAdequacy

parser = argparse.ArgumentParser(prog="water_g", description="Causal testing for the water system.")
parser.add_argument("-a", "--attacks", type=str, help="Path to JSON attacks file.", required=True)
parser.add_argument("-d", "--dag", type=str, help="Path to dag file.", required=True)
parser.add_argument(
    "-s",
    "--safe_ranges",
    type=str,
    help="Path to JSON file defining safe ranges for the output variables.",
    required=True,
)
parser.add_argument(
    "-t", "--timesteps_per_intervention", type=int, help="Timesteps per intervention (defaults to 1).", default=1
)
parser.add_argument(
    "-o",
    "--outfile",
    type=str,
    help="Path to save JSON results file (defaults to `logs/log.json`).",
    default="logs/log.json",
)
parser.add_argument("-b", "--background", nargs="+", help="The background confounders.", default=[])
parser.add_argument(
    "-A",
    "--adequacy",
    help="Specify this flag to record the causal test adequacy. (This will significantly increase the runtime.)",
    action="store_true",
)
parser.add_argument(
    "-S",
    "--silent",
    help="Silence exceptions and store them as part of the result rather than crashing early.",
    action="store_true",
)
parser.add_argument(
    "-i",
    "--attack_index",
    type=int,
    help="The index of the attack to execute.",
    required=False,
)
parser.add_argument(
    "-c",
    "--ci_alpha",
    type=float,
    help="The alpha to use in confidence intervals.",
    default=0.05,
)
parser.add_argument(
    "-T",
    "--total_time",
    type=int,
    help="The total time of the study.",
    default=None,
)
parser.add_argument(
    "-B",
    "--block_size",
    type=int,
    help="The number of interventions to consider at once.",
    default=1,
)
parser.add_argument(
    "--start_time",
    type=int,
    help="The start time.",
    default=0,
)
parser.add_argument("datafile", type=str, help="Path to the long format data file.")

# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(message)s",
#     handlers=[logging.FileHandler("logs/debug.log"), logging.StreamHandler()],
# )


if __name__ == "__main__":
    args = parser.parse_args()

    os.makedirs(os.sep.join(os.path.normpath(args.outfile).split(os.sep)[:-1]), exist_ok=True)
    if args.datafile.endswith(".pqt"):
        df = pd.read_parquet(args.datafile)
    elif args.datafile.endswith(".csv"):
        df = pd.read_csv(args.datafile)
    else:
        raise ValueError("datafile must be .csv or .pqt")
    df = df.loc[df["time"].between(args.start_time, args.total_time)]

    dag = nx.nx_pydot.read_dot(args.dag)
    with open(args.attacks) as f:
        attacks = jsonpickle.decode("".join(f.readlines()))
    with open(args.safe_ranges) as f:
        safe_ranges = jsonpickle.decode("".join(f.readlines()))

    if args.attack_index is not None:
        attacks = [attacks[args.attack_index]]

    for inx, attack in enumerate(attacks, 1):
        print("ATTACK", inx + args.attack_index - 1 if args.attack_index is not None else inx)
        outcome = attack["outcome"]
        logging.debug(f"\nOUTCOME: {outcome}")

        attack["attack"] = list(filter(lambda x: args.start_time < x[0] < args.total_time, attack["attack"]))

        min, max = safe_ranges[outcome]["lo"], safe_ranges[outcome]["hi"]

        attack["safe_range"] = (min, max)
        control_strategy = attack["attack"]
        logging.debug(f"  CONTROL STRATEGY   {control_strategy}")
        attack["control_strategy"] = control_strategy

        if not (~df[outcome].between(min, max)).any():
            logging.error(
                f"  No faults with {outcome}. Cannot perform estimation.\n"
                f"  Observed range [{df[outcome].min()}, {df[outcome].max()}].\n"
                f"  Safe range {safe_ranges[outcome]}"
            )
            attack["error"] = "No faults observed. P(error) = 0"
            continue
        if df[outcome].between(min, max).all():
            logging.error(
                f"  All faults with {outcome}. Cannot perform estimation.\n"
                f"  Observed range [{df[outcome].min()}, {df[outcome].max()}].\n"
                f"  Safe range {safe_ranges[outcome]}"
            )
            attack["error"] = "Only faults observed. P(error) = 1"
            continue

        if any(var not in df for _, var, _ in control_strategy):
            logging.error("  Missing data for control_strategy")
            attack["error"] = "Missing data for control_strategy"
            continue
        if any(var not in dag.nodes for _, var, _ in control_strategy):
            logging.error("  Missing node for control_strategy")
            attack["error"] = "Missing node for control_strategy"
            continue

        indexed_control = list(enumerate(control_strategy))
        for i in range(0, len(control_strategy), args.block_size):
            print(f"Event {i}/{len(control_strategy)}")
            start_time = time.time()
            if "treatment_strategies" not in attack:
                attack["treatment_strategies"] = []
            indexed_capabilities = indexed_control[i : i + args.block_size]
            treatment_strategy = [x[:] for x in control_strategy]
            for i, capability in indexed_capabilities:
                _, variable, value = capability
                # Treatment strategy is the same, but with one capability negated
                # i.e. we examine the counterfactual "What if we had not done that?"
                treatment_strategy[i][2] = int(not value)
            result = {"treatment_strategy": treatment_strategy}
            attack["treatment_strategies"].append(result)

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

            logging.debug(f"  TREATMENT STRATEGY {treatment_strategy}")
            logging.debug(f"  OUTCOME {outcome}")
            logging.debug(f"  SAFE RANGE {min} {max}")

            neighbours = list(dag.predecessors(variable))
            neighbours += list(dag.successors(variable))

            assert len(neighbours) > 0, f"No neighbours for node {variable}"

            if "time" not in args.background:
                args.background.append("time")
            fitBLswitch_formula = f"xo_t_do ~ {' + '.join(args.background)}"
            df["within_safe_range"] = df[outcome].between(min, max)

            if args.silent:
                try:
                    estimation_model = IPCWEstimator(
                        df,
                        args.timesteps_per_intervention,
                        control_strategy,
                        treatment_strategy,
                        outcome,
                        "within_safe_range",
                        fit_bl_switch_formula=fitBLswitch_formula,
                        fit_bltd_switch_formula=f"{fitBLswitch_formula} + {' + '.join(neighbours)}",
                        eligibility=None,
                        alpha=args.ci_alpha,
                        total_time=args.total_time
                        # elligibility = safe_ranges[outcome].get("eligibility", None),
                    )
                except ValueError as e:
                    logging.error(f"ValueError: {e}")
                    result["error"] = f"ValueError: {e}"
                    continue
            else:
                estimation_model = IPCWEstimator(
                    df,
                    args.timesteps_per_intervention,
                    control_strategy,
                    treatment_strategy,
                    outcome,
                    "within_safe_range",
                    fit_bl_switch_formula=fitBLswitch_formula,
                    fit_bltd_switch_formula=f"{fitBLswitch_formula} + {' + '.join(neighbours)}",
                    eligibility=None,
                    alpha=args.ci_alpha,
                    total_time=args.total_time
                    # elligibility = safe_ranges[outcome].get("eligibility", None),
                )

            if args.silent:
                try:
                    causal_test_result = causal_test_case.execute_test(estimation_model, None)
                except np.linalg.LinAlgError:
                    logging.error("LinAlgError when executing test: Could not estimate hazard_ratio.")
                    result["error"] = "LinAlgError when executing test: Could not estimate hazard_ratio."
                    continue
                except lifelines.exceptions.ConvergenceError:
                    logging.error("ConvergenceError when executing test: Could not estimate hazard_ratio.")
                    result["error"] = "ConvergenceError when executing test: Could not estimate hazard_ratio."
                    continue
                except ValueError as e:
                    logging.error(f"ValueError: {e}")
                    result["error"] = f"ValueError: {e}"
                    continue
            else:
                causal_test_result = causal_test_case.execute_test(estimation_model, None)

            assert causal_test_result.test_value.value is not None, "Test result shouldn't be none."

            if causal_test_result.test_value.value is None:
                logging.error("Error: Causal effect not estimated.")
                result["error"] = "Failed to estimate hazard_ratio."
                continue
            estimation_time = time.time()
            result["estimation_time"] = estimation_time - start_time

            if args.adequacy and "error" not in result:
                adequacy_metric = DataAdequacy(causal_test_case, estimation_model, group_by="id")
                adequacy_metric.measure_adequacy()
                causal_test_result.adequacy = adequacy_metric
                adequacy_time = time.time()
                result["adequacy_time"] = adequacy_time - start_time
            result["result"] = causal_test_result.to_dict(json=True)
            result["passed"] = causal_test_case.expected_causal_effect.apply(causal_test_result)
            result["alpha"] = args.ci_alpha

            result["fit_bltd_switch_formula"] = estimation_model.fit_bltd_switch_formula

            logging.debug(f"  {result}")
        with open(args.outfile, "w") as f:
            print(jsonpickle.encode(attacks[:inx], indent=2, unpicklable=False), file=f)
with open(args.outfile, "w") as f:
    print(jsonpickle.encode(attacks, indent=2, unpicklable=False), file=f)
