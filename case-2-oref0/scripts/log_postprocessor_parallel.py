"""
This module checks the estimated sequences of attacks to see which ones still yield a fault.
"""

import json
import argparse
from itertools import product
from multiprocessing import Pool
from functools import partial

import pandas as pd
from tqdm import tqdm

from fault_finder import reproduce_fault
from aps_digitaltwin.util import intervention_values


def check_fault(candidate, attack):
    still_fault, _ = reproduce_fault(
        timesteps=499,
        initial_carbs=attack["initial_carbs"],
        initial_bg=attack["initial_bg"],
        initial_iob=attack["initial_iob"],
        constants=attack["constants"],
        interventions=candidate,
    )
    return candidate, still_fault


def build_attack(attack: dict, threads: int = None):
    """
    Run the attack and check if the interventions estimated significant lead to a fault.
    Add back interventions until the failure manifests.

    :param attack: Dictionary with details of the attack and causal effect estimate.
    :param threads: Number of threads to use if executing combinations in parallel
    """

    # Already processed
    if "combinatorial_sim_runs" in attack:
        return attack

    treatment_strategies = [
        (
            treatment_strategy | treatment_strategy["result"]
            if "result" in treatment_strategy
            else treatment_strategy | {"ci_low": [None], "ci_high": [None]}
        )
        for treatment_strategy in attack["treatment_strategies"]
    ]
    treatment_strategies = pd.DataFrame(treatment_strategies)

    treatment_strategies["ci_low"] = [c[0] for c in treatment_strategies["ci_low"]]
    treatment_strategies["ci_high"] = [c[0] for c in treatment_strategies["ci_high"]]
    treatment_strategies["significant"] = (treatment_strategies["ci_low"] > 1) | (treatment_strategies["ci_high"] < 1)
    treatment_strategies = treatment_strategies.loc[~treatment_strategies["significant"]]
    treatment_strategies["below_1"] = 1 - treatment_strategies["ci_low"]
    treatment_strategies["above_1"] = treatment_strategies["ci_high"] - 1
    treatment_strategies["rank"] = treatment_strategies[["below_1", "above_1"]].min(axis=1)
    treatment_strategies.sort_values("rank", inplace=True)

    interventions = []
    for treatment_strategy in attack["treatment_strategies"]:
        if "result" not in treatment_strategy or not (
            treatment_strategy["result"]["ci_low"][0] < 1 < treatment_strategy["result"]["ci_high"][0]
        ):
            interventions.append(attack["attack"][treatment_strategy["intervention_index"]])
    interventions = [(t, v, intervention_values[v]) for t, v, _ in interventions]

    still_fault, _ = reproduce_fault(
        timesteps=499,
        initial_carbs=attack["initial_carbs"],
        initial_bg=attack["initial_bg"],
        initial_iob=attack["initial_iob"],
        interventions=interventions,
        constants=attack["constants"],
    )
    attack["pure_estimate_fault"] = still_fault
    attack["estimated_interventions"] = list(interventions)

    simulator_runs = 1

    # Add back interventions in the correct order until a failure is observed
    interventions_to_add = list(treatment_strategies["intervention_index"])

    if threads is None:
        while not still_fault and interventions_to_add:
            t, v, _ = attack["attack"][interventions_to_add.pop(0)]
            interventions.append((t, v, intervention_values[v]))
            _, still_fault = check_fault(interventions, attack)
            simulator_runs += 1
    else:
        runs = []
        for intervention in interventions_to_add:
            t, v, _ = attack["attack"][intervention]
            interventions.append((t, v, intervention_values[v]))
            runs.append(list(interventions))
        with Pool(threads) as pool:
            results = pool.map(partial(check_fault, attack=attack), runs)
        for _, still_fault in results:
            simulator_runs += 1
            if still_fault:
                break
    print("extended_interventions", len(list(interventions)))
    print("simulator_runs", simulator_runs)

    attack["extended_estimate_fault"] = still_fault
    attack["extended_interventions"] = list(interventions)
    attack["simulator_runs"] = simulator_runs

    # Further minimise the trace by considering all combinations of the remaining interventions, starting with the
    # minimum number of interventions and gradually working back up.
    minimal = dict(enumerate(interventions))
    minimal_keys = sorted(list(minimal.keys()))

    combinations = sorted(list(product([0, 1], repeat=len(minimal))), key=sum)[1:]
    combinatorial_sim_runs = 0
    if threads is None:
        for mask in combinations:
            candidate = [minimal[k] for m, k in zip(mask, minimal_keys) if m]
            candidate, still_fault = check_fault(candidate, attack)
            combinatorial_sim_runs += 1
            if still_fault:
                minimal = candidate
                break
    else:
        # Split into blocks to minimise wasted computation
        blocks = [combinations[i : i + threads] for i in range(0, len(combinations), threads)]
        print(len(combinations), "combinations")
        print(len(blocks), "blocks")
        for block in tqdm(blocks):
            with Pool(threads) as pool:
                results = pool.map(
                    partial(check_fault, attack=attack),
                    [[minimal[k] for m, k in zip(mask, minimal_keys) if m] for mask in block],
                )
            for candidate, still_fault in results:
                combinatorial_sim_runs += 1
                if still_fault:
                    minimal = candidate
                    break
            else:
                continue  # only executed if the inner loop did NOT break
            break  # only executed if the inner loop DID break

    attack["minimised_extended_interventions"] = minimal
    attack["combinatorial_sim_runs"] = combinatorial_sim_runs

    return attack


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Log Postprocessor",
        description="Postprocess the causal test results to get the minimal attack traces.",
    )
    parser.add_argument("-t", "--threads", type=int, help="The number of threads to use in parallel excution mode.")
    parser.add_argument("test_results", help="The location of the causal test results JSON log.")
    args = parser.parse_args()

    with open(args.test_results) as f:
        attacks = json.load(f)

    processed_attacks = [
        build_attack(attack, args.threads) for attack in sorted(attacks, key=lambda a: a["attack_index"])
    ]
    with open(args.test_results, "w") as f:
        json.dump(processed_attacks, f)
