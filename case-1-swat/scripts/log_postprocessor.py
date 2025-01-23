"""
This module checks the estimated sequences of attacks to see which ones still yield a fault.
"""

import json
import sys
from itertools import product

import pandas as pd


def reproduce_fault(interventions: list, minimal: list):
    """
    A facsimile of running SWaT, since we do not have access to the simulator or the system.
    We here say that a fault manifests if the interventions is a superset of the minimal attack.
    This may not be strictly correct, since certain interventions may cancel each other out, but is the best we can do.

    :param interventions: The list of interventions.
    :param minimal: The minimal attack.
    """
    return all(i in interventions for i in minimal)


def build_attack(attack: dict):
    """
    Run the attack and check if the interventions estimated significant lead to a fault.
    Add back interventions until the failure manifests.

    :param attack: dict with details of the attack and causal effect estimate.
    """

    # Already processed
    # if "combinatorial_sim_runs" in attack:
    #     return attack

    if "error" in attack:
        if "treatment_strategies" not in attack:
            assert attack["error"] in [
                "Missing data for control_strategy",
                "No faults observed. P(error) = 0",
            ], f"Bad error {attack['error']} in {attack}"
            # Populate with dummy data if we haven't found anything
            attack["treatment_strategies"] = [{"intervention_index": i} for i in range(len(attack["attack"]))]

    treatment_strategies = [
        (
            treatment_strategy | treatment_strategy["result"]
            if "result" in treatment_strategy
            else treatment_strategy | {"ci_low": [None], "ci_high": [None]}
        )
        for treatment_strategy in attack["treatment_strategies"]
    ]
    treatment_strategies = pd.DataFrame(treatment_strategies)

    # If we've been able to estimate stuff, reorder according to causality
    if "ci_low" in treatment_strategies and "ci_high" in treatment_strategies:
        treatment_strategies["ci_low"] = [c[0] for c in treatment_strategies["ci_low"]]
        treatment_strategies["ci_high"] = [c[0] for c in treatment_strategies["ci_high"]]
        treatment_strategies["significant"] = (treatment_strategies["ci_low"] > 1) | (
            treatment_strategies["ci_high"] < 1
        )
        treatment_strategies = treatment_strategies.loc[~treatment_strategies["significant"]]
        treatment_strategies["below_1"] = 1 - treatment_strategies["ci_low"]
        treatment_strategies["above_1"] = treatment_strategies["ci_high"] - 1
        treatment_strategies["rank"] = treatment_strategies[["below_1", "above_1"]].min(axis=1)
        treatment_strategies.sort_values(["rank", "intervention_index"], inplace=True)
        # TODO
        # Other potential orderings include swapping rank and intervention index
        # and also sorting first by score, and then merging in the unestimated events by time step
    else:
        # else default to greedy minimal
        treatment_strategies.sort_values("intervention_index", inplace=True)

    interventions = []
    for treatment_strategy in attack["treatment_strategies"]:
        if (
            # "result" not in treatment_strategy or
            "result" in treatment_strategy
            and not (treatment_strategy["result"]["ci_low"][0] < 1 < treatment_strategy["result"]["ci_high"][0])
        ):
            interventions.append(attack["attack"][treatment_strategy["intervention_index"]])

    still_fault = reproduce_fault(interventions=interventions, minimal=attack["minimal"])
    attack["pure_estimate_fault"] = still_fault
    attack["estimated_interventions"] = list(interventions)

    simulator_runs = 1

    interventions_to_add = list(treatment_strategies["intervention_index"])
    while not still_fault and interventions_to_add:
        next_intervention = attack["attack"][interventions_to_add.pop(0)]
        if next_intervention in interventions:
            continue
        interventions.append(next_intervention)
        still_fault = reproduce_fault(interventions=interventions, minimal=attack["minimal"])
        simulator_runs += 1
    attack["extended_estimate_fault"] = still_fault
    attack["extended_interventions"] = list(interventions)
    attack["simulator_runs"] = simulator_runs

    # Further minimise the trace by considering all combinations of the remaining interventions, starting with the
    # minimum number of interventions and gradually working back up.
    minimal = dict(enumerate(interventions))
    minimal_keys = sorted(list(minimal.keys()))
    combinatorial_sim_runs = 0
    for mask in sorted(list(product([0, 1], repeat=len(minimal))), key=sum)[1:]:
        candidate = [minimal[k] for m, k in zip(mask, minimal_keys) if m]
        still_fault = reproduce_fault(interventions=candidate, minimal=attack["minimal"])
        combinatorial_sim_runs += 1
        if still_fault:
            minimal = candidate
            break
    attack["minimised_extended_interventions"] = minimal
    attack["combinatorial_sim_runs"] = combinatorial_sim_runs
    attack["greedy_minimal"] = attack["minimal"]

    return attack


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Please provide a JSON log file to process.")

    print(sys.argv[1])
    with open(sys.argv[1]) as f:
        attacks = json.load(f)

    processed_attacks = list(map(build_attack, sorted(attacks, key=lambda a: a["attack_index"])))
    with open(sys.argv[1], "w") as f:
        json.dump(processed_attacks, f)
