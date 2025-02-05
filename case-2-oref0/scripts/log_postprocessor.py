"""
This module checks the estimated sequences of attacks to see which ones still yield a fault.
"""

import json
import sys
from itertools import product

import pandas as pd

from fault_finder import reproduce_fault
from aps_digitaltwin.util import intervention_values


def build_attack(attack: dict):
    """
    Run the attack and check if the interventions estimated significant lead to a fault.
    Add back interventions until the failure manifests.

    :param attack: dict with details of the attack and causal effect estimate.
    """

    # Already processed
    # if "combinatorial_sim_runs" in attack:
    #     return attack

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
    treatment_strategies["below_1"] = (1 - treatment_strategies["ci_low"]) / (
        treatment_strategies["ci_high"] - treatment_strategies["ci_low"]
    )
    treatment_strategies["above_1"] = (treatment_strategies["ci_high"] - 1) / (
        treatment_strategies["ci_high"] - treatment_strategies["ci_low"]
    )
    treatment_strategies["rank"] = treatment_strategies[["below_1", "above_1"]].min(axis=1)
    treatment_strategies.sort_values(["rank", "intervention_index"], inplace=True, ascending=[True, False])

    interventions = []
    for treatment_strategy in attack["treatment_strategies"]:
        if (
            # "result" not in treatment_strategy or
            "result" in treatment_strategy
            and not (treatment_strategy["result"]["ci_low"][0] < 1 < treatment_strategy["result"]["ci_high"][0])
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

    interventions_to_add = list(treatment_strategies["intervention_index"])
    while not still_fault and interventions_to_add:
        t, v, _ = attack["attack"][interventions_to_add.pop(0)]
        if (t, v, intervention_values[v]) in interventions:
            continue
        interventions.append((t, v, intervention_values[v]))
        still_fault, _ = reproduce_fault(
            timesteps=499,
            initial_carbs=attack["initial_carbs"],
            initial_bg=attack["initial_bg"],
            initial_iob=attack["initial_iob"],
            constants=attack["constants"],
            interventions=interventions,
        )
        simulator_runs += 1
    interventions.sort()
    attack["extended_estimate_fault"] = still_fault
    attack["extended_interventions"] = list(interventions)
    attack["simulator_runs"] = simulator_runs

    # Apply the greedy heuristic to the tool-minimised trace
    for intervention in sorted(attack["estimated_interventions"]):
        simulator_runs += 1
        still_fault, _ = reproduce_fault(
            timesteps=499,
            initial_carbs=attack["initial_carbs"],
            initial_bg=attack["initial_bg"],
            initial_iob=attack["initial_iob"],
            constants=attack["constants"],
            interventions=[i for i in interventions if i != intervention],
        )
        if still_fault:
            interventions.remove(intervention)
    attack["reduced_extended_interventions"] = list(interventions)
    attack["reduced_simulator_runs"] = simulator_runs

    # Further minimise the trace by considering all combinations of the remaining interventions, starting with the
    # minimum number of interventions and gradually working back up.
    # Don't do this for attacks longer than 20 because it's too expensive
    if len(attack["attack"]) > 20:
        return attack
    minimal = dict(enumerate(interventions))
    minimal_keys = sorted(list(minimal.keys()))
    combinatorial_sim_runs = 0
    for mask in sorted(list(product([0, 1], repeat=len(minimal))), key=sum)[1:]:
        candidate = [minimal[k] for m, k in zip(mask, minimal_keys) if m]
        still_fault, _ = reproduce_fault(
            timesteps=499,
            initial_carbs=attack["initial_carbs"],
            initial_bg=attack["initial_bg"],
            initial_iob=attack["initial_iob"],
            constants=attack["constants"],
            interventions=candidate,
        )
        combinatorial_sim_runs += 1
        if still_fault:
            minimal = candidate
            break
    attack["minimised_extended_interventions"] = minimal
    attack["combinatorial_sim_runs"] = combinatorial_sim_runs

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
