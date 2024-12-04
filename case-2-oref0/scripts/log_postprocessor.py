"""
This module checks the estimated sequences of attacks to see which ones still yield a fault.
"""

import json
import sys
import pandas as pd
from uuid import uuid4
from itertools import product

from fault_finder import reproduce_fault
from aps_digitaltwin.util import intervention_values


def check_if_still_fault(attack: dict):
    """
    Run the attack and check if the interventions estimated significant lead to a fault.

    :param attack: dict with details of the attack and causal effect estimate.
    """
    interventions = []
    for treatment_strategy in attack["treatment_strategies"]:
        if "result" not in treatment_strategy or not (
            treatment_strategy["result"]["ci_low"][0] < 1 < treatment_strategy["result"]["ci_high"][0]
        ):
            interventions.append(attack["attack"][treatment_strategy["intervention_index"]])
    interventions = [(t, v, intervention_values[v]) for t, v, _ in interventions]
    still_fault, _ = reproduce_fault(
        timesteps=500,
        initial_carbs=attack["initial_carbs"],
        initial_bg=attack["initial_bg"],
        initial_iob=attack["initial_iob"],
        interventions=interventions,
        constants=attack["constants"],
        save_path=f"/tmp/{uuid4()}.csv",
    )
    return still_fault, len(interventions)


def build_attack(attack: dict):
    """
    Run the attack and check if the interventions estimated significant lead to a fault.
    Add back interventions until the failure manifests.

    :param attack: dict with details of the attack and causal effect estimate.
    """
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

    interventions_to_add = list(treatment_strategies["intervention_index"])
    while not still_fault and interventions_to_add:
        t, v, _ = attack["attack"][interventions_to_add.pop(0)]
        interventions.append((t, v, intervention_values[v]))
        still_fault, _ = reproduce_fault(
            timesteps=499,
            initial_carbs=attack["initial_carbs"],
            initial_bg=attack["initial_bg"],
            initial_iob=attack["initial_iob"],
            constants=attack["constants"],
            interventions=interventions,
        )
    attack["extended_estimate_fault"] = still_fault
    attack["extended_interventions"] = list(interventions)

    # Further minimise the trace by considering all combinations of the remaining interventions, starting with the
    # minimum number of interventions and gradually working back up. This step is necessary because the greedy
    # method above can sometimes mark spurious events as necessary causes, presumably due to interactions between
    # interventions. However, it is too expensive to start off simply by considering combinations.
    minimal = dict(enumerate(interventions))
    minimal_keys = sorted(list(minimal.keys()))
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
        if still_fault:
            minimal = candidate
            break
    attack["minimised_extended_interventions"] = minimal

    return attack


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Please provide a JSON log file to process.")

    with open(sys.argv[1]) as f:
        attacks = json.load(f)

    processed_attacks = list(map(build_attack, sorted(attacks, key=lambda a: a["attack_index"])))
    with open(sys.argv[1], "w") as f:
        json.dump(processed_attacks, f)
