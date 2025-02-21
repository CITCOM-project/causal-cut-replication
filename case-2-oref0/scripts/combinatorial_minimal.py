"""
This module computes the full combinatorial minimum of each successful attack.
"""

import json
import sys
from itertools import product, combinations

from multiprocessing import Pool
from tqdm import tqdm

from fault_finder import reproduce_fault
from aps_digitaltwin.util import intervention_values


def check_fault(attack, interventions):
    still_fault, _ = reproduce_fault(
        timesteps=499,
        initial_carbs=attack["initial_carbs"],
        initial_bg=attack["initial_bg"],
        initial_iob=attack["initial_iob"],
        constants=attack["constants"],
        interventions=interventions,
    )
    return still_fault


def minimise(attack):
    interventions = [(t, v, intervention_values[v]) for t, v, _ in attack["attack"]]

    if len(attack["attack"]) > 20:
        return attack
    minimal = dict(enumerate(interventions))
    minimal_keys = sorted(list(minimal.keys()))
    combinatorial_sim_runs = 0
    for mask in sorted(list(product([0, 1], repeat=len(minimal))), key=sum)[1:]:
        candidate = [minimal[k] for m, k in zip(mask, minimal_keys) if m]
        still_fault = check_fault(attack, candidate)
        combinatorial_sim_runs += 1
        if still_fault:
            minimal = candidate
            break
    attack["combinatorial_minimal"] = minimal
    attack["combinatorial_minimal_sim_runs"] = combinatorial_sim_runs
    return attack

def length_filter(a):
    return len(a["attack"]) < 20


if __name__ == "__main__":
    THREADS = 15
    if len(sys.argv) != 2:
        raise ValueError("Please provide a JSON log file to process.")

    print(sys.argv[1])
    with open(sys.argv[1]) as f:
        attacks = json.load(f)

    # Use this for the ones where the combinatorial minimal runs are quick to find
    with Pool(THREADS) as pool:
        processed_attacks = list(pool.map(minimise, filter(length_filter, attacks)))

    # Use this for the ones where the combinatorial minimal runs are slow to find.
    for attack in attacks:
        if length_filter(attack):
            continue
        print(attack["attack"])
        breakout = False
        for attack_length in range(1, len(attack["minimal"])):
            print(f"Trying {attack_length} interventions")
            interventions_to_try = list(combinations(attack["attack"], attack_length))
            chunks = [interventions_to_try[i : i + THREADS] for i in range(0, len(interventions_to_try), THREADS)]
            with Pool(THREADS) as pool:
                for chunk in tqdm(chunks):
                    faults = pool.starmap(check_fault, [(attack, interventions) for interventions in chunk])
                    if any(faults):
                        attack["combinatorial_minimal"] = list(filter(lambda x: x[1], zip(faults, chunk)))[0][0]
                        # attack["combinatorial_minimal_sim_runs"] = combinatorial_sim_runs
                        breakout = True
                        break
            attack["combinatorial_minimal"] = attack["minimal"]
            if breakout:
                break
        break

    with open(sys.argv[1], "w") as f:
        json.dump(processed_attacks, f)
