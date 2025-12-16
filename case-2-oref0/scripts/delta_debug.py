"""
Delta debugging for the oref0 case study.

Note: You must `cd` into the "case-2-oref0" directory before running this one.
Otherwise oref0 won't find the config data it needs.
Then run as `python scripts/delta_debug.py successful_attacks.json`
"""

import sys
import json
from multiprocessing import Pool

from fault_finder import reproduce_fault
from aps_digitaltwin.util import intervention_values

from plotting.ddmin import ddmin


def dd_reproduce_fault(
    interventions: list, initial_carbs: float, initial_bg: float, initial_iob: float, constants: list, failure: str
):
    return (
        reproduce_fault(
            timesteps=499,
            initial_carbs=initial_carbs,
            initial_bg=initial_bg,
            initial_iob=initial_iob,
            interventions=[(t, v, intervention_values[v]) for t, v, _ in interventions],
            constants=constants,
        )[0]
        == failure
    )


assert len(sys.argv) == 2, "Please provide the attack file"

with open(sys.argv[1]) as f:
    attacks = json.load(f)


def parallell_ddmin(attack):
    return ddmin(
        dd_reproduce_fault,
        attack["attack"],
        attack["initial_carbs"],
        attack["initial_bg"],
        attack["initial_iob"],
        attack["constants"],
        attack["failure"],
    )


with Pool() as pool:
    results = pool.map(parallell_ddmin, attacks)

for (minimal, executions), attack in zip(results, attacks):
    attack["ddmin"] = minimal
    attack["ddmin_executions"] = executions

with open(sys.argv[1], "w") as f:
    json.dump(attacks, f)
