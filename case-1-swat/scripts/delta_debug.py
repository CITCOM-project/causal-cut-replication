"""
Delta debugging for the SWaT case study.
You can run this one from the root directory of the repo.
Run as `python case-1-swat/scripts/delta_debug.py case-1-swat/successful_attacks_mutated.json`
"""

import sys
import json

from log_postprocessor import reproduce_fault
from plotting.ddmin import ddmin

assert len(sys.argv) == 2, "Please provide the attack file"

with open(sys.argv[1]) as f:
    attacks = json.load(f)


for attack in attacks:

    minimal, executions = ddmin(reproduce_fault, attack["attack"], attack["minimal"])

    attack["ddmin"] = minimal
    attack["ddmin_executions"] = executions

with open(sys.argv[1], "w") as f:
    json.dump(attacks, f)
